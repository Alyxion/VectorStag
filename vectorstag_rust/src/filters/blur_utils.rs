//! Box blur utilities for Gaussian blur approximation

/// Compute integral image for a single channel (from 4-channel interleaved data)
#[inline]
pub fn compute_integral_image(src: &[f32], w: usize, h: usize, channel: usize) -> Vec<f64> {
    let mut integral = vec![0.0f64; (w + 1) * (h + 1)];
    let iw = w + 1;

    for y in 0..h {
        let mut row_sum = 0.0f64;
        for x in 0..w {
            row_sum += src[(y * w + x) * 4 + channel] as f64;
            integral[(y + 1) * iw + (x + 1)] = row_sum + integral[y * iw + (x + 1)];
        }
    }
    integral
}

/// Compute integral image for single-channel data
pub fn compute_integral_image_single(src: &[f32], w: usize, h: usize) -> Vec<f64> {
    let mut integral = vec![0.0f64; (w + 1) * (h + 1)];
    let iw = w + 1;
    for y in 0..h {
        let mut row_sum = 0.0f64;
        for x in 0..w {
            row_sum += src[y * w + x] as f64;
            integral[(y + 1) * iw + (x + 1)] = row_sum + integral[y * iw + (x + 1)];
        }
    }
    integral
}

#[inline]
pub fn integral_query(integral: &[f64], iw: usize, x1: usize, y1: usize, x2: usize, y2: usize) -> f64 {
    integral[y2 * iw + x2] - integral[y1 * iw + x2] - integral[y2 * iw + x1] + integral[y1 * iw + x1]
}

/// Box blur using integral images (4-channel RGBA)
#[inline]
pub fn box_blur_integral(src: &[f32], dst: &mut [f32], w: usize, h: usize, rx: usize, ry: usize) {
    if w == 0 || h == 0 { return; }

    let integral_r = compute_integral_image(src, w, h, 0);
    let integral_g = compute_integral_image(src, w, h, 1);
    let integral_b = compute_integral_image(src, w, h, 2);
    let integral_a = compute_integral_image(src, w, h, 3);

    let iw = w + 1;

    for y in 0..h {
        let y1 = if y >= ry { y - ry } else { 0 };
        let y2 = (y + ry + 1).min(h);

        for x in 0..w {
            let x1 = if x >= rx { x - rx } else { 0 };
            let x2 = (x + rx + 1).min(w);

            let area = ((x2 - x1) * (y2 - y1)) as f64;
            let inv_area = if area > 0.0 { 1.0 / area } else { 0.0 };

            let idx = (y * w + x) * 4;
            dst[idx]     = (integral_query(&integral_r, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 1] = (integral_query(&integral_g, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 2] = (integral_query(&integral_b, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 3] = (integral_query(&integral_a, iw, x1, y1, x2, y2) * inv_area) as f32;
        }
    }
}

/// Box blur for single-channel data
pub fn box_blur_single_channel(src: &[f32], dst: &mut [f32], w: usize, h: usize, rx: usize, ry: usize) {
    if w == 0 || h == 0 { return; }
    let integral = compute_integral_image_single(src, w, h);
    let iw = w + 1;

    for y in 0..h {
        let y1 = if y >= ry { y - ry } else { 0 };
        let y2 = if y + ry < h { y + ry + 1 } else { h };
        for x in 0..w {
            let x1 = if x >= rx { x - rx } else { 0 };
            let x2 = if x + rx < w { x + rx + 1 } else { w };
            let sum = integral_query(&integral, iw, x1, y1, x2, y2);
            let count = ((x2 - x1) * (y2 - y1)) as f64;
            dst[y * w + x] = (sum / count) as f32;
        }
    }
}

/// Calculate box radius from standard deviation for 3-pass box blur approximation
/// Formula: box_width = sqrt(12σ²/n + 1), radius = (width - 1) / 2
#[inline]
pub fn std_dev_to_box_radius(std_dev: f32) -> usize {
    (((12.0 * std_dev * std_dev / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize
}

/// Alpha-weighted box blur using integral images (4-channel RGBA, premultiplied)
/// RGB channels use alpha-weighted averaging: sum(R*A) / sum(A)
/// This prevents transparent pixels from darkening the result
#[inline]
pub fn box_blur_integral_alpha_weighted(src: &[f32], dst: &mut [f32], w: usize, h: usize, rx: usize, ry: usize) {
    if w == 0 || h == 0 { return; }

    // For alpha-weighted blur, we need:
    // - Integral of R*A, G*A, B*A (premultiplied RGB - already the case)
    // - Integral of A (for weighting)
    // Then: blurred_R = sum(R*A) / sum(A), blurred_A = sum(A) / count

    // src is already in premultiplied form, so src[R] = R*A
    let integral_ra = compute_integral_image(src, w, h, 0);  // sum of R*A
    let integral_ga = compute_integral_image(src, w, h, 1);  // sum of G*A
    let integral_ba = compute_integral_image(src, w, h, 2);  // sum of B*A
    let integral_a = compute_integral_image(src, w, h, 3);   // sum of A

    let iw = w + 1;

    for y in 0..h {
        let y1 = if y >= ry { y - ry } else { 0 };
        let y2 = (y + ry + 1).min(h);

        for x in 0..w {
            let x1 = if x >= rx { x - rx } else { 0 };
            let x2 = (x + rx + 1).min(w);

            let area = ((x2 - x1) * (y2 - y1)) as f64;
            let sum_a = integral_query(&integral_a, iw, x1, y1, x2, y2);

            let idx = (y * w + x) * 4;

            // Alpha: regular area average
            let blurred_a = if area > 0.0 { sum_a / area } else { 0.0 };

            // RGB: alpha-weighted average (output is premultiplied)
            // We have sum(R*A) and sum(A), we want blurred_R*blurred_A
            // For proper alpha-weighted blur: blurred_R = sum(R*A) / sum(A)
            // Then premultiplied output: blurred_R * blurred_A
            if sum_a > 1e-10 {
                let sum_ra = integral_query(&integral_ra, iw, x1, y1, x2, y2);
                let sum_ga = integral_query(&integral_ga, iw, x1, y1, x2, y2);
                let sum_ba = integral_query(&integral_ba, iw, x1, y1, x2, y2);

                // Straight alpha colors: R = sum(R*A)/sum(A)
                let r = sum_ra / sum_a;
                let g = sum_ga / sum_a;
                let b = sum_ba / sum_a;

                // Output in premultiplied form
                dst[idx]     = (r * blurred_a) as f32;
                dst[idx + 1] = (g * blurred_a) as f32;
                dst[idx + 2] = (b * blurred_a) as f32;
            } else {
                dst[idx] = 0.0;
                dst[idx + 1] = 0.0;
                dst[idx + 2] = 0.0;
            }
            dst[idx + 3] = blurred_a as f32;
        }
    }
}
