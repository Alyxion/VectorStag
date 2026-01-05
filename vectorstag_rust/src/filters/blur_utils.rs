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
