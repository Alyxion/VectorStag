//! feDropShadow - create drop shadow effect

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;
use super::blur_utils::integral_query;

pub fn fe_drop_shadow_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    dx: f32,
    dy: f32,
    std_dev_x: f32,
    std_dev_y: f32,
    flood_r: f32, flood_g: f32, flood_b: f32, flood_a: f32,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    let dx_i = dx.round() as i32;
    let dy_i = dy.round() as i32;

    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    let mut alpha = vec![0.0f32; h * w];
    for y in 0..h {
        let src_y = y as i32 - dy_i;
        if src_y < 0 || src_y >= h as i32 { continue; }
        for x in 0..w {
            let src_x = x as i32 - dx_i;
            if src_x < 0 || src_x >= w as i32 { continue; }
            let a = src[[src_y as usize, src_x as usize, 3]];
            alpha[y * w + x] = a * flood_a;
        }
    }

    if std_dev_x >= 0.5 || std_dev_y >= 0.5 {
        // Box blur approximation: radius = (sqrt(12σ²/n + 1) - 1) / 2
        let box_radius_x = (((12.0 * std_dev_x * std_dev_x / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize;
        let box_radius_y = (((12.0 * std_dev_y * std_dev_y / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize;

        let mut current = &mut alpha;
        let mut next = &mut vec![0.0f32; h * w];

        for _ in 0..3 {
            box_blur_single_channel(current, next, w, h, box_radius_x, box_radius_y);
            std::mem::swap(&mut current, &mut next);
        }
    }

    let mut dst = Array3::<f32>::zeros((h, w, 4));
    for y in 0..h {
        for x in 0..w {
            let shadow_a = alpha[y * w + x];
            // Shadow
            if shadow_a > 0.0 {
                dst[[y, x, 0]] = flood_r;
                dst[[y, x, 1]] = flood_g;
                dst[[y, x, 2]] = flood_b;
                dst[[y, x, 3]] = shadow_a;
            }
            
            // Composite source over shadow
            let src_a = src[[y, x, 3]];
            if src_a > 0.0 {
                let inv_src_a = 1.0 - src_a;
                dst[[y, x, 0]] = src[[y, x, 0]] + dst[[y, x, 0]] * inv_src_a;
                dst[[y, x, 1]] = src[[y, x, 1]] + dst[[y, x, 1]] * inv_src_a;
                dst[[y, x, 2]] = src[[y, x, 2]] + dst[[y, x, 2]] * inv_src_a;
                dst[[y, x, 3]] = src_a + dst[[y, x, 3]] * inv_src_a;
            }
        }
    }
    dst
}

fn box_blur_single_channel(src: &[f32], dst: &mut [f32], w: usize, h: usize, rx: usize, ry: usize) {
    if w == 0 || h == 0 { return; }
    let integral = compute_integral_image_single(src, w, h);
    let iw = w + 1;
    let area_x = (2 * rx + 1) as f64;
    let area_y = (2 * ry + 1) as f64;
    let inv_area = 1.0 / (area_x * area_y);

    for y in 0..h {
        let y1 = if y >= ry { y - ry } else { 0 };
        let y2 = if y + ry < h { y + ry + 1 } else { h };
        for x in 0..w {
            let x1 = if x >= rx { x - rx } else { 0 };
            let x2 = if x + rx < w { x + rx + 1 } else { w };
            let sum = integral_query(&integral, iw, x1, y1, x2, y2);
            let count = ((x2 - x1) * (y2 - y1)) as f64;
            // Adjust for edge cases if necessary, or just use box area approx
            dst[y * w + x] = (sum / count) as f32; // Normalizing by actual kernel overlap area
        }
    }
}

fn compute_integral_image_single(src: &[f32], w: usize, h: usize) -> Vec<f64> {
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

pub fn fe_drop_shadow_impl(
    src: &ndarray::ArrayView3<u8>,
    dx: f32,
    dy: f32,
    std_dev_x: f32,
    std_dev_y: f32,
    flood_r: u8, flood_g: u8, flood_b: u8, flood_a: u8,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    let dx_i = dx.round() as i32;
    let dy_i = dy.round() as i32;

    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    let mut alpha = vec![0.0f32; h * w];
    for y in 0..h {
        let src_y = y as i32 - dy_i;
        if src_y < 0 || src_y >= h as i32 { continue; }
        for x in 0..w {
            let src_x = x as i32 - dx_i;
            if src_x < 0 || src_x >= w as i32 { continue; }
            let a = src[[src_y as usize, src_x as usize, 3]] as f32;
            alpha[y * w + x] = a * (flood_a as f32 / 255.0);
        }
    }

    if std_dev_x >= 0.5 || std_dev_y >= 0.5 {
        // Box blur approximation: radius = (sqrt(12σ²/n + 1) - 1) / 2
        let box_radius_x = (((12.0 * std_dev_x * std_dev_x / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize;
        let box_radius_y = (((12.0 * std_dev_y * std_dev_y / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize;

        for _ in 0..3 {
            if box_radius_x > 0 {
                let mut next = vec![0.0f32; h * w];
                for y in 0..h {
                    let mut sum = 0.0f32;
                    for i in 0..box_radius_x.min(w) + 1 {
                        sum += alpha[y * w + i];
                    }
                    let mut left = 0i32 - box_radius_x as i32;
                    let mut right = box_radius_x as i32;

                    for x in 0..w {
                        if right < w as i32 && right > x as i32 {
                            sum += alpha[y * w + right as usize];
                        }
                        let count = (right.min(w as i32 - 1) - left.max(0) + 1) as f32;
                        next[y * w + x] = sum / count;
                        if left >= 0 {
                            sum -= alpha[y * w + left as usize];
                        }
                        left += 1;
                        right += 1;
                    }
                }
                alpha = next;
            }

            if box_radius_y > 0 {
                let mut next = vec![0.0f32; h * w];
                for x in 0..w {
                    let mut sum = 0.0f32;
                    for i in 0..box_radius_y.min(h) + 1 {
                        sum += alpha[i * w + x];
                    }
                    let mut top = 0i32 - box_radius_y as i32;
                    let mut bottom = box_radius_y as i32;

                    for y in 0..h {
                        if bottom < h as i32 && bottom > y as i32 {
                            sum += alpha[bottom as usize * w + x];
                        }
                        let count = (bottom.min(h as i32 - 1) - top.max(0) + 1) as f32;
                        next[y * w + x] = sum / count;
                        if top >= 0 {
                            sum -= alpha[top as usize * w + x];
                        }
                        top += 1;
                        bottom += 1;
                    }
                }
                alpha = next;
            }
        }
    }

    let mut dst = Array3::<u8>::zeros((h, w, 4));
    for y in 0..h {
        for x in 0..w {
            let shadow_a = alpha[y * w + x] / 255.0;
            let src_a = src[[y, x, 3]] as f32 / 255.0;

            let out_a = src_a + shadow_a * (1.0 - src_a);

            if out_a > 0.0 {
                for c in 0..3 {
                    let src_c = src[[y, x, c]] as f32 / 255.0;
                    let shadow_c = match c { 0 => flood_r, 1 => flood_g, _ => flood_b } as f32 / 255.0;
                    let out_c = (src_c * src_a + shadow_c * shadow_a * (1.0 - src_a)) / out_a;
                    dst[[y, x, c]] = (out_c * 255.0).clamp(0.0, 255.0) as u8;
                }
                dst[[y, x, 3]] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// feDropShadow - create drop shadow effect
#[pyfunction]
pub fn fe_drop_shadow<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    dx: f32,
    dy: f32,
    std_dev_x: f32,
    std_dev_y: f32,
    flood_r: u8, flood_g: u8, flood_b: u8, flood_a: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_drop_shadow_impl(&arr, dx, dy, std_dev_x, std_dev_y, flood_r, flood_g, flood_b, flood_a).into_pyarray(py)
}

