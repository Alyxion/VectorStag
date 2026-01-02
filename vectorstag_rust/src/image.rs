//! Image operations: compositing, resizing, color space conversion

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;
use rayon::prelude::*;

/// Alpha blend a single pixel (inline for SIMD-friendly code)
#[inline(always)]
pub fn blend_pixel(dst: &mut [u8], src: &[u8]) {
    let src_a = src[3] as u32;
    if src_a == 0 { return; }

    if src_a == 255 {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 255;
    } else {
        let dst_a = dst[3] as u32;
        let inv_src_a = 255 - src_a;
        let out_a = src_a + (dst_a * inv_src_a / 255);

        if out_a > 0 {
            dst[0] = ((src[0] as u32 * src_a + dst[0] as u32 * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
            dst[1] = ((src[1] as u32 * src_a + dst[1] as u32 * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
            dst[2] = ((src[2] as u32 * src_a + dst[2] as u32 * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
            dst[3] = out_a.min(255) as u8;
        }
    }
}

/// Alpha composite source onto destination in-place at given offset
/// Uses Porter-Duff over operator: out = src + dst * (1 - src_alpha)
#[pyfunction]
pub fn alpha_composite_inplace<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    offset_x: i32,
    offset_y: i32,
) {
    let src_arr = src.as_array();
    let mut dst_arr = dst.as_array_mut();

    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);
    let (src_h, src_w, _) = (src_arr.shape()[0], src_arr.shape()[1], src_arr.shape()[2]);

    let start_x = offset_x.max(0) as usize;
    let start_y = offset_y.max(0) as usize;
    let end_x = ((offset_x + src_w as i32) as usize).min(dst_w);
    let end_y = ((offset_y + src_h as i32) as usize).min(dst_h);

    let src_start_x = (-offset_x).max(0) as usize;
    let src_start_y = (-offset_y).max(0) as usize;

    for dy in start_y..end_y {
        let sy = src_start_y + (dy - start_y);
        if sy >= src_h { break; }

        for dx in start_x..end_x {
            let sx = src_start_x + (dx - start_x);
            if sx >= src_w { break; }

            let src_a = src_arr[[sy, sx, 3]] as u32;
            if src_a == 0 { continue; }

            if src_a == 255 {
                dst_arr[[dy, dx, 0]] = src_arr[[sy, sx, 0]];
                dst_arr[[dy, dx, 1]] = src_arr[[sy, sx, 1]];
                dst_arr[[dy, dx, 2]] = src_arr[[sy, sx, 2]];
                dst_arr[[dy, dx, 3]] = 255;
            } else {
                let dst_a = dst_arr[[dy, dx, 3]] as u32;
                let inv_src_a = 255 - src_a;
                let out_a = src_a + (dst_a * inv_src_a / 255);

                if out_a == 0 {
                    dst_arr[[dy, dx, 0]] = 0;
                    dst_arr[[dy, dx, 1]] = 0;
                    dst_arr[[dy, dx, 2]] = 0;
                    dst_arr[[dy, dx, 3]] = 0;
                } else {
                    for c in 0..3 {
                        let src_c = src_arr[[sy, sx, c]] as u32;
                        let dst_c = dst_arr[[dy, dx, c]] as u32;
                        let out_c = (src_c * src_a + dst_c * dst_a * inv_src_a / 255) / out_a;
                        dst_arr[[dy, dx, c]] = out_c.min(255) as u8;
                    }
                    dst_arr[[dy, dx, 3]] = out_a.min(255) as u8;
                }
            }
        }
    }
}

/// Fast NxN downscale with premultiplied alpha blending
#[inline]
fn downscale_nxn(
    src_arr: &ndarray::ArrayView3<u8>,
    dst: &mut Array3<u8>,
    new_width: usize,
    new_height: usize,
    scale: usize,
) {
    let pixels_per_block = (scale * scale) as u32;

    for dy in 0..new_height {
        let sy_base = dy * scale;
        for dx in 0..new_width {
            let sx_base = dx * scale;

            let mut sum_r = 0u64;
            let mut sum_g = 0u64;
            let mut sum_b = 0u64;
            let mut sum_a = 0u64;

            for oy in 0..scale {
                let sy = sy_base + oy;
                for ox in 0..scale {
                    let sx = sx_base + ox;
                    let a = src_arr[[sy, sx, 3]] as u64;
                    sum_r += src_arr[[sy, sx, 0]] as u64 * a;
                    sum_g += src_arr[[sy, sx, 1]] as u64 * a;
                    sum_b += src_arr[[sy, sx, 2]] as u64 * a;
                    sum_a += a;
                }
            }

            if sum_a > 0 {
                dst[[dy, dx, 0]] = (sum_r / sum_a).min(255) as u8;
                dst[[dy, dx, 1]] = (sum_g / sum_a).min(255) as u8;
                dst[[dy, dx, 2]] = (sum_b / sum_a).min(255) as u8;
                dst[[dy, dx, 3]] = (sum_a / pixels_per_block as u64) as u8;
            }
        }
    }
}

/// Resize RGBA image using box filter (area averaging for downscale)
#[pyfunction]
pub fn resize_rgba<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    new_width: usize,
    new_height: usize,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let src_arr = src.as_array();
    let (src_h, src_w, _) = (src_arr.shape()[0], src_arr.shape()[1], src_arr.shape()[2]);

    if new_width == 0 || new_height == 0 {
        return Array3::<u8>::zeros((new_height, new_width, 4)).into_pyarray(py);
    }

    let mut dst = Array3::<u8>::zeros((new_height, new_width, 4));

    // Check for exact integer downscale (2x, 3x, 4x, etc.)
    if src_w % new_width == 0 && src_h % new_height == 0 {
        let scale_x = src_w / new_width;
        let scale_y = src_h / new_height;

        if scale_x == scale_y && scale_x >= 2 && scale_x <= 8 {
            downscale_nxn(&src_arr, &mut dst, new_width, new_height, scale_x);
            return dst.into_pyarray(py);
        }
    }

    // General area averaging for non-integer scales
    let scale_x = src_w as f64 / new_width as f64;
    let scale_y = src_h as f64 / new_height as f64;

    for dy in 0..new_height {
        let src_y_start = (dy as f64 * scale_y).floor() as usize;
        let src_y_end = (((dy + 1) as f64 * scale_y).ceil() as usize).min(src_h);

        for dx in 0..new_width {
            let src_x_start = (dx as f64 * scale_x).floor() as usize;
            let src_x_end = (((dx + 1) as f64 * scale_x).ceil() as usize).min(src_w);

            let mut sum_r = 0u64;
            let mut sum_g = 0u64;
            let mut sum_b = 0u64;
            let mut sum_a = 0u64;

            for sy in src_y_start..src_y_end {
                for sx in src_x_start..src_x_end {
                    let a = src_arr[[sy, sx, 3]] as u64;
                    sum_r += src_arr[[sy, sx, 0]] as u64 * a;
                    sum_g += src_arr[[sy, sx, 1]] as u64 * a;
                    sum_b += src_arr[[sy, sx, 2]] as u64 * a;
                    sum_a += a;
                }
            }

            let count = ((src_y_end - src_y_start) * (src_x_end - src_x_start)) as u64;
            if sum_a > 0 {
                dst[[dy, dx, 0]] = ((sum_r / sum_a).min(255)) as u8;
                dst[[dy, dx, 1]] = ((sum_g / sum_a).min(255)) as u8;
                dst[[dy, dx, 2]] = ((sum_b / sum_a).min(255)) as u8;
            }
            dst[[dy, dx, 3]] = (sum_a / count.max(1)) as u8;
        }
    }

    dst.into_pyarray(py)
}

/// Convert sRGB to linearRGB color space (for filter operations)
#[pyfunction]
pub fn srgb_to_linear<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let mut lut = [0u8; 256];
    for i in 0..256 {
        let c = i as f32 / 255.0;
        let linear = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
        lut[i] = (linear * 255.0).round() as u8;
    }

    for y in 0..h {
        for x in 0..w {
            dst[[y, x, 0]] = lut[arr[[y, x, 0]] as usize];
            dst[[y, x, 1]] = lut[arr[[y, x, 1]] as usize];
            dst[[y, x, 2]] = lut[arr[[y, x, 2]] as usize];
            dst[[y, x, 3]] = arr[[y, x, 3]];
        }
    }

    dst.into_pyarray(py)
}

/// Convert linearRGB to sRGB color space (after filter operations)
#[pyfunction]
pub fn linear_to_srgb<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let mut lut = [0u8; 256];
    for i in 0..256 {
        let c = i as f32 / 255.0;
        let srgb = if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        lut[i] = (srgb * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    for y in 0..h {
        for x in 0..w {
            dst[[y, x, 0]] = lut[arr[[y, x, 0]] as usize];
            dst[[y, x, 1]] = lut[arr[[y, x, 1]] as usize];
            dst[[y, x, 2]] = lut[arr[[y, x, 2]] as usize];
            dst[[y, x, 3]] = arr[[y, x, 3]];
        }
    }

    dst.into_pyarray(py)
}

/// Apply a grayscale clip mask to an RGBA image's alpha channel (in-place)
/// This replaces: temp_image.putalpha(ImageChops.multiply(temp_image.split()[3], mask))
#[pyfunction]
pub fn apply_clip_mask<'py>(
    _py: Python<'py>,
    mut img: numpy::PyReadwriteArray3<'py, u8>,
    mask: numpy::PyReadonlyArray2<'py, u8>,
) {
    let mask_arr = mask.as_array();
    let mut img_arr = img.as_array_mut();

    let (img_h, img_w, _) = (img_arr.shape()[0], img_arr.shape()[1], img_arr.shape()[2]);
    let (mask_h, mask_w) = (mask_arr.shape()[0], mask_arr.shape()[1]);

    // Dimensions must match
    if img_h != mask_h || img_w != mask_w {
        return;
    }

    for y in 0..img_h {
        for x in 0..img_w {
            let img_alpha = img_arr[[y, x, 3]] as u32;
            let mask_val = mask_arr[[y, x]] as u32;
            // Multiply alpha by mask value (both 0-255, result 0-255)
            img_arr[[y, x, 3]] = ((img_alpha * mask_val) / 255) as u8;
        }
    }
}

/// Apply a grayscale mask and alpha composite onto destination in one pass
/// Combines apply_clip_mask + alpha_composite_inplace for clip path rendering
/// Optional bounds (min_x, min_y, max_x, max_y) to limit processing region
#[pyfunction]
#[pyo3(signature = (dst, src, mask, offset_x, offset_y, bounds=None))]
pub fn apply_mask_and_composite<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    mask: numpy::PyReadonlyArray2<'py, u8>,
    offset_x: i32,
    offset_y: i32,
    bounds: Option<(i32, i32, i32, i32)>,
) {
    let src_arr = src.as_array();
    let mask_arr = mask.as_array();
    let mut dst_arr = dst.as_array_mut();

    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);
    let (src_h, src_w, _) = (src_arr.shape()[0], src_arr.shape()[1], src_arr.shape()[2]);
    let (mask_h, mask_w) = (mask_arr.shape()[0], mask_arr.shape()[1]);

    // Dimensions must match between src and mask
    if src_h != mask_h || src_w != mask_w {
        return;
    }

    // Calculate processing bounds
    let (mut start_x, mut start_y, mut end_x, mut end_y) = if let Some((bx1, by1, bx2, by2)) = bounds {
        // Use provided bounds, clamped to valid range
        (
            (bx1.max(0) as usize).max(offset_x.max(0) as usize),
            (by1.max(0) as usize).max(offset_y.max(0) as usize),
            ((bx2 as usize).min(dst_w)).min((offset_x + src_w as i32) as usize),
            ((by2 as usize).min(dst_h)).min((offset_y + src_h as i32) as usize),
        )
    } else {
        (
            offset_x.max(0) as usize,
            offset_y.max(0) as usize,
            ((offset_x + src_w as i32) as usize).min(dst_w),
            ((offset_y + src_h as i32) as usize).min(dst_h),
        )
    };

    // Ensure valid range
    if start_x >= end_x || start_y >= end_y {
        return;
    }

    let src_start_x = (-offset_x).max(0) as usize;
    let src_start_y = (-offset_y).max(0) as usize;

    for dy in start_y..end_y {
        let sy = src_start_y + (dy - start_y);
        if sy >= src_h { break; }

        for dx in start_x..end_x {
            let sx = src_start_x + (dx - start_x);
            if sx >= src_w { break; }

            // Apply mask to source alpha
            let src_a_orig = src_arr[[sy, sx, 3]] as u32;
            let mask_val = mask_arr[[sy, sx]] as u32;
            let src_a = (src_a_orig * mask_val / 255) as u32;

            if src_a == 0 { continue; }

            if src_a == 255 {
                dst_arr[[dy, dx, 0]] = src_arr[[sy, sx, 0]];
                dst_arr[[dy, dx, 1]] = src_arr[[sy, sx, 1]];
                dst_arr[[dy, dx, 2]] = src_arr[[sy, sx, 2]];
                dst_arr[[dy, dx, 3]] = 255;
            } else {
                let dst_a = dst_arr[[dy, dx, 3]] as u32;
                let inv_src_a = 255 - src_a;
                let out_a = src_a + (dst_a * inv_src_a / 255);

                if out_a == 0 {
                    dst_arr[[dy, dx, 0]] = 0;
                    dst_arr[[dy, dx, 1]] = 0;
                    dst_arr[[dy, dx, 2]] = 0;
                    dst_arr[[dy, dx, 3]] = 0;
                } else {
                    for c in 0..3 {
                        let src_c = src_arr[[sy, sx, c]] as u32;
                        let dst_c = dst_arr[[dy, dx, c]] as u32;
                        let out_c = (src_c * src_a + dst_c * dst_a * inv_src_a / 255) / out_a;
                        dst_arr[[dy, dx, c]] = out_c.min(255) as u8;
                    }
                    dst_arr[[dy, dx, 3]] = out_a.min(255) as u8;
                }
            }
        }
    }
}

/// Register image module functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(alpha_composite_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(resize_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(srgb_to_linear, m)?)?;
    m.add_function(wrap_pyfunction!(linear_to_srgb, m)?)?;
    m.add_function(wrap_pyfunction!(apply_clip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(apply_mask_and_composite, m)?)?;
    Ok(())
}
