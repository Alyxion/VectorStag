//! feGaussianBlur - optimized Gaussian blur

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;
use super::blur_utils::box_blur_integral;

pub fn fe_gaussian_blur_impl_f32(src: &ndarray::ArrayView3<f32>, std_dev_x: f32, std_dev_y: f32) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    if std_dev_x < 0.5 && std_dev_y < 0.5 {
        return src.to_owned();
    }

    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    // Box blur approximation of Gaussian: box_width = sqrt(12σ²/n + 1) for n passes
    // For 3 passes: width ≈ sqrt(4σ² + 1), radius = (width - 1) / 2
    let box_radius_x = (((12.0 * std_dev_x * std_dev_x / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize;
    let box_radius_y = (((12.0 * std_dev_y * std_dev_y / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize;

    let total_pixels = h * w * 4;

    let mut buf_a = vec![0.0f32; total_pixels];
    let mut buf_b = vec![0.0f32; total_pixels];

    // Copy src to buffer
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            buf_a[idx] = src[[y, x, 0]];
            buf_a[idx + 1] = src[[y, x, 1]];
            buf_a[idx + 2] = src[[y, x, 2]];
            buf_a[idx + 3] = src[[y, x, 3]];
        }
    }

    let mut current = &mut buf_a;
    let mut next = &mut buf_b;

    if box_radius_x > 0 || box_radius_y > 0 {
        for _ in 0..3 {
            box_blur_integral(current, next, w, h, box_radius_x, box_radius_y);
            std::mem::swap(&mut current, &mut next);
        }
    }

    let mut dst = Array3::<f32>::zeros((h, w, 4));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            dst[[y, x, 0]] = current[idx].clamp(0.0, 1.0);
            dst[[y, x, 1]] = current[idx + 1].clamp(0.0, 1.0);
            dst[[y, x, 2]] = current[idx + 2].clamp(0.0, 1.0);
            dst[[y, x, 3]] = current[idx + 3].clamp(0.0, 1.0);
        }
    }
    dst
}

pub fn fe_gaussian_blur_impl(src: &ndarray::ArrayView3<u8>, std_dev_x: f32, std_dev_y: f32) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    if std_dev_x < 0.5 && std_dev_y < 0.5 {
        let mut dst = Array3::<u8>::zeros((h, w, 4));
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = src[[y, x, c]]; }
            }
        }
        return dst;
    }

    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    // Box blur approximation of Gaussian: box_width = sqrt(12σ²/n + 1) for n passes
    // For 3 passes: width ≈ sqrt(4σ² + 1), radius = (width - 1) / 2
    let box_radius_x = (((12.0 * std_dev_x * std_dev_x / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize;
    let box_radius_y = (((12.0 * std_dev_y * std_dev_y / 3.0 + 1.0).sqrt() - 1.0) / 2.0 + 0.5).floor() as usize;

    let total_pixels = h * w * 4;

    let mut buf_a = vec![0.0f32; total_pixels];
    let mut buf_b = vec![0.0f32; total_pixels];

    // Copy src to buffer (assuming already premultiplied)
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            buf_a[idx] = src[[y, x, 0]] as f32;
            buf_a[idx + 1] = src[[y, x, 1]] as f32;
            buf_a[idx + 2] = src[[y, x, 2]] as f32;
            buf_a[idx + 3] = src[[y, x, 3]] as f32;
        }
    }

    let mut current = &mut buf_a;
    let mut next = &mut buf_b;

    if box_radius_x > 0 || box_radius_y > 0 {
        for _ in 0..3 {
            box_blur_integral(current, next, w, h, box_radius_x, box_radius_y);
            std::mem::swap(&mut current, &mut next);
        }
    }

    let mut dst = Array3::<u8>::zeros((h, w, 4));
    for i in 0..total_pixels {
        // Flat copy back to Array3 structure
        // We can optimize this but let's stick to the loop for now or reuse index
    }
    
    // Using explicit loops to map back to Array3
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            dst[[y, x, 0]] = current[idx].clamp(0.0, 255.0) as u8;
            dst[[y, x, 1]] = current[idx + 1].clamp(0.0, 255.0) as u8;
            dst[[y, x, 2]] = current[idx + 2].clamp(0.0, 255.0) as u8;
            dst[[y, x, 3]] = current[idx + 3].clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

/// feGaussianBlur - optimized Gaussian blur
#[pyfunction]
pub fn fe_gaussian_blur<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    std_dev_x: f32,
    std_dev_y: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_gaussian_blur_impl(&arr, std_dev_x, std_dev_y).into_pyarray(py)
}

