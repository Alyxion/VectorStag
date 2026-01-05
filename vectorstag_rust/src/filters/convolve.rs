//! feConvolveMatrix - apply convolution kernel

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_convolve_matrix_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    order_x: usize,
    order_y: usize,
    kernel: &[f32],
    divisor: f32,
    bias: f32,
    target_x: usize,
    target_y: usize,
    edge_mode: u8,
    preserve_alpha: bool,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    let div = if divisor.abs() < 1e-10 { 1.0 } else { divisor };
    let scaled_kernel: Vec<f32> = kernel.iter().map(|k| k / div).collect();
    
    let h_i = h as i32;
    let w_i = w as i32;
    let target_y_i = target_y as i32;
    let target_x_i = target_x as i32;

    let channels = if preserve_alpha { 3 } else { 4 };

    if scaled_kernel.is_empty() || order_x == 0 || order_y == 0 {
        return src.to_owned();
    }

    if edge_mode == 0 {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    let sy = (y_i + ky as i32 - target_y_i).clamp(0, h_i - 1) as usize;
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }
                        let sx = (x_i + kx as i32 - target_x_i).clamp(0, w_i - 1) as usize;
                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += src[[sy, sx, c]] * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    let add_bias = if c == 3 { 0.0 } else { bias };
                    dst[[y, x, c]] = (sum[c] + add_bias).clamp(0.0, 1.0);
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = src[[y, x, 3]];
                }
            }
        }
    } else {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }

                        let sy = y_i + ky as i32 - target_y_i;
                        let sx = x_i + kx as i32 - target_x_i;

                        let (sy, sx) = match edge_mode {
                            1 => (sy.rem_euclid(h_i), sx.rem_euclid(w_i)),
                            _ => {
                                if sy < 0 || sy >= h_i || sx < 0 || sx >= w_i { continue; }
                                (sy, sx)
                            }
                        };

                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += src[[sy as usize, sx as usize, c]] * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    let add_bias = if c == 3 { 0.0 } else { bias };
                    dst[[y, x, c]] = (sum[c] + add_bias).clamp(0.0, 1.0);
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = src[[y, x, 3]];
                }
            }
        }
    }
    dst
}

pub fn fe_convolve_matrix_impl(
    src: &ndarray::ArrayView3<u8>,
    order_x: usize,
    order_y: usize,
    kernel: &[f32],
    divisor: f32,
    bias: f32,
    target_x: usize,
    target_y: usize,
    edge_mode: u8,
    preserve_alpha: bool,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let div = if divisor.abs() < 1e-10 { 1.0 } else { divisor };
    let scaled_kernel: Vec<f32> = kernel.iter().map(|k| k / div).collect();
    let bias_255 = bias * 255.0;

    let h_i = h as i32;
    let w_i = w as i32;
    let target_y_i = target_y as i32;
    let target_x_i = target_x as i32;

    let channels = if preserve_alpha { 3 } else { 4 };

    if scaled_kernel.is_empty() || order_x == 0 || order_y == 0 {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = src[[y, x, c]]; }
            }
        }
        return dst;
    }

    if edge_mode == 0 {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    let sy = (y_i + ky as i32 - target_y_i).clamp(0, h_i - 1) as usize;
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }
                        let sx = (x_i + kx as i32 - target_x_i).clamp(0, w_i - 1) as usize;
                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += src[[sy, sx, c]] as f32 * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    dst[[y, x, c]] = (sum[c] + bias_255).clamp(0.0, 255.0) as u8;
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = src[[y, x, 3]];
                }
            }
        }
    } else {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }

                        let sy = y_i + ky as i32 - target_y_i;
                        let sx = x_i + kx as i32 - target_x_i;

                        let (sy, sx) = match edge_mode {
                            1 => (sy.rem_euclid(h_i), sx.rem_euclid(w_i)),
                            _ => {
                                if sy < 0 || sy >= h_i || sx < 0 || sx >= w_i { continue; }
                                (sy, sx)
                            }
                        };

                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += src[[sy as usize, sx as usize, c]] as f32 * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    dst[[y, x, c]] = (sum[c] + bias_255).clamp(0.0, 255.0) as u8;
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = src[[y, x, 3]];
                }
            }
        }
    }
    dst
}

/// feConvolveMatrix - apply convolution kernel
#[pyfunction]
pub fn fe_convolve_matrix<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    order_x: usize,
    order_y: usize,
    kernel: Vec<f32>,
    divisor: f32,
    bias: f32,
    target_x: usize,
    target_y: usize,
    edge_mode: u8,
    preserve_alpha: bool,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_convolve_matrix_impl(&arr, order_x, order_y, &kernel, divisor, bias, target_x, target_y, edge_mode, preserve_alpha).into_pyarray(py)
}

pub fn generate_gradients(seed: i32) -> [[f64; 2]; 256] {
    let mut gradients = [[0.0f64; 2]; 256];
    let mut rng = seed as u32;

    for i in 0..256 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let angle = (rng as f64 / u32::MAX as f64) * std::f64::consts::PI * 2.0;
        gradients[i] = [angle.cos(), angle.sin()];
    }

    gradients
}

pub fn perlin_noise(x: f64, y: f64, channel: usize, gradients: &[[f64; 2]; 256]) -> f64 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);

    let hash = |x: i32, y: i32, c: usize| -> usize {
        ((x.wrapping_mul(1619) ^ y.wrapping_mul(31337) ^ (c as i32 * 6971)) & 0xFF) as usize
    };

    let g00 = &gradients[hash(x0, y0, channel)];
    let g10 = &gradients[hash(x1, y0, channel)];
    let g01 = &gradients[hash(x0, y1, channel)];
    let g11 = &gradients[hash(x1, y1, channel)];

    let n00 = g00[0] * fx + g00[1] * fy;
    let n10 = g10[0] * (fx - 1.0) + g10[1] * fy;
    let n01 = g01[0] * fx + g01[1] * (fy - 1.0);
    let n11 = g11[0] * (fx - 1.0) + g11[1] * (fy - 1.0);

    let nx0 = n00 + u * (n10 - n00);
    let nx1 = n01 + u * (n11 - n01);

    nx0 + v * (nx1 - nx0)
}

