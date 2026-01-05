//! feTurbulence - generate Perlin noise

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;
use super::convolve::{generate_gradients, perlin_noise};

pub fn fe_turbulence_impl_f32(
    width: usize,
    height: usize,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: usize,
    seed: i32,
    noise_type: u8,
    _stitch_tiles: bool,
) -> Array3<f32> {
    let mut pixels = Array3::<f32>::zeros((height, width, 4));
    let gradients = generate_gradients(seed);

    for y in 0..height {
        for x in 0..width {
            for c in 0..4 {
                let mut noise = 0.0f64;
                let mut amplitude = 1.0f64;
                let mut freq_x = base_freq_x;
                let mut freq_y = base_freq_y;

                for _ in 0..num_octaves {
                    let nx = x as f64 * freq_x;
                    let ny = y as f64 * freq_y;
                    let n = perlin_noise(nx, ny, c, &gradients);

                    if noise_type == 0 {
                        noise += n.abs() * amplitude;
                    } else {
                        noise += n * amplitude;
                    }

                    amplitude *= 0.5;
                    freq_x *= 2.0;
                    freq_y *= 2.0;
                }

                let val = if noise_type == 0 {
                    noise
                } else {
                    (noise + 1.0) * 0.5
                };

                pixels[[y, x, c]] = val.clamp(0.0, 1.0) as f32;
            }
        }
    }
    pixels
}

pub fn fe_turbulence_impl(
    width: usize,
    height: usize,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: usize,
    seed: i32,
    noise_type: u8,
    _stitch_tiles: bool,
) -> Array3<u8> {
    let mut pixels = Array3::<u8>::zeros((height, width, 4));
    let gradients = generate_gradients(seed);

    for y in 0..height {
        for x in 0..width {
            for c in 0..4 {
                let mut noise = 0.0f64;
                let mut amplitude = 1.0f64;
                let mut freq_x = base_freq_x;
                let mut freq_y = base_freq_y;

                for _ in 0..num_octaves {
                    let nx = x as f64 * freq_x;
                    let ny = y as f64 * freq_y;
                    let n = perlin_noise(nx, ny, c, &gradients);

                    if noise_type == 0 {
                        noise += n.abs() * amplitude;
                    } else {
                        noise += n * amplitude;
                    }

                    amplitude *= 0.5;
                    freq_x *= 2.0;
                    freq_y *= 2.0;
                }

                let val = if noise_type == 0 {
                    noise
                } else {
                    (noise + 1.0) * 0.5
                };

                pixels[[y, x, c]] = (val * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    pixels
}

/// feTurbulence - generate Perlin noise
#[pyfunction]
pub fn fe_turbulence<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: usize,
    seed: i32,
    noise_type: u8,
    stitch_tiles: bool,
) -> Bound<'py, numpy::PyArray3<u8>> {
    fe_turbulence_impl(width, height, base_freq_x, base_freq_y, num_octaves, seed, noise_type, stitch_tiles).into_pyarray(py)
}

