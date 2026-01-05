//! feFlood - fill entire region with solid color

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_flood_impl_f32(
    width: usize,
    height: usize,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) -> Array3<f32> {
    let mut dst = Array3::<f32>::zeros((height, width, 4));
    for y in 0..height {
        for x in 0..width {
            dst[[y, x, 0]] = r;
            dst[[y, x, 1]] = g;
            dst[[y, x, 2]] = b;
            dst[[y, x, 3]] = a;
        }
    }
    dst
}

pub fn fe_flood_impl(width: usize, height: usize, r: u8, g: u8, b: u8, a: u8) -> Array3<u8> {
    let mut pixels = Array3::<u8>::zeros((height, width, 4));
    for y in 0..height {
        for x in 0..width {
            pixels[[y, x, 0]] = r;
            pixels[[y, x, 1]] = g;
            pixels[[y, x, 2]] = b;
            pixels[[y, x, 3]] = a;
        }
    }
    pixels
}

/// feFlood - fill entire region with solid color
#[pyfunction]
pub fn fe_flood<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    r: u8, g: u8, b: u8, a: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    fe_flood_impl(width, height, r, g, b, a).into_pyarray(py)
}

#[pyfunction]
pub fn fe_flood_f32<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    r: f32, g: f32, b: f32, a: f32,
) -> Bound<'py, numpy::PyArray3<f32>> {
    fe_flood_impl_f32(width, height, r, g, b, a).into_pyarray(py)
}
