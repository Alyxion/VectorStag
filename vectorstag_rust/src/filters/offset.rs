//! feOffset - offset image by dx, dy

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_offset_impl_f32(src: &ndarray::ArrayView3<f32>, dx: i32, dy: i32) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        let src_y = y as i32 - dy;
        if src_y < 0 || src_y >= h as i32 {
            continue;
        }
        for x in 0..w {
            let src_x = x as i32 - dx;
            if src_x < 0 || src_x >= w as i32 {
                continue;
            }
            for c in 0..4 {
                dst[[y, x, c]] = src[[src_y as usize, src_x as usize, c]];
            }
        }
    }
    dst
}

pub fn fe_offset_impl(src: &ndarray::ArrayView3<u8>, dx: i32, dy: i32) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        let src_y = y as i32 - dy;
        if src_y < 0 || src_y >= h as i32 {
            continue;
        }
        for x in 0..w {
            let src_x = x as i32 - dx;
            if src_x < 0 || src_x >= w as i32 {
                continue;
            }
            for c in 0..4 {
                dst[[y, x, c]] = src[[src_y as usize, src_x as usize, c]];
            }
        }
    }
    dst
}

/// feOffset - offset image by dx, dy
#[pyfunction]
pub fn fe_offset<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    dx: i32,
    dy: i32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let src_arr = src.as_array();
    fe_offset_impl(&src_arr, dx, dy).into_pyarray(py)
}
