//! feTile - tile input image to fill region

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_tile_impl_f32(src: &ndarray::ArrayView3<f32>, out_width: usize, out_height: usize) -> Array3<f32> {
    let (src_h, src_w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((out_height, out_width, 4));

    if src_h == 0 || src_w == 0 {
        return dst;
    }

    for y in 0..out_height {
        let src_y = y % src_h;
        for x in 0..out_width {
            let src_x = x % src_w;
            for c in 0..4 {
                dst[[y, x, c]] = src[[src_y, src_x, c]];
            }
        }
    }
    dst
}

pub fn fe_tile_impl(src: &ndarray::ArrayView3<u8>, out_width: usize, out_height: usize) -> Array3<u8> {
    let (src_h, src_w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((out_height, out_width, 4));

    if src_h == 0 || src_w == 0 {
        return dst;
    }

    for y in 0..out_height {
        let src_y = y % src_h;
        for x in 0..out_width {
            let src_x = x % src_w;
            for c in 0..4 {
                dst[[y, x, c]] = src[[src_y, src_x, c]];
            }
        }
    }
    dst
}

/// feTile - tile input image to fill region
#[pyfunction]
pub fn fe_tile<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    out_width: usize,
    out_height: usize,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let src_arr = src.as_array();
    fe_tile_impl(&src_arr, out_width, out_height).into_pyarray(py)
}

