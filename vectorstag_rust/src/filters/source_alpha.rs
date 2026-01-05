//! Source alpha extraction utility

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn get_source_alpha_impl_f32(src: &ndarray::ArrayView3<f32>) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a = src[[y, x, 3]];
            dst[[y, x, 0]] = 0.0;
            dst[[y, x, 1]] = 0.0;
            dst[[y, x, 2]] = 0.0;
            dst[[y, x, 3]] = a;
        }
    }
    dst
}

pub fn get_source_alpha_impl(src: &ndarray::ArrayView3<u8>) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a = src[[y, x, 3]];
            dst[[y, x, 0]] = 0;
            dst[[y, x, 1]] = 0;
            dst[[y, x, 2]] = 0;
            dst[[y, x, 3]] = a;
        }
    }
    dst
}

/// Get SourceAlpha from input image
#[pyfunction]
pub fn get_source_alpha<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    get_source_alpha_impl(&arr).into_pyarray(py)
}
