//! feMerge - merge multiple layers

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_merge_impl_f32(layers: &[ndarray::ArrayView3<f32>]) -> Array3<f32> {
    if layers.is_empty() {
        return Array3::<f32>::zeros((1, 1, 4));
    }

    let first = &layers[0];
    let (h, w, _) = (first.shape()[0], first.shape()[1], first.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                dst[[y, x, c]] = first[[y, x, c]];
            }
        }
    }

    for layer in layers.iter().skip(1) {
        let src = layer;
        for y in 0..h {
            for x in 0..w {
                let src_a = src[[y, x, 3]];
                let inv_sa = 1.0 - src_a;

                for c in 0..4 {
                    let dst_c = dst[[y, x, c]];
                    let src_c = src[[y, x, c]];
                    let out = src_c + dst_c * inv_sa;
                    dst[[y, x, c]] = out.clamp(0.0, 1.0);
                }
            }
        }
    }
    dst
}

pub fn fe_merge_impl(layers: &[ndarray::ArrayView3<u8>]) -> Array3<u8> {
    if layers.is_empty() {
        return Array3::<u8>::zeros((1, 1, 4));
    }

    let first = &layers[0];
    let (h, w, _) = (first.shape()[0], first.shape()[1], first.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                dst[[y, x, c]] = first[[y, x, c]];
            }
        }
    }

    for layer in layers.iter().skip(1) {
        let src = layer;
        for y in 0..h {
            for x in 0..w {
                let src_a = src[[y, x, 3]] as f32 / 255.0;
                let inv_sa = 1.0 - src_a;

                for c in 0..4 {
                    let dst_c = dst[[y, x, c]] as f32 / 255.0;
                    let src_c = src[[y, x, c]] as f32 / 255.0;
                    let out = src_c + dst_c * inv_sa;
                    dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    dst
}

/// feMerge - merge multiple layers
#[pyfunction]
pub fn fe_merge<'py>(
    py: Python<'py>,
    layers: Vec<numpy::PyReadonlyArray3<'py, u8>>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let layer_views: Vec<_> = layers.iter().map(|l| l.as_array()).collect();
    fe_merge_impl(&layer_views).into_pyarray(py)
}
