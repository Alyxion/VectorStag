//! feComposite - Porter-Duff compositing

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_composite_impl_f32(arr1: &ndarray::ArrayView3<f32>, arr2: &ndarray::ArrayView3<f32>, operator: u8, k1: f32, k2: f32, k3: f32, k4: f32) -> Array3<f32> {
    let (h, w, _) = (arr1.shape()[0], arr1.shape()[1], arr1.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a1 = arr1[[y, x, 3]];
            let a2 = arr2[[y, x, 3]];

            let (fa, fb) = match operator {
                0 => (1.0, 1.0 - a1),  // over
                1 => (a2, 0.0),        // in
                2 => (1.0 - a2, 0.0),  // out
                3 => (a2, 1.0 - a1),   // atop
                4 => (1.0 - a2, 1.0 - a1), // xor
                5 => (0.0, 0.0),       // arithmetic
                _ => (1.0, 1.0 - a1),
            };

            for c in 0..4 {
                let c1 = arr1[[y, x, c]];
                let c2 = arr2[[y, x, c]];

                let out = if operator == 5 {
                    k1 * c1 * c2 + k2 * c1 + k3 * c2 + k4
                } else {
                    c1 * fa + c2 * fb
                };

                dst[[y, x, c]] = out.clamp(0.0, 1.0);
            }
        }
    }
    dst
}

pub fn fe_composite_impl(arr1: &ndarray::ArrayView3<u8>, arr2: &ndarray::ArrayView3<u8>, operator: u8, k1: f32, k2: f32, k3: f32, k4: f32) -> Array3<u8> {
    let (h, w, _) = (arr1.shape()[0], arr1.shape()[1], arr1.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a1 = arr1[[y, x, 3]] as f32 / 255.0;
            let a2 = arr2[[y, x, 3]] as f32 / 255.0;

            let (fa, fb) = match operator {
                0 => (1.0, 1.0 - a1),  // over
                1 => (a2, 0.0),        // in
                2 => (1.0 - a2, 0.0),  // out
                3 => (a2, 1.0 - a1),   // atop
                4 => (1.0 - a2, 1.0 - a1), // xor
                5 => (0.0, 0.0),       // arithmetic
                _ => (1.0, 1.0 - a1),
            };

            for c in 0..4 {
                let c1 = arr1[[y, x, c]] as f32 / 255.0;
                let c2 = arr2[[y, x, c]] as f32 / 255.0;

                let out = if operator == 5 {
                    k1 * c1 * c2 + k2 * c1 + k3 * c2 + k4
                } else {
                    c1 * fa + c2 * fb
                };

                dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// feComposite - Porter-Duff compositing
#[pyfunction]
pub fn fe_composite<'py>(
    py: Python<'py>,
    in1: numpy::PyReadonlyArray3<'py, u8>,
    in2: numpy::PyReadonlyArray3<'py, u8>,
    operator: u8,
    k1: f32, k2: f32, k3: f32, k4: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr1 = in1.as_array();
    let arr2 = in2.as_array();
    fe_composite_impl(&arr1, &arr2, operator, k1, k2, k3, k4).into_pyarray(py)
}
