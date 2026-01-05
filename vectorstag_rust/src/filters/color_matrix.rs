//! feColorMatrix - apply color transformation matrix

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_color_matrix_impl_f32(src: &ndarray::ArrayView3<f32>, matrix_type: u8, values: &[f32]) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    let mut m = [[0.0f32; 5]; 4];

    let set_identity = |mat: &mut [[f32; 5]; 4]| {
        mat[0] = [1.0, 0.0, 0.0, 0.0, 0.0];
        mat[1] = [0.0, 1.0, 0.0, 0.0, 0.0];
        mat[2] = [0.0, 0.0, 1.0, 0.0, 0.0];
        mat[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
    };

    match matrix_type {
        0 => {
            if values.len() != 20 {
                set_identity(&mut m);
            } else {
                for i in 0..4 {
                    for j in 0..5 {
                        m[i][j] = values[i * 5 + j];
                    }
                }
            }
        }
        1 => {
            let s = if values.len() == 1 { values[0] } else { 1.0 };
            m[0] = [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[1] = [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[2] = [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        2 => {
            let angle_deg = if values.len() == 1 { values[0] } else { 0.0 };
            let angle = angle_deg.to_radians();
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            m[0] = [0.213 + cos_a * 0.787 - sin_a * 0.213, 0.715 - cos_a * 0.715 - sin_a * 0.715, 0.072 - cos_a * 0.072 + sin_a * 0.928, 0.0, 0.0];
            m[1] = [0.213 - cos_a * 0.213 + sin_a * 0.143, 0.715 + cos_a * 0.285 + sin_a * 0.140, 0.072 - cos_a * 0.072 - sin_a * 0.283, 0.0, 0.0];
            m[2] = [0.213 - cos_a * 0.213 - sin_a * 0.787, 0.715 - cos_a * 0.715 + sin_a * 0.715, 0.072 + cos_a * 0.928 + sin_a * 0.072, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        3 => {
            m[0] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[1] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[2] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[3] = [0.2126, 0.7152, 0.0722, 0.0, 0.0];
        }
        _ => {
            set_identity(&mut m);
        }
    }

    for y in 0..h {
        for x in 0..w {
            let r = src[[y, x, 0]];
            let g = src[[y, x, 1]];
            let b = src[[y, x, 2]];
            let a = src[[y, x, 3]];

            let out_r = (m[0][0] * r + m[0][1] * g + m[0][2] * b + m[0][3] * a + m[0][4]).clamp(0.0, 1.0);
            let out_g = (m[1][0] * r + m[1][1] * g + m[1][2] * b + m[1][3] * a + m[1][4]).clamp(0.0, 1.0);
            let out_b = (m[2][0] * r + m[2][1] * g + m[2][2] * b + m[2][3] * a + m[2][4]).clamp(0.0, 1.0);
            let out_a = (m[3][0] * r + m[3][1] * g + m[3][2] * b + m[3][3] * a + m[3][4]).clamp(0.0, 1.0);

            dst[[y, x, 0]] = out_r;
            dst[[y, x, 1]] = out_g;
            dst[[y, x, 2]] = out_b;
            dst[[y, x, 3]] = out_a;
        }
    }
    dst
}

pub fn fe_color_matrix_impl(src: &ndarray::ArrayView3<u8>, matrix_type: u8, values: &[f32]) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let mut m = [[0.0f32; 5]; 4];

    let set_identity = |mat: &mut [[f32; 5]; 4]| {
        mat[0] = [1.0, 0.0, 0.0, 0.0, 0.0];
        mat[1] = [0.0, 1.0, 0.0, 0.0, 0.0];
        mat[2] = [0.0, 0.0, 1.0, 0.0, 0.0];
        mat[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
    };

    match matrix_type {
        0 => {
            if values.len() != 20 {
                set_identity(&mut m);
            } else {
                for i in 0..4 {
                    for j in 0..5 {
                        m[i][j] = values[i * 5 + j];
                    }
                }
            }
        }
        1 => {
            let s = if values.len() == 1 { values[0] } else { 1.0 };
            m[0] = [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[1] = [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[2] = [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        2 => {
            let angle_deg = if values.len() == 1 { values[0] } else { 0.0 };
            let angle = angle_deg.to_radians();
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            m[0] = [0.213 + cos_a * 0.787 - sin_a * 0.213, 0.715 - cos_a * 0.715 - sin_a * 0.715, 0.072 - cos_a * 0.072 + sin_a * 0.928, 0.0, 0.0];
            m[1] = [0.213 - cos_a * 0.213 + sin_a * 0.143, 0.715 + cos_a * 0.285 + sin_a * 0.140, 0.072 - cos_a * 0.072 - sin_a * 0.283, 0.0, 0.0];
            m[2] = [0.213 - cos_a * 0.213 - sin_a * 0.787, 0.715 - cos_a * 0.715 + sin_a * 0.715, 0.072 + cos_a * 0.928 + sin_a * 0.072, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        3 => {
            m[0] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[1] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[2] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[3] = [0.2126, 0.7152, 0.0722, 0.0, 0.0];
        }
        _ => {
            set_identity(&mut m);
        }
    }

    for y in 0..h {
        for x in 0..w {
            let r = src[[y, x, 0]] as f32 / 255.0;
            let g = src[[y, x, 1]] as f32 / 255.0;
            let b = src[[y, x, 2]] as f32 / 255.0;
            let a = src[[y, x, 3]] as f32 / 255.0;

            let out_r = (m[0][0] * r + m[0][1] * g + m[0][2] * b + m[0][3] * a + m[0][4]).clamp(0.0, 1.0);
            let out_g = (m[1][0] * r + m[1][1] * g + m[1][2] * b + m[1][3] * a + m[1][4]).clamp(0.0, 1.0);
            let out_b = (m[2][0] * r + m[2][1] * g + m[2][2] * b + m[2][3] * a + m[2][4]).clamp(0.0, 1.0);
            let out_a = (m[3][0] * r + m[3][1] * g + m[3][2] * b + m[3][3] * a + m[3][4]).clamp(0.0, 1.0);

            dst[[y, x, 0]] = (out_r * 255.0) as u8;
            dst[[y, x, 1]] = (out_g * 255.0) as u8;
            dst[[y, x, 2]] = (out_b * 255.0) as u8;
            dst[[y, x, 3]] = (out_a * 255.0) as u8;
        }
    }
    dst
}

/// feColorMatrix - apply color transformation matrix
#[pyfunction]
pub fn fe_color_matrix<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    matrix_type: u8,
    values: Vec<f32>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_color_matrix_impl(&arr, matrix_type, &values).into_pyarray(py)
}
