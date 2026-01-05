//! feComponentTransfer - apply transfer function to each channel

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_component_transfer_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    func_r: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_g: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_b: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_a: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    fn apply_transfer(val: f32, func: &(u8, Vec<f32>, f32, f32, f32, f32, f32)) -> f32 {
        let (func_type, table, slope, intercept, amplitude, exponent, offset) = func;
        match func_type {
            0 => val,
            1 => {
                if table.len() < 2 { return val; }
                let n = table.len() - 1;
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                let frac = val * n as f32 - k as f32;
                table[k] * (1.0 - frac) + table[k + 1] * frac
            }
            2 => {
                if table.is_empty() { return val; }
                let n = table.len();
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                table[k]
            }
            3 => slope * val + intercept,
            4 => amplitude * val.powf(*exponent) + offset,
            _ => val,
        }
    }

    let funcs = [func_r, func_g, func_b, func_a];
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let val = src[[y, x, c]];
                let out = apply_transfer(val, funcs[c]);
                dst[[y, x, c]] = out.clamp(0.0, 1.0);
            }
        }
    }
    dst
}

pub fn fe_component_transfer_impl(
    src: &ndarray::ArrayView3<u8>,
    func_r: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_g: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_b: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_a: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    fn apply_transfer(val: f32, func: &(u8, Vec<f32>, f32, f32, f32, f32, f32)) -> f32 {
        let (func_type, table, slope, intercept, amplitude, exponent, offset) = func;
        match func_type {
            0 => val,
            1 => {
                if table.len() < 2 { return val; }
                let n = table.len() - 1;
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                let frac = val * n as f32 - k as f32;
                table[k] * (1.0 - frac) + table[k + 1] * frac
            }
            2 => {
                if table.is_empty() { return val; }
                let n = table.len();
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                table[k]
            }
            3 => slope * val + intercept,
            4 => amplitude * val.powf(*exponent) + offset,
            _ => val,
        }
    }

    let funcs = [func_r, func_g, func_b, func_a];
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let val = src[[y, x, c]] as f32 / 255.0;
                let out = apply_transfer(val, funcs[c]);
                dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

#[pyfunction]
pub fn fe_component_transfer<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    func_r: (u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_g: (u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_b: (u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_a: (u8, Vec<f32>, f32, f32, f32, f32, f32),
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_component_transfer_impl(&arr, &func_r, &func_g, &func_b, &func_a).into_pyarray(py)
}
