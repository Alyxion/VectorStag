//! feMorphology - erode or dilate

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

/// Van Herk-Gil-Werman algorithm for 1D sliding window min/max
#[inline]
fn vhg_sliding_minmax(data: &[u8], radius: usize, is_min: bool) -> Vec<u8> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let window = 2 * radius + 1;

    if window >= n {
        let result_val = if is_min {
            *data.iter().min().unwrap_or(&0)
        } else {
            *data.iter().max().unwrap_or(&0)
        };
        return vec![result_val; n];
    }

    let mut result = vec![0u8; n];
    let num_blocks = (n + window - 1) / window;
    let mut prefix = vec![0u8; n];
    let mut suffix = vec![0u8; n];

    for block in 0..num_blocks {
        let block_start = block * window;
        let block_end = ((block + 1) * window).min(n);

        let mut val = if is_min { 255u8 } else { 0u8 };
        for i in (block_start..block_end).rev() {
            if is_min { val = val.min(data[i]); } else { val = val.max(data[i]); }
            suffix[i] = val;
        }

        val = if is_min { 255u8 } else { 0u8 };
        for i in block_start..block_end {
            if is_min { val = val.min(data[i]); } else { val = val.max(data[i]); }
            prefix[i] = val;
        }
    }

    for i in 0..n {
        let left = if i >= radius { i - radius } else { 0 };
        let right = if i + radius < n { i + radius } else { n - 1 };
        if is_min {
            result[i] = suffix[left].min(prefix[right]);
        } else {
            result[i] = suffix[left].max(prefix[right]);
        }
    }

    result
}

/// Van Herk-Gil-Werman algorithm for 1D sliding window min/max (f32)
#[inline]
fn vhg_sliding_minmax_f32(data: &[f32], radius: usize, is_min: bool) -> Vec<f32> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let window = 2 * radius + 1;

    if window >= n {
        let result_val = if is_min {
            *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
        } else {
            *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
        };
        return vec![result_val; n];
    }

    let mut result = vec![0.0f32; n];
    let num_blocks = (n + window - 1) / window;
    let mut prefix = vec![0.0f32; n];
    let mut suffix = vec![0.0f32; n];

    for block in 0..num_blocks {
        let block_start = block * window;
        let block_end = ((block + 1) * window).min(n);

        let mut val = if is_min { f32::INFINITY } else { f32::NEG_INFINITY };
        for i in (block_start..block_end).rev() {
            if is_min { val = val.min(data[i]); } else { val = val.max(data[i]); }
            suffix[i] = val;
        }

        val = if is_min { f32::INFINITY } else { f32::NEG_INFINITY };
        for i in block_start..block_end {
            if is_min { val = val.min(data[i]); } else { val = val.max(data[i]); }
            prefix[i] = val;
        }
    }

    for i in 0..n {
        let left = if i >= radius { i - radius } else { 0 };
        let right = if i + radius < n { i + radius } else { n - 1 };
        if is_min {
            result[i] = suffix[left].min(prefix[right]);
        } else {
            result[i] = suffix[left].max(prefix[right]);
        }
    }

    result
}

pub fn fe_morphology_impl_f32(src: &ndarray::ArrayView3<f32>, operator: u8, radius_x: f32, radius_y: f32) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    let rx = radius_x.round() as usize;
    let ry = radius_y.round() as usize;
    let is_erode = operator == 0;

    if rx == 0 && ry == 0 {
        return src.to_owned();
    }

    let mut temp = Array3::<f32>::zeros((h, w, 4));

    if rx > 0 {
        for y in 0..h {
            for c in 0..4 {
                let row: Vec<f32> = (0..w).map(|x| src[[y, x, c]]).collect();
                let result = vhg_sliding_minmax_f32(&row, rx, is_erode);
                for x in 0..w { temp[[y, x, c]] = result[x]; }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { temp[[y, x, c]] = src[[y, x, c]]; }
            }
        }
    }

    let mut dst = Array3::<f32>::zeros((h, w, 4));

    if ry > 0 {
        for x in 0..w {
            for c in 0..4 {
                let col: Vec<f32> = (0..h).map(|y| temp[[y, x, c]]).collect();
                let result = vhg_sliding_minmax_f32(&col, ry, is_erode);
                for y in 0..h { dst[[y, x, c]] = result[y]; }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = temp[[y, x, c]]; }
            }
        }
    }
    dst
}

pub fn fe_morphology_impl(src: &ndarray::ArrayView3<u8>, operator: u8, radius_x: f32, radius_y: f32) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    let rx = radius_x.round() as usize;
    let ry = radius_y.round() as usize;
    let is_erode = operator == 0;

    if rx == 0 && ry == 0 {
        let mut dst = Array3::<u8>::zeros((h, w, 4));
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = src[[y, x, c]]; }
            }
        }
        return dst;
    }

    let mut temp = Array3::<u8>::zeros((h, w, 4));

    if rx > 0 {
        for y in 0..h {
            for c in 0..4 {
                let row: Vec<u8> = (0..w).map(|x| src[[y, x, c]]).collect();
                let result = vhg_sliding_minmax(&row, rx, is_erode);
                for x in 0..w { temp[[y, x, c]] = result[x]; }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { temp[[y, x, c]] = src[[y, x, c]]; }
            }
        }
    }

    let mut dst = Array3::<u8>::zeros((h, w, 4));

    if ry > 0 {
        for x in 0..w {
            for c in 0..4 {
                let col: Vec<u8> = (0..h).map(|y| temp[[y, x, c]]).collect();
                let result = vhg_sliding_minmax(&col, ry, is_erode);
                for y in 0..h { dst[[y, x, c]] = result[y]; }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = temp[[y, x, c]]; }
            }
        }
    }
    dst
}

/// feMorphology - erode or dilate
#[pyfunction]
pub fn fe_morphology<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    operator: u8,
    radius_x: f32,
    radius_y: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_morphology_impl(&arr, operator, radius_x, radius_y).into_pyarray(py)
}

