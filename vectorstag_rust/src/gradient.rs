//! Gradient interpolation and rendering operations

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

/// Interpolate gradient colors for an entire image
/// Takes t-values array and gradient stops, returns RGBA pixels
#[pyfunction]
pub fn interpolate_gradient_colors<'py>(
    py: Python<'py>,
    t: numpy::PyReadonlyArray2<'py, f32>,
    offsets: Vec<f32>,
    colors: Vec<(u8, u8, u8, u8)>,
    opacity: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let t_arr = t.as_array();
    let height = t_arr.shape()[0];
    let width = t_arr.shape()[1];

    let mut pixels = Array3::<u8>::zeros((height, width, 4));

    if offsets.is_empty() || colors.is_empty() {
        return pixels.into_pyarray(py);
    }

    let n_stops = offsets.len();

    for y in 0..height {
        for x in 0..width {
            let t_val = t_arr[[y, x]];

            let (r, g, b, a) = if t_val <= offsets[0] {
                colors[0]
            } else if t_val >= offsets[n_stops - 1] {
                colors[n_stops - 1]
            } else {
                // Binary search for the right interval
                let mut lo = 0;
                let mut hi = n_stops - 1;
                while lo < hi - 1 {
                    let mid = (lo + hi) / 2;
                    if offsets[mid] <= t_val {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }

                let s1_offset = offsets[lo];
                let s2_offset = offsets[hi];
                let (s1_r, s1_g, s1_b, s1_a) = colors[lo];
                let (s2_r, s2_g, s2_b, s2_a) = colors[hi];

                let denom = s2_offset - s1_offset;
                let ratio = if denom.abs() < 1e-10 {
                    0.0
                } else {
                    ((t_val - s1_offset) / denom).clamp(0.0, 1.0)
                };

                let r = s1_r as f32 + ratio * (s2_r as f32 - s1_r as f32);
                let g = s1_g as f32 + ratio * (s2_g as f32 - s1_g as f32);
                let b = s1_b as f32 + ratio * (s2_b as f32 - s1_b as f32);
                let a = s1_a as f32 + ratio * (s2_a as f32 - s1_a as f32);

                (r as u8, g as u8, b as u8, a as u8)
            };

            pixels[[y, x, 0]] = r;
            pixels[[y, x, 1]] = g;
            pixels[[y, x, 2]] = b;
            pixels[[y, x, 3]] = ((a as f32) * opacity) as u8;
        }
    }

    pixels.into_pyarray(py)
}

/// Interpolate color at a single t value - helper for gradient functions
#[inline]
fn interpolate_color_at_t(
    t_val: f32,
    offsets: &[f32],
    colors: &[(u8, u8, u8, u8)],
    opacity: f32,
) -> (u8, u8, u8, u8) {
    let n_stops = offsets.len();

    if t_val <= offsets[0] {
        let (r, g, b, a) = colors[0];
        return (r, g, b, ((a as f32) * opacity) as u8);
    } else if t_val >= offsets[n_stops - 1] {
        let (r, g, b, a) = colors[n_stops - 1];
        return (r, g, b, ((a as f32) * opacity) as u8);
    }

    // Binary search for the right interval
    let mut lo = 0;
    let mut hi = n_stops - 1;
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if offsets[mid] <= t_val {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let s1_offset = offsets[lo];
    let s2_offset = offsets[hi];
    let (s1_r, s1_g, s1_b, s1_a) = colors[lo];
    let (s2_r, s2_g, s2_b, s2_a) = colors[hi];

    let denom = s2_offset - s1_offset;
    let ratio = if denom.abs() < 1e-10 {
        0.0
    } else {
        ((t_val - s1_offset) / denom).clamp(0.0, 1.0)
    };

    let r = s1_r as f32 + ratio * (s2_r as f32 - s1_r as f32);
    let g = s1_g as f32 + ratio * (s2_g as f32 - s1_g as f32);
    let b = s1_b as f32 + ratio * (s2_b as f32 - s1_b as f32);
    let a = s1_a as f32 + ratio * (s2_a as f32 - s1_a as f32);

    (r as u8, g as u8, b as u8, ((a) * opacity) as u8)
}

/// Apply spread method to t value
#[inline]
fn apply_spread_method(t: f32, spread_method: u8) -> f32 {
    match spread_method {
        1 => t.rem_euclid(1.0), // repeat
        2 => { // reflect
            let t2 = t.rem_euclid(2.0);
            if t2 > 1.0 { 2.0 - t2 } else { t2 }
        },
        _ => t.clamp(0.0, 1.0), // pad (default)
    }
}

/// Create a linear gradient image directly (computes t and interpolates in one pass)
#[pyfunction]
pub fn create_linear_gradient_image<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    offset_x: i32,
    offset_y: i32,
    x1: f32, y1: f32,
    dx: f32, dy: f32,
    length: f32,
    offsets: Vec<f32>,
    colors: Vec<(u8, u8, u8, u8)>,
    opacity: f32,
    spread_method: u8, // 0=pad, 1=repeat, 2=reflect
) -> Bound<'py, numpy::PyArray3<u8>> {
    let mut pixels = Array3::<u8>::zeros((height, width, 4));

    if offsets.is_empty() || colors.is_empty() || length.abs() < 1e-10 {
        return pixels.into_pyarray(py);
    }

    for row in 0..height {
        let wy = (row as i32 + offset_y) as f32;
        for col in 0..width {
            let wx = (col as i32 + offset_x) as f32;
            let t_raw = ((wx - x1) * dx + (wy - y1) * dy) / length;
            let t = apply_spread_method(t_raw, spread_method);
            let (r, g, b, a) = interpolate_color_at_t(t, &offsets, &colors, opacity);
            pixels[[row, col, 0]] = r;
            pixels[[row, col, 1]] = g;
            pixels[[row, col, 2]] = b;
            pixels[[row, col, 3]] = a;
        }
    }

    pixels.into_pyarray(py)
}

/// Create a radial gradient image directly (computes t with inverse transform and interpolates in one pass)
#[pyfunction]
pub fn create_radial_gradient_image<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    offset_x: i32,
    offset_y: i32,
    cx: f32, cy: f32, radius: f32,
    inv_a: f32, inv_b: f32, inv_c: f32, inv_d: f32, inv_e: f32, inv_f: f32,
    offsets: Vec<f32>,
    colors: Vec<(u8, u8, u8, u8)>,
    opacity: f32,
    spread_method: u8, // 0=pad, 1=repeat, 2=reflect
) -> Bound<'py, numpy::PyArray3<u8>> {
    let mut pixels = Array3::<u8>::zeros((height, width, 4));

    if offsets.is_empty() || colors.is_empty() || radius.abs() < 1e-10 {
        return pixels.into_pyarray(py);
    }

    for row in 0..height {
        let wy = (row as i32 + offset_y) as f32;
        for col in 0..width {
            let wx = (col as i32 + offset_x) as f32;
            // Inverse transform to gradient space
            let gx = inv_a * wx + inv_b * wy + inv_e;
            let gy = inv_c * wx + inv_d * wy + inv_f;
            // Distance from center, normalized
            let dist = ((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy)).sqrt();
            let t_raw = dist / radius;
            let t = apply_spread_method(t_raw, spread_method);
            let (r, g, b, a) = interpolate_color_at_t(t, &offsets, &colors, opacity);
            pixels[[row, col, 0]] = r;
            pixels[[row, col, 1]] = g;
            pixels[[row, col, 2]] = b;
            pixels[[row, col, 3]] = a;
        }
    }

    pixels.into_pyarray(py)
}

/// Register gradient module functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interpolate_gradient_colors, m)?)?;
    m.add_function(wrap_pyfunction!(create_linear_gradient_image, m)?)?;
    m.add_function(wrap_pyfunction!(create_radial_gradient_image, m)?)?;
    Ok(())
}
