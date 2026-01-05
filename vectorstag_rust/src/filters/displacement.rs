//! feDisplacementMap - displace pixels using map

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_displacement_map_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    map: &ndarray::ArrayView3<f32>,
    scale: f32,
    x_channel: u8,
    y_channel: u8,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let dx_val = map[[y, x, x_channel as usize]] - 0.5;
            let dy_val = map[[y, x, y_channel as usize]] - 0.5;

            let src_x = x as f32 + dx_val * scale;
            let src_y = y as f32 + dy_val * scale;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            for c in 0..4 {
                let get_pixel = |px: i32, py: i32| -> f32 {
                    if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                        0.0
                    } else {
                        src[[py as usize, px as usize, c]]
                    }
                };

                let v00 = get_pixel(x0, y0);
                let v10 = get_pixel(x1, y0);
                let v01 = get_pixel(x0, y1);
                let v11 = get_pixel(x1, y1);

                let v0 = v00 + fx * (v10 - v00);
                let v1 = v01 + fx * (v11 - v01);
                let v = v0 + fy * (v1 - v0);

                dst[[y, x, c]] = v.clamp(0.0, 1.0);
            }
        }
    }
    dst
}

pub fn fe_displacement_map_impl(
    src: &ndarray::ArrayView3<u8>,
    map: &ndarray::ArrayView3<u8>,
    scale: f32,
    x_channel: u8,
    y_channel: u8,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let dx_val = map[[y, x, x_channel as usize]] as f32 / 255.0 - 0.5;
            let dy_val = map[[y, x, y_channel as usize]] as f32 / 255.0 - 0.5;

            let src_x = x as f32 + dx_val * scale;
            let src_y = y as f32 + dy_val * scale;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            for c in 0..4 {
                let get_pixel = |px: i32, py: i32| -> f32 {
                    if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                        0.0
                    } else {
                        src[[py as usize, px as usize, c]] as f32
                    }
                };

                let v00 = get_pixel(x0, y0);
                let v10 = get_pixel(x1, y0);
                let v01 = get_pixel(x0, y1);
                let v11 = get_pixel(x1, y1);

                let v0 = v00 + fx * (v10 - v00);
                let v1 = v01 + fx * (v11 - v01);
                let v = v0 + fy * (v1 - v0);

                dst[[y, x, c]] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// feDisplacementMap - displace pixels using map
#[pyfunction]
pub fn fe_displacement_map<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    map: numpy::PyReadonlyArray3<'py, u8>,
    scale: f32,
    x_channel: u8,
    y_channel: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let src_arr = src.as_array();
    let map_arr = map.as_array();
    fe_displacement_map_impl(&src_arr, &map_arr, scale, x_channel, y_channel).into_pyarray(py)
}

