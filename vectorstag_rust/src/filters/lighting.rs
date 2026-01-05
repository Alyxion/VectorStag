//! feDiffuseLighting and feSpecularLighting

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

pub fn fe_diffuse_lighting_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    surface_scale: f32,
    diffuse_constant: f32,
    light_color: (f32, f32, f32), // normalized
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    specular_exponent: f32, limiting_cone_angle: f32,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    // For lighting, we need alpha map from src alpha
    // We treat 'src' as the source image, and use its alpha channel as height map
    
    let get_height = |x: i32, y: i32| -> f32 {
        if x < 0 || x >= w as i32 || y < 0 || y >= h as i32 {
            0.0
        } else {
            src[[y as usize, x as usize, 3]] * surface_scale
        }
    };

    let (lx, ly, lz) = if light_type == 0 {
        // Distant light
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();
        (az_rad.cos() * el_rad.cos(), az_rad.sin() * el_rad.cos(), el_rad.sin())
    } else {
        (0.0, 0.0, 0.0)
    };
    let (dist_lx, dist_ly, dist_lz) = (lx, ly, lz);

    for y in 0..h {
        let iy = y as i32;
        for x in 0..w {
            let ix = x as i32;
            
            // Calculate normal vector
            let z_nw = get_height(ix - 1, iy - 1);
            let z_n  = get_height(ix,     iy - 1);
            let z_ne = get_height(ix + 1, iy - 1);
            let z_w  = get_height(ix - 1, iy);
            let z_e  = get_height(ix + 1, iy);
            let z_sw = get_height(ix - 1, iy + 1);
            let z_s  = get_height(ix,     iy + 1);
            let z_se = get_height(ix + 1, iy + 1);

            let nx = -((z_ne + 2.0 * z_e + z_se) - (z_nw + 2.0 * z_w + z_sw)) / 4.0;
            let ny = -((z_sw + 2.0 * z_s + z_se) - (z_nw + 2.0 * z_n + z_ne)) / 4.0;
            let nz = 1.0;
            
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = (nx / len, ny / len, nz / len);

            // Light vector
            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                let z = get_height(ix, iy);
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - z;
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (dist_lx, dist_ly, dist_lz)
            };

            let n_dot_l = (nx * lx + ny * ly + nz * lz).max(0.0);
            
            let intensity = if light_type == 2 {
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos { 0.0 } else { l_dot_s.powf(specular_exponent) }
            } else {
                1.0
            };

            let diffuse = diffuse_constant * n_dot_l * intensity;
            
            // Result is alpha 1.0, color modulated by diffuse factor
            dst[[y, x, 0]] = (light_color.0 * diffuse).clamp(0.0, 1.0);
            dst[[y, x, 1]] = (light_color.1 * diffuse).clamp(0.0, 1.0);
            dst[[y, x, 2]] = (light_color.2 * diffuse).clamp(0.0, 1.0);
            dst[[y, x, 3]] = 1.0;
        }
    }
    dst
}

pub fn fe_diffuse_lighting_impl(
    src: &ndarray::ArrayView3<u8>,
    surface_scale: f32,
    diffuse_constant: f32,
    light_color: (u8, u8, u8),
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    specular_exponent: f32, limiting_cone_angle: f32,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let (lx, ly, lz) = if light_type == 0 {
        let az = azimuth.to_radians();
        let el = elevation.to_radians();
        (az.cos() * el.cos(), az.sin() * el.cos(), el.sin())
    } else {
        (0.0, 0.0, 0.0)
    };

    for y in 0..h {
        for x in 0..w {
            let get_height = |px: i32, py: i32| -> f32 {
                if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 { return 0.0; }
                src[[py as usize, px as usize, 3]] as f32 / 255.0 * surface_scale
            };

            let ix = x as i32;
            let iy = y as i32;

            let dx = get_height(ix + 1, iy) - get_height(ix - 1, iy);
            let dy = get_height(ix, iy + 1) - get_height(ix, iy - 1);

            let nx = -dx;
            let ny = -dy;
            let nz = 1.0f32;
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = (nx / n_len, ny / n_len, nz / n_len);

            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - get_height(ix, iy);
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (lx, ly, lz)
            };

            let n_dot_l = (nx * lx + ny * ly + nz * lz).max(0.0);

            let intensity = if light_type == 2 {
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos { 0.0 } else { l_dot_s.powf(specular_exponent) }
            } else {
                1.0
            };

            let diffuse = n_dot_l * diffuse_constant * intensity;

            dst[[y, x, 0]] = (light_color.0 as f32 * diffuse).clamp(0.0, 255.0) as u8;
            dst[[y, x, 1]] = (light_color.1 as f32 * diffuse).clamp(0.0, 255.0) as u8;
            dst[[y, x, 2]] = (light_color.2 as f32 * diffuse).clamp(0.0, 255.0) as u8;
            dst[[y, x, 3]] = 255;
        }
    }
    dst
}

/// feDiffuseLighting - diffuse lighting effect
#[pyfunction]
pub fn fe_diffuse_lighting<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    surface_scale: f32,
    diffuse_constant: f32,
    light_color: (u8, u8, u8),
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    specular_exponent: f32, limiting_cone_angle: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_diffuse_lighting_impl(
        &arr, surface_scale, diffuse_constant, light_color, light_type,
        azimuth, elevation, light_x, light_y, light_z,
        points_at_x, points_at_y, points_at_z, specular_exponent, limiting_cone_angle
    ).into_pyarray(py)
}

pub fn fe_specular_lighting_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    surface_scale: f32,
    specular_constant: f32,
    specular_exponent: f32,
    light_color: (f32, f32, f32), // normalized
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    spot_exponent: f32, limiting_cone_angle: f32,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    let get_height = |x: i32, y: i32| -> f32 {
        if x < 0 || x >= w as i32 || y < 0 || y >= h as i32 {
            0.0
        } else {
            src[[y as usize, x as usize, 3]] * surface_scale
        }
    };

    let (dist_lx, dist_ly, dist_lz) = if light_type == 0 {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();
        (az_rad.cos() * el_rad.cos(), az_rad.sin() * el_rad.cos(), el_rad.sin())
    } else {
        (0.0, 0.0, 0.0)
    };

    // Eye vector (0, 0, 1) usually
    let (ex, ey, ez) = (0.0, 0.0, 1.0);

    for y in 0..h {
        let iy = y as i32;
        for x in 0..w {
            let ix = x as i32;
            
            // Calculate normal vector
            let z_nw = get_height(ix - 1, iy - 1);
            let z_n  = get_height(ix,     iy - 1);
            let z_ne = get_height(ix + 1, iy - 1);
            let z_w  = get_height(ix - 1, iy);
            let z_e  = get_height(ix + 1, iy);
            let z_sw = get_height(ix - 1, iy + 1);
            let z_s  = get_height(ix,     iy + 1);
            let z_se = get_height(ix + 1, iy + 1);

            let nx = -((z_ne + 2.0 * z_e + z_se) - (z_nw + 2.0 * z_w + z_sw)) / 4.0;
            let ny = -((z_sw + 2.0 * z_s + z_se) - (z_nw + 2.0 * z_n + z_ne)) / 4.0;
            let nz = 1.0;
            
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = (nx / len, ny / len, nz / len);

            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                let z = get_height(ix, iy);
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - z;
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (dist_lx, dist_ly, dist_lz)
            };

            let hx = lx + ex;
            let hy = ly + ey;
            let hz = lz + ez;
            let h_len = (hx * hx + hy * hy + hz * hz).sqrt();
            let (hx, hy, hz) = if h_len > 1e-6 { (hx / h_len, hy / h_len, hz / h_len) } else { (0.0, 0.0, 1.0) };

            let n_dot_h = (nx * hx + ny * hy + nz * hz).max(0.0);

            let intensity = if light_type == 2 {
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos { 0.0 } else { l_dot_s.powf(spot_exponent) }
            } else {
                1.0
            };

            let specular = n_dot_h.powf(specular_exponent) * specular_constant * intensity;

            dst[[y, x, 0]] = (light_color.0 * specular).clamp(0.0, 1.0);
            dst[[y, x, 1]] = (light_color.1 * specular).clamp(0.0, 1.0);
            dst[[y, x, 2]] = (light_color.2 * specular).clamp(0.0, 1.0);
            let max_rgb = dst[[y, x, 0]].max(dst[[y, x, 1]]).max(dst[[y, x, 2]]);
            dst[[y, x, 3]] = max_rgb;
        }
    }
    dst
}

pub fn fe_specular_lighting_impl(
    src: &ndarray::ArrayView3<u8>,
    surface_scale: f32,
    specular_constant: f32,
    specular_exponent_param: f32,
    light_color: (u8, u8, u8),
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    spot_exponent: f32, limiting_cone_angle: f32,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let specular_exponent = specular_exponent_param.clamp(1.0, 128.0);
    let spec_constant = specular_constant.max(0.0);

    let (ex, ey, ez) = (0.0f32, 0.0, 1.0);

    let (dist_lx, dist_ly, dist_lz) = if light_type == 0 {
        let az = azimuth.to_radians();
        let el = elevation.to_radians();
        (az.cos() * el.cos(), az.sin() * el.cos(), el.sin())
    } else {
        (0.0, 0.0, 0.0)
    };

    for y in 0..h {
        for x in 0..w {
            let get_height = |px: i32, py: i32| -> f32 {
                if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 { return 0.0; }
                src[[py as usize, px as usize, 3]] as f32 / 255.0 * surface_scale
            };

            let ix = x as i32;
            let iy = y as i32;

            let dx = get_height(ix + 1, iy) - get_height(ix - 1, iy);
            let dy = get_height(ix, iy + 1) - get_height(ix, iy - 1);

            let nx = -dx;
            let ny = -dy;
            let nz = 1.0f32;
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = if n_len > 1e-6 { (nx / n_len, ny / n_len, nz / n_len) } else { (0.0, 0.0, 1.0) };

            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                let z = get_height(ix, iy);
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - z;
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (dist_lx, dist_ly, dist_lz)
            };

            let hx = lx + ex;
            let hy = ly + ey;
            let hz = lz + ez;
            let h_len = (hx * hx + hy * hy + hz * hz).sqrt();
            let (hx, hy, hz) = if h_len > 1e-6 { (hx / h_len, hy / h_len, hz / h_len) } else { (0.0, 0.0, 1.0) };

            let n_dot_h = (nx * hx + ny * hy + nz * hz).max(0.0);

            let intensity = if light_type == 2 {
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos { 0.0 } else { l_dot_s.powf(spot_exponent) }
            } else {
                1.0
            };

            let specular = n_dot_h.powf(specular_exponent) * spec_constant * intensity;

            dst[[y, x, 0]] = (light_color.0 as f32 * specular).clamp(0.0, 255.0) as u8;
            dst[[y, x, 1]] = (light_color.1 as f32 * specular).clamp(0.0, 255.0) as u8;
            dst[[y, x, 2]] = (light_color.2 as f32 * specular).clamp(0.0, 255.0) as u8;
            let max_rgb = dst[[y, x, 0]].max(dst[[y, x, 1]]).max(dst[[y, x, 2]]);
            dst[[y, x, 3]] = max_rgb;
        }
    }
    dst
}

/// feSpecularLighting - specular lighting effect
#[pyfunction]
pub fn fe_specular_lighting<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    surface_scale: f32,
    specular_constant: f32,
    specular_exponent_param: f32,
    light_color: (u8, u8, u8),
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    spot_exponent: f32, limiting_cone_angle: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_specular_lighting_impl(
        &arr, surface_scale, specular_constant, specular_exponent_param, light_color, light_type,
        azimuth, elevation, light_x, light_y, light_z,
        points_at_x, points_at_y, points_at_z, spot_exponent, limiting_cone_angle
    ).into_pyarray(py)
}

