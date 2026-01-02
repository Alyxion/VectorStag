//! Polygon fill and stroke operations

use pyo3::prelude::*;
use numpy::{PyArray2, IntoPyArray};
use ndarray::Array2;

/// Check if two line segments AB and CD intersect
#[inline]
fn ccw(px: f64, py: f64, qx: f64, qy: f64, rx: f64, ry: f64) -> bool {
    (ry - py) * (qx - px) > (qy - py) * (rx - px)
}

#[inline]
fn segments_intersect(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, dx: f64, dy: f64) -> bool {
    ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) &&
    ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)
}

/// Check if a polygon has self-intersecting edges
#[pyfunction]
pub fn is_self_intersecting(points: Vec<(f64, f64)>) -> bool {
    let n = points.len();
    if n < 4 {
        return false;
    }

    if n > 200 {
        return true;
    }

    let mut pts = points.clone();
    if pts[0] != pts[n - 1] {
        pts.push(pts[0]);
    }

    let n = pts.len() - 1;
    let max_checks: usize = 5000;
    let total_pairs = n * (n.saturating_sub(3)) / 2;

    if total_pairs <= max_checks {
        for i in 0..n {
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                if segments_intersect(
                    pts[i].0, pts[i].1, pts[i + 1].0, pts[i + 1].1,
                    pts[j].0, pts[j].1, pts[j + 1].0, pts[j + 1].1,
                ) {
                    return true;
                }
            }
        }
    } else {
        let mut checked = 0;
        for i in 0..n {
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                if segments_intersect(
                    pts[i].0, pts[i].1, pts[i + 1].0, pts[i + 1].1,
                    pts[j].0, pts[j].1, pts[j + 1].0, pts[j + 1].1,
                ) {
                    return true;
                }
                checked += 1;
                if checked >= max_checks {
                    return false;
                }
            }
        }
    }

    false
}

/// Fill a polygon using nonzero winding rule
#[pyfunction]
pub fn fill_polygon_nonzero<'py>(
    py: Python<'py>,
    points: Vec<(f64, f64)>,
    width: usize,
    height: usize,
    min_x: i32,
    min_y: i32,
) -> Bound<'py, PyArray2<u8>> {
    let n = points.len();
    if n < 3 {
        return Array2::<u8>::zeros((height, width)).into_pyarray(py);
    }

    let mut pts = points.clone();
    if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
        pts.push(pts[0]);
    }

    let mut edges: Vec<(f64, f64, f64, f64, i32)> = Vec::with_capacity(pts.len());

    for i in 0..pts.len() - 1 {
        let (mut x1, mut y1) = pts[i];
        let (mut x2, mut y2) = pts[i + 1];

        if (y1 - y2).abs() < 1e-10 {
            continue;
        }

        let direction = if y1 > y2 {
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
            -1
        } else {
            1
        };

        edges.push((x1, y1, x2, y2, direction));
    }

    let mut mask = Array2::<u8>::zeros((height, width));

    for y in 0..height {
        let screen_y = (y as i32 + min_y) as f64 + 0.5;
        let mut intersections: Vec<(f64, i32)> = Vec::with_capacity(edges.len());

        for &(x1, y1, x2, y2, direction) in &edges {
            if y1 <= screen_y && screen_y < y2 {
                let t = (screen_y - y1) / (y2 - y1);
                let x_int = x1 + t * (x2 - x1);
                intersections.push((x_int, direction));
            }
        }

        if intersections.is_empty() {
            continue;
        }

        intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut winding: i32 = 0;
        let mut prev_x: Option<f64> = None;

        for (x_int, direction) in intersections {
            if winding != 0 {
                if let Some(px) = prev_x {
                    let x_start = (px - min_x as f64).max(0.0) as usize;
                    let x_end = ((x_int - min_x as f64) as usize).min(width);
                    if x_start < x_end {
                        for x in x_start..x_end {
                            mask[[y, x]] = 255;
                        }
                    }
                }
            }
            winding += direction;
            prev_x = Some(x_int);
        }
    }

    mask.into_pyarray(py)
}

/// Fill a polygon using even-odd rule
#[pyfunction]
pub fn fill_polygon_evenodd<'py>(
    py: Python<'py>,
    points: Vec<(f64, f64)>,
    width: usize,
    height: usize,
    min_x: i32,
    min_y: i32,
) -> Bound<'py, PyArray2<u8>> {
    let n = points.len();
    if n < 3 {
        return Array2::<u8>::zeros((height, width)).into_pyarray(py);
    }

    let mut pts = points.clone();
    if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
        pts.push(pts[0]);
    }

    let mut edges: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(pts.len());

    for i in 0..pts.len() - 1 {
        let (mut x1, mut y1) = pts[i];
        let (mut x2, mut y2) = pts[i + 1];

        if (y1 - y2).abs() < 1e-10 {
            continue;
        }

        if y1 > y2 {
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
        }

        edges.push((x1, y1, x2, y2));
    }

    let mut mask = Array2::<u8>::zeros((height, width));

    for y in 0..height {
        let screen_y = (y as i32 + min_y) as f64 + 0.5;
        let mut intersections: Vec<f64> = Vec::with_capacity(edges.len());

        for &(x1, y1, x2, y2) in &edges {
            if y1 <= screen_y && screen_y < y2 {
                let t = (screen_y - y1) / (y2 - y1);
                let x_int = x1 + t * (x2 - x1);
                intersections.push(x_int);
            }
        }

        if intersections.is_empty() {
            continue;
        }

        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for pair in intersections.chunks(2) {
            if pair.len() == 2 {
                let x_start = (pair[0] - min_x as f64).max(0.0) as usize;
                let x_end = ((pair[1] - min_x as f64) as usize).min(width);
                if x_start < x_end {
                    for x in x_start..x_end {
                        mask[[y, x]] = 255;
                    }
                }
            }
        }
    }

    mask.into_pyarray(py)
}

/// Fill multiple polygons using even-odd rule
#[pyfunction]
pub fn fill_multi_polygon_evenodd<'py>(
    py: Python<'py>,
    polygons: Vec<Vec<(f64, f64)>>,
    width: usize,
    height: usize,
    min_x: i32,
    min_y: i32,
) -> Bound<'py, PyArray2<u8>> {
    if polygons.is_empty() {
        return Array2::<u8>::zeros((height, width)).into_pyarray(py);
    }

    let mut edges: Vec<(f64, f64, f64, f64)> = Vec::new();

    for points in &polygons {
        let n = points.len();
        if n < 3 {
            continue;
        }

        let mut pts = points.clone();
        if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
            pts.push(pts[0]);
        }

        for i in 0..pts.len() - 1 {
            let (mut x1, mut y1) = pts[i];
            let (mut x2, mut y2) = pts[i + 1];

            if (y1 - y2).abs() < 1e-10 {
                continue;
            }

            if y1 > y2 {
                std::mem::swap(&mut x1, &mut x2);
                std::mem::swap(&mut y1, &mut y2);
            }

            edges.push((x1, y1, x2, y2));
        }
    }

    let mut mask = Array2::<u8>::zeros((height, width));

    for y in 0..height {
        let screen_y = (y as i32 + min_y) as f64 + 0.5;
        let mut intersections: Vec<f64> = Vec::with_capacity(edges.len());

        for &(x1, y1, x2, y2) in &edges {
            if y1 <= screen_y && screen_y < y2 {
                let t = (screen_y - y1) / (y2 - y1);
                let x_int = x1 + t * (x2 - x1);
                intersections.push(x_int);
            }
        }

        if intersections.is_empty() {
            continue;
        }

        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for pair in intersections.chunks(2) {
            if pair.len() == 2 {
                let x_start = (pair[0] - min_x as f64).max(0.0) as usize;
                let x_end = ((pair[1] - min_x as f64) as usize).min(width);
                if x_start < x_end {
                    for x in x_start..x_end {
                        mask[[y, x]] = 255;
                    }
                }
            }
        }
    }

    mask.into_pyarray(py)
}

/// Fill multiple polygons using nonzero winding rule
#[pyfunction]
pub fn fill_multi_polygon_nonzero<'py>(
    py: Python<'py>,
    polygons: Vec<Vec<(f64, f64)>>,
    width: usize,
    height: usize,
    min_x: i32,
    min_y: i32,
) -> Bound<'py, PyArray2<u8>> {
    if polygons.is_empty() {
        return Array2::<u8>::zeros((height, width)).into_pyarray(py);
    }

    let mut edges: Vec<(f64, f64, f64, f64, i32)> = Vec::new();

    for points in &polygons {
        let n = points.len();
        if n < 3 {
            continue;
        }

        let mut pts = points.clone();
        if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
            pts.push(pts[0]);
        }

        for i in 0..pts.len() - 1 {
            let (mut x1, mut y1) = pts[i];
            let (mut x2, mut y2) = pts[i + 1];

            if (y1 - y2).abs() < 1e-10 {
                continue;
            }

            let direction = if y1 > y2 {
                std::mem::swap(&mut x1, &mut x2);
                std::mem::swap(&mut y1, &mut y2);
                -1
            } else {
                1
            };

            edges.push((x1, y1, x2, y2, direction));
        }
    }

    let mut mask = Array2::<u8>::zeros((height, width));

    for y in 0..height {
        let screen_y = (y as i32 + min_y) as f64 + 0.5;
        let mut intersections: Vec<(f64, i32)> = Vec::with_capacity(edges.len());

        for &(x1, y1, x2, y2, direction) in &edges {
            if y1 <= screen_y && screen_y < y2 {
                let t = (screen_y - y1) / (y2 - y1);
                let x_int = x1 + t * (x2 - x1);
                intersections.push((x_int, direction));
            }
        }

        if intersections.is_empty() {
            continue;
        }

        intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut winding: i32 = 0;
        let mut prev_x: Option<f64> = None;

        for (x_int, direction) in intersections {
            if winding != 0 {
                if let Some(px) = prev_x {
                    let x_start = (px - min_x as f64).max(0.0) as usize;
                    let x_end = ((x_int - min_x as f64) as usize).min(width);
                    if x_start < x_end {
                        for x in x_start..x_end {
                            mask[[y, x]] = 255;
                        }
                    }
                }
            }
            winding += direction;
            prev_x = Some(x_int);
        }
    }

    mask.into_pyarray(py)
}

/// Fill multiple polygons using union (any pixel inside any polygon is filled)
#[pyfunction]
pub fn fill_polygons_union<'py>(
    py: Python<'py>,
    polygons: Vec<Vec<(f64, f64)>>,
    width: usize,
    height: usize,
    min_x: i32,
    min_y: i32,
) -> Bound<'py, PyArray2<u8>> {
    if polygons.is_empty() {
        return Array2::<u8>::zeros((height, width)).into_pyarray(py);
    }

    let mut mask = Array2::<u8>::zeros((height, width));

    for points in &polygons {
        let n = points.len();
        if n < 3 {
            continue;
        }

        let mut pts = points.clone();
        if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
            pts.push(pts[0]);
        }

        let mut edges: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(pts.len());

        for i in 0..pts.len() - 1 {
            let (mut x1, mut y1) = pts[i];
            let (mut x2, mut y2) = pts[i + 1];

            if (y1 - y2).abs() < 1e-10 {
                continue;
            }

            if y1 > y2 {
                std::mem::swap(&mut x1, &mut x2);
                std::mem::swap(&mut y1, &mut y2);
            }

            edges.push((x1, y1, x2, y2));
        }

        for y in 0..height {
            let screen_y = (y as i32 + min_y) as f64 + 0.5;
            let mut intersections: Vec<f64> = Vec::with_capacity(edges.len());

            for &(x1, y1, x2, y2) in &edges {
                if y1 <= screen_y && screen_y < y2 {
                    let t = (screen_y - y1) / (y2 - y1);
                    let x_int = x1 + t * (x2 - x1);
                    intersections.push(x_int);
                }
            }

            if intersections.is_empty() {
                continue;
            }

            intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            for pair in intersections.chunks(2) {
                if pair.len() == 2 {
                    let x_start = (pair[0] - min_x as f64).max(0.0) as usize;
                    let x_end = ((pair[1] - min_x as f64) as usize).min(width);
                    if x_start < x_end {
                        for x in x_start..x_end {
                            mask[[y, x]] = 255;
                        }
                    }
                }
            }
        }
    }

    mask.into_pyarray(py)
}

/// Fill polygon with solid color and composite directly onto destination array
#[pyfunction]
pub fn fill_polygon_to_array<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    points: Vec<(f64, f64)>,
    r: u8, g: u8, b: u8, a: u8,
    fill_rule: u8,
) {
    let n = points.len();
    if n < 3 || a == 0 { return; }

    let mut dst_arr = dst.as_array_mut();
    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);

    let raw_min_x = points.iter().map(|p| p.0.floor() as i32).min().unwrap_or(0);
    let raw_max_x = points.iter().map(|p| p.0.ceil() as i32).max().unwrap_or(0);
    let raw_min_y = points.iter().map(|p| p.1.floor() as i32).min().unwrap_or(0);
    let raw_max_y = points.iter().map(|p| p.1.ceil() as i32).max().unwrap_or(0);

    let min_x = raw_min_x.max(0).min(dst_w as i32) as usize;
    let max_x = raw_max_x.max(0).min(dst_w as i32) as usize;
    let min_y = raw_min_y.max(0).min(dst_h as i32) as usize;
    let max_y = raw_max_y.max(0).min(dst_h as i32) as usize;

    if min_x >= max_x || min_y >= max_y { return; }

    let mut pts = points.clone();
    if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
        pts.push(pts[0]);
    }

    let mut edges: Vec<(f64, f64, f64, f64, i32)> = Vec::with_capacity(pts.len());
    for i in 0..pts.len() - 1 {
        let (mut x1, mut y1) = pts[i];
        let (mut x2, mut y2) = pts[i + 1];
        if (y1 - y2).abs() < 1e-10 { continue; }
        let direction = if y1 > y2 {
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
            -1
        } else { 1 };
        edges.push((x1, y1, x2, y2, direction));
    }

    let src_a = a as u32;
    let inv_src_a = 255 - src_a;

    for y in min_y..max_y {
        let screen_y = y as f64 + 0.5;
        let mut intersections: Vec<(f64, i32)> = Vec::with_capacity(edges.len());

        for &(x1, y1, x2, y2, direction) in &edges {
            if y1 <= screen_y && screen_y < y2 {
                let t = (screen_y - y1) / (y2 - y1);
                let x_int = x1 + t * (x2 - x1);
                intersections.push((x_int, direction));
            }
        }

        if intersections.is_empty() { continue; }
        intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut winding = 0i32;
        let mut i = 0;

        while i < intersections.len() {
            let x_start = intersections[i].0;
            winding += intersections[i].1;

            while i + 1 < intersections.len() {
                let inside = if fill_rule == 1 { winding % 2 != 0 } else { winding != 0 };
                if !inside { break; }
                i += 1;
                winding += intersections[i].1;
            }

            let x_end = if i < intersections.len() { intersections[i].0 } else { x_start };
            let px_start = (x_start.floor() as usize).max(min_x);
            let px_end = (x_end.ceil() as usize).min(max_x);

            for x in px_start..px_end {
                if src_a == 255 {
                    dst_arr[[y, x, 0]] = r;
                    dst_arr[[y, x, 1]] = g;
                    dst_arr[[y, x, 2]] = b;
                    dst_arr[[y, x, 3]] = 255;
                } else {
                    let dst_a = dst_arr[[y, x, 3]] as u32;
                    let out_a = src_a + (dst_a * inv_src_a / 255);
                    if out_a > 0 {
                        let dst_r = dst_arr[[y, x, 0]] as u32;
                        let dst_g = dst_arr[[y, x, 1]] as u32;
                        let dst_b = dst_arr[[y, x, 2]] as u32;
                        dst_arr[[y, x, 0]] = ((r as u32 * src_a + dst_r * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
                        dst_arr[[y, x, 1]] = ((g as u32 * src_a + dst_g * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
                        dst_arr[[y, x, 2]] = ((b as u32 * src_a + dst_b * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
                        dst_arr[[y, x, 3]] = out_a.min(255) as u8;
                    }
                }
            }
            i += 1;
        }
    }
}

/// Fill multiple polygons with solid color and composite directly onto destination array
#[pyfunction]
pub fn fill_multi_polygon_to_array<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    all_points: Vec<Vec<(f64, f64)>>,
    r: u8, g: u8, b: u8, a: u8,
    fill_rule: u8,
) {
    if all_points.is_empty() || a == 0 { return; }

    let mut dst_arr = dst.as_array_mut();
    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);

    let mut raw_min_x = i32::MAX;
    let mut raw_max_x = i32::MIN;
    let mut raw_min_y = i32::MAX;
    let mut raw_max_y = i32::MIN;

    let mut all_edges: Vec<(f64, f64, f64, f64, i32)> = Vec::new();

    for points in &all_points {
        let n = points.len();
        if n < 3 { continue; }

        for p in points {
            let px = p.0.floor() as i32;
            let py = p.1.floor() as i32;
            raw_min_x = raw_min_x.min(px);
            raw_max_x = raw_max_x.max(p.0.ceil() as i32);
            raw_min_y = raw_min_y.min(py);
            raw_max_y = raw_max_y.max(p.1.ceil() as i32);
        }

        let mut pts = points.clone();
        if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
            pts.push(pts[0]);
        }

        for i in 0..pts.len() - 1 {
            let (mut x1, mut y1) = pts[i];
            let (mut x2, mut y2) = pts[i + 1];
            if (y1 - y2).abs() < 1e-10 { continue; }
            let direction = if y1 > y2 {
                std::mem::swap(&mut x1, &mut x2);
                std::mem::swap(&mut y1, &mut y2);
                -1
            } else { 1 };
            all_edges.push((x1, y1, x2, y2, direction));
        }
    }

    if raw_min_x == i32::MAX || raw_max_x == i32::MIN { return; }

    let global_min_x = raw_min_x.max(0).min(dst_w as i32) as usize;
    let global_max_x = raw_max_x.max(0).min(dst_w as i32) as usize;
    let global_min_y = raw_min_y.max(0).min(dst_h as i32) as usize;
    let global_max_y = raw_max_y.max(0).min(dst_h as i32) as usize;

    if global_min_x >= global_max_x || global_min_y >= global_max_y { return; }

    let src_a = a as u32;
    let inv_src_a = 255 - src_a;

    for y in global_min_y..global_max_y {
        let screen_y = y as f64 + 0.5;
        let mut intersections: Vec<(f64, i32)> = Vec::with_capacity(all_edges.len());

        for &(x1, y1, x2, y2, direction) in &all_edges {
            if y1 <= screen_y && screen_y < y2 {
                let t = (screen_y - y1) / (y2 - y1);
                let x_int = x1 + t * (x2 - x1);
                intersections.push((x_int, direction));
            }
        }

        if intersections.is_empty() { continue; }
        intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut winding: i32 = 0;
        let mut prev_x: Option<f64> = None;

        for (x_int, direction) in intersections {
            let should_fill = if fill_rule == 1 { winding % 2 != 0 } else { winding != 0 };
            if should_fill {
                if let Some(px) = prev_x {
                    let px_start = (px.floor() as usize).max(global_min_x);
                    let px_end = (x_int.ceil() as usize).min(global_max_x);

                    for x in px_start..px_end {
                        if src_a == 255 {
                            dst_arr[[y, x, 0]] = r;
                            dst_arr[[y, x, 1]] = g;
                            dst_arr[[y, x, 2]] = b;
                            dst_arr[[y, x, 3]] = 255;
                        } else {
                            let dst_a = dst_arr[[y, x, 3]] as u32;
                            let out_a = src_a + (dst_a * inv_src_a / 255);
                            if out_a > 0 {
                                let dst_r = dst_arr[[y, x, 0]] as u32;
                                let dst_g = dst_arr[[y, x, 1]] as u32;
                                let dst_b = dst_arr[[y, x, 2]] as u32;
                                dst_arr[[y, x, 0]] = ((r as u32 * src_a + dst_r * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
                                dst_arr[[y, x, 1]] = ((g as u32 * src_a + dst_g * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
                                dst_arr[[y, x, 2]] = ((b as u32 * src_a + dst_b * dst_a * inv_src_a / 255) / out_a).min(255) as u8;
                                dst_arr[[y, x, 3]] = out_a.min(255) as u8;
                            }
                        }
                    }
                }
            }
            winding += direction;
            prev_x = Some(x_int);
        }
    }
}

/// Fill polygon with anti-aliased edges using subpixel coverage
#[pyfunction]
pub fn fill_polygon_aa_to_array<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    points: Vec<(f64, f64)>,
    r: u8, g: u8, b: u8, a: u8,
    fill_rule: u8,
) {
    let n = points.len();
    if n < 3 || a == 0 { return; }

    let mut dst_arr = dst.as_array_mut();
    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);

    let raw_min_x = points.iter().map(|p| p.0.floor() as i32).min().unwrap_or(0);
    let raw_max_x = points.iter().map(|p| p.0.ceil() as i32).max().unwrap_or(0);
    let raw_min_y = points.iter().map(|p| p.1.floor() as i32).min().unwrap_or(0);
    let raw_max_y = points.iter().map(|p| p.1.ceil() as i32).max().unwrap_or(0);

    let min_x = raw_min_x.max(0).min(dst_w as i32) as usize;
    let max_x = raw_max_x.max(0).min(dst_w as i32) as usize;
    let min_y = raw_min_y.max(0).min(dst_h as i32) as usize;
    let max_y = raw_max_y.max(0).min(dst_h as i32) as usize;

    if min_x >= max_x || min_y >= max_y { return; }

    let mut pts = points.clone();
    if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
        pts.push(pts[0]);
    }

    let mut edges: Vec<(f64, f64, f64, f64, i32)> = Vec::with_capacity(pts.len());
    for i in 0..pts.len() - 1 {
        let (mut x1, mut y1) = pts[i];
        let (mut x2, mut y2) = pts[i + 1];
        if (y1 - y2).abs() < 1e-10 { continue; }
        let direction = if y1 > y2 {
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
            -1
        } else { 1 };
        edges.push((x1, y1, x2, y2, direction));
    }

    let src_a_f = a as f64 / 255.0;
    let samples = [0.125, 0.375, 0.625, 0.875];
    let sample_weight = 0.25;

    for y in min_y..max_y {
        let mut coverage: Vec<f64> = vec![0.0; max_x - min_x];

        for &sample_offset in &samples {
            let screen_y = y as f64 + sample_offset;
            let mut intersections: Vec<(f64, i32)> = Vec::with_capacity(edges.len());

            for &(x1, y1, x2, y2, direction) in &edges {
                if y1 <= screen_y && screen_y < y2 {
                    let t = (screen_y - y1) / (y2 - y1);
                    let x_int = x1 + t * (x2 - x1);
                    intersections.push((x_int, direction));
                }
            }

            if intersections.is_empty() { continue; }
            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut winding = 0i32;
            let mut i = 0;

            while i < intersections.len() {
                let x_start = intersections[i].0;
                winding += intersections[i].1;

                while i + 1 < intersections.len() {
                    let inside = if fill_rule == 1 { winding % 2 != 0 } else { winding != 0 };
                    if !inside { break; }
                    i += 1;
                    winding += intersections[i].1;
                }

                let x_end = if i < intersections.len() { intersections[i].0 } else { x_start };
                let px_start_f = x_start - min_x as f64;
                let px_end_f = x_end - min_x as f64;

                let px_start_int = px_start_f.floor() as i32;
                let px_end_int = px_end_f.ceil() as i32;

                for px in px_start_int..px_end_int {
                    if px < 0 || px >= coverage.len() as i32 { continue; }
                    let px_u = px as usize;
                    let px_left = px as f64;
                    let px_right = px_left + 1.0;
                    let left_bound = px_left.max(px_start_f);
                    let right_bound = px_right.min(px_end_f);
                    let pixel_coverage = (right_bound - left_bound).max(0.0);
                    coverage[px_u] += pixel_coverage * sample_weight;
                }

                i += 1;
            }
        }

        for (i, &cov) in coverage.iter().enumerate() {
            if cov <= 0.0 { continue; }
            let x = min_x + i;

            let effective_alpha = (src_a_f * cov.min(1.0) * 255.0).round() as u32;
            if effective_alpha == 0 { continue; }

            let inv_alpha = 255 - effective_alpha;

            if effective_alpha >= 255 {
                dst_arr[[y, x, 0]] = r;
                dst_arr[[y, x, 1]] = g;
                dst_arr[[y, x, 2]] = b;
                dst_arr[[y, x, 3]] = 255;
            } else {
                let dst_a = dst_arr[[y, x, 3]] as u32;
                let out_a = effective_alpha + (dst_a * inv_alpha / 255);
                if out_a > 0 {
                    let dst_r = dst_arr[[y, x, 0]] as u32;
                    let dst_g = dst_arr[[y, x, 1]] as u32;
                    let dst_b = dst_arr[[y, x, 2]] as u32;
                    dst_arr[[y, x, 0]] = ((r as u32 * effective_alpha + dst_r * dst_a * inv_alpha / 255) / out_a).min(255) as u8;
                    dst_arr[[y, x, 1]] = ((g as u32 * effective_alpha + dst_g * dst_a * inv_alpha / 255) / out_a).min(255) as u8;
                    dst_arr[[y, x, 2]] = ((b as u32 * effective_alpha + dst_b * dst_a * inv_alpha / 255) / out_a).min(255) as u8;
                    dst_arr[[y, x, 3]] = out_a.min(255) as u8;
                }
            }
        }
    }
}

/// Fill multiple polygons with anti-aliased edges using subpixel coverage
#[pyfunction]
pub fn fill_multi_polygon_aa_to_array<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    all_points: Vec<Vec<(f64, f64)>>,
    r: u8, g: u8, b: u8, a: u8,
    fill_rule: u8,
) {
    if all_points.is_empty() || a == 0 { return; }

    let mut dst_arr = dst.as_array_mut();
    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);

    let mut raw_min_x = i32::MAX;
    let mut raw_max_x = i32::MIN;
    let mut raw_min_y = i32::MAX;
    let mut raw_max_y = i32::MIN;

    let mut all_edges: Vec<(f64, f64, f64, f64, i32)> = Vec::new();

    for points in &all_points {
        let n = points.len();
        if n < 3 { continue; }

        for p in points {
            let px = p.0.floor() as i32;
            let py = p.1.floor() as i32;
            raw_min_x = raw_min_x.min(px);
            raw_max_x = raw_max_x.max(p.0.ceil() as i32);
            raw_min_y = raw_min_y.min(py);
            raw_max_y = raw_max_y.max(p.1.ceil() as i32);
        }

        let mut pts = points.clone();
        if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
            pts.push(pts[0]);
        }

        for i in 0..pts.len() - 1 {
            let (mut x1, mut y1) = pts[i];
            let (mut x2, mut y2) = pts[i + 1];
            if (y1 - y2).abs() < 1e-10 { continue; }
            let direction = if y1 > y2 {
                std::mem::swap(&mut x1, &mut x2);
                std::mem::swap(&mut y1, &mut y2);
                -1
            } else { 1 };
            all_edges.push((x1, y1, x2, y2, direction));
        }
    }

    if raw_min_x == i32::MAX || raw_max_x == i32::MIN { return; }

    let min_x = raw_min_x.max(0).min(dst_w as i32) as usize;
    let max_x = raw_max_x.max(0).min(dst_w as i32) as usize;
    let min_y = raw_min_y.max(0).min(dst_h as i32) as usize;
    let max_y = raw_max_y.max(0).min(dst_h as i32) as usize;

    if min_x >= max_x || min_y >= max_y { return; }

    let src_a_f = a as f64 / 255.0;
    let samples = [0.125, 0.375, 0.625, 0.875];
    let sample_weight = 0.25;

    for y in min_y..max_y {
        let mut coverage: Vec<f64> = vec![0.0; max_x - min_x];

        for &sample_offset in &samples {
            let screen_y = y as f64 + sample_offset;
            let mut intersections: Vec<(f64, i32)> = Vec::with_capacity(all_edges.len());

            for &(x1, y1, x2, y2, direction) in &all_edges {
                if y1 <= screen_y && screen_y < y2 {
                    let t = (screen_y - y1) / (y2 - y1);
                    let x_int = x1 + t * (x2 - x1);
                    intersections.push((x_int, direction));
                }
            }

            if intersections.is_empty() { continue; }
            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut winding = 0i32;
            let mut i = 0;

            while i < intersections.len() {
                let x_start = intersections[i].0;
                winding += intersections[i].1;

                while i + 1 < intersections.len() {
                    let inside = if fill_rule == 1 { winding % 2 != 0 } else { winding != 0 };
                    if !inside { break; }
                    i += 1;
                    winding += intersections[i].1;
                }

                let x_end = if i < intersections.len() { intersections[i].0 } else { x_start };
                let px_start_f = x_start - min_x as f64;
                let px_end_f = x_end - min_x as f64;

                let px_start_int = px_start_f.floor() as i32;
                let px_end_int = px_end_f.ceil() as i32;

                for px in px_start_int..px_end_int {
                    if px < 0 || px >= coverage.len() as i32 { continue; }
                    let px_u = px as usize;
                    let px_left = px as f64;
                    let px_right = px_left + 1.0;
                    let left_bound = px_left.max(px_start_f);
                    let right_bound = px_right.min(px_end_f);
                    let pixel_coverage = (right_bound - left_bound).max(0.0);
                    coverage[px_u] += pixel_coverage * sample_weight;
                }

                i += 1;
            }
        }

        for (i, &cov) in coverage.iter().enumerate() {
            if cov <= 0.0 { continue; }
            let x = min_x + i;

            let effective_alpha = (src_a_f * cov.min(1.0) * 255.0).round() as u32;
            if effective_alpha == 0 { continue; }

            let inv_alpha = 255 - effective_alpha;

            if effective_alpha >= 255 {
                dst_arr[[y, x, 0]] = r;
                dst_arr[[y, x, 1]] = g;
                dst_arr[[y, x, 2]] = b;
                dst_arr[[y, x, 3]] = 255;
            } else {
                let dst_a = dst_arr[[y, x, 3]] as u32;
                let out_a = effective_alpha + (dst_a * inv_alpha / 255);
                if out_a > 0 {
                    let dst_r = dst_arr[[y, x, 0]] as u32;
                    let dst_g = dst_arr[[y, x, 1]] as u32;
                    let dst_b = dst_arr[[y, x, 2]] as u32;
                    dst_arr[[y, x, 0]] = ((r as u32 * effective_alpha + dst_r * dst_a * inv_alpha / 255) / out_a).min(255) as u8;
                    dst_arr[[y, x, 1]] = ((g as u32 * effective_alpha + dst_g * dst_a * inv_alpha / 255) / out_a).min(255) as u8;
                    dst_arr[[y, x, 2]] = ((b as u32 * effective_alpha + dst_b * dst_a * inv_alpha / 255) / out_a).min(255) as u8;
                    dst_arr[[y, x, 3]] = out_a.min(255) as u8;
                }
            }
        }
    }
}

/// Register polygon functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_self_intersecting, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygon_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygon_evenodd, m)?)?;
    m.add_function(wrap_pyfunction!(fill_multi_polygon_evenodd, m)?)?;
    m.add_function(wrap_pyfunction!(fill_multi_polygon_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygons_union, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygon_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(fill_multi_polygon_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygon_aa_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(fill_multi_polygon_aa_to_array, m)?)?;
    Ok(())
}
