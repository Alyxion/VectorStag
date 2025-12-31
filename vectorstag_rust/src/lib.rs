use pyo3::prelude::*;
use pyo3::types::PyList;
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
/// Returns true if any non-adjacent edges intersect
#[pyfunction]
fn is_self_intersecting(points: Vec<(f64, f64)>) -> bool {
    let n = points.len();
    if n < 4 {
        return false;
    }

    // For very complex polygons, assume they might be self-intersecting
    if n > 200 {
        return true;
    }

    // Close the polygon if needed
    let mut pts = points.clone();
    if pts[0] != pts[n - 1] {
        pts.push(pts[0]);
    }

    let n = pts.len() - 1;
    let max_checks: usize = 5000;
    let total_pairs = n * (n.saturating_sub(3)) / 2;

    if total_pairs <= max_checks {
        // Full check for smaller polygons
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
        // Sample-based check for medium polygons
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
/// Returns a mask array (height x width) with 255 for filled pixels
#[pyfunction]
fn fill_polygon_nonzero<'py>(
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

    // Close the polygon if needed
    let mut pts = points.clone();
    if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
        pts.push(pts[0]);
    }

    // Build edge list (non-horizontal edges only)
    let mut edges: Vec<(f64, f64, f64, f64, i32)> = Vec::with_capacity(pts.len());

    for i in 0..pts.len() - 1 {
        let (mut x1, mut y1) = pts[i];
        let (mut x2, mut y2) = pts[i + 1];

        // Skip horizontal edges
        if (y1 - y2).abs() < 1e-10 {
            continue;
        }

        // Ensure y1 < y2, track direction
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

    // Scanline fill
    for y in 0..height {
        let screen_y = (y as i32 + min_y) as f64 + 0.5;

        // Find intersections with all edges
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

        // Sort by x
        intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Fill using winding count
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
/// Returns a mask array (height x width) with 255 for filled pixels
#[pyfunction]
fn fill_polygon_evenodd<'py>(
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

    // Close the polygon if needed
    let mut pts = points.clone();
    if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
        pts.push(pts[0]);
    }

    // Build edge list (non-horizontal edges only)
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

        // Even-odd: fill between pairs
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

/// Fill multiple polygons using even-odd rule (holes where they overlap)
#[pyfunction]
fn fill_multi_polygon_evenodd<'py>(
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

    // Collect all edges from all polygons
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
fn fill_multi_polygon_nonzero<'py>(
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

    // Collect all edges from all polygons with direction
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

            // Track direction: +1 if going down, -1 if going up
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

        // Fill using winding count
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
/// Used for stroke rendering where quads may overlap
#[pyfunction]
fn fill_polygons_union<'py>(
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

    // For each polygon, fill it using even-odd (simple convex polygon fill)
    for points in &polygons {
        let n = points.len();
        if n < 3 {
            continue;
        }

        // Get bounding box for this polygon
        let poly_min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let poly_max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let poly_min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let poly_max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let y_start = ((poly_min_y - min_y as f64).max(0.0) as usize).min(height);
        let y_end = ((poly_max_y - min_y as f64 + 1.0).max(0.0) as usize).min(height);

        // Close the polygon if needed
        let mut pts = points.clone();
        if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
            pts.push(pts[0]);
        }

        // Build edges for this polygon
        let mut edges: Vec<(f64, f64, f64, f64)> = Vec::new();
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

        // Fill using scanline
        for y in y_start..y_end {
            let screen_y = (y as i32 + min_y) as f64 + 0.5;

            let mut intersections: Vec<f64> = Vec::new();

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

            // Even-odd fill
            for pair in intersections.chunks(2) {
                if pair.len() == 2 {
                    let x_start = (pair[0] - min_x as f64).max(0.0) as usize;
                    let x_end = ((pair[1] - min_x as f64) as usize + 1).min(width);
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

/// Render closed polygon stroke to a mask buffer
/// Computes offset points and fills the stroke region
#[pyfunction]
fn render_stroke_closed_polygon<'py>(
    py: Python<'py>,
    points: Vec<(f64, f64)>,
    half_width: f64,
    miterlimit: f64,
    width: usize,
    height: usize,
    min_x: i32,
    min_y: i32,
    linejoin: &str,
) -> Bound<'py, PyArray2<u8>> {
    let n = points.len();
    if n < 3 {
        return Array2::<u8>::zeros((height, width)).into_pyarray(py);
    }

    let use_bevel = linejoin == "round" || linejoin == "bevel";

    // Compute left and right edge points
    let mut left_points: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut right_points: Vec<(f64, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        let p_prev = points[(i + n - 1) % n];
        let p_curr = points[i];
        let p_next = points[(i + 1) % n];

        // Direction vectors
        let d1 = normalize(subtract(p_curr, p_prev));
        let d2 = normalize(subtract(p_next, p_curr));

        // Perpendiculars
        let perp1 = (-d1.1, d1.0);
        let perp2 = (-d2.1, d2.0);

        let cross = d1.0 * d2.1 - d1.1 * d2.0;

        let (left_pt, right_pt) = if cross.abs() > 0.001 {
            if use_bevel {
                // For round/bevel joins, use the average perpendicular (shorter corner)
                let avg_perp = normalize((perp1.0 + perp2.0, perp1.1 + perp2.1));
                let left_pt = (p_curr.0 + avg_perp.0 * half_width, p_curr.1 + avg_perp.1 * half_width);
                let right_pt = (p_curr.0 - avg_perp.0 * half_width, p_curr.1 - avg_perp.1 * half_width);
                (left_pt, right_pt)
            } else {
                // Compute miter intersection
                let left_p1 = (p_curr.0 + perp1.0 * half_width, p_curr.1 + perp1.1 * half_width);
                let left_p2 = (p_curr.0 + perp2.0 * half_width, p_curr.1 + perp2.1 * half_width);
                let right_p1 = (p_curr.0 - perp1.0 * half_width, p_curr.1 - perp1.1 * half_width);
                let right_p2 = (p_curr.0 - perp2.0 * half_width, p_curr.1 - perp2.1 * half_width);

                let mut left_pt = line_intersection(left_p1, d1, left_p2, d2).unwrap_or(left_p1);
                let mut right_pt = line_intersection(right_p1, d1, right_p2, d2).unwrap_or(right_p1);

                // Apply miterlimit
                let max_miter = miterlimit * half_width;
                let left_dist = ((left_pt.0 - p_curr.0).powi(2) + (left_pt.1 - p_curr.1).powi(2)).sqrt();
                let right_dist = ((right_pt.0 - p_curr.0).powi(2) + (right_pt.1 - p_curr.1).powi(2)).sqrt();

                if left_dist > max_miter {
                    let avg_perp = normalize((perp1.0 + perp2.0, perp1.1 + perp2.1));
                    left_pt = (p_curr.0 + avg_perp.0 * half_width, p_curr.1 + avg_perp.1 * half_width);
                }
                if right_dist > max_miter {
                    let avg_perp = normalize((perp1.0 + perp2.0, perp1.1 + perp2.1));
                    right_pt = (p_curr.0 - avg_perp.0 * half_width, p_curr.1 - avg_perp.1 * half_width);
                }

                (left_pt, right_pt)
            }
        } else {
            // Nearly collinear
            let avg_perp = normalize((perp1.0 + perp2.0, perp1.1 + perp2.1));
            let left_pt = (p_curr.0 + avg_perp.0 * half_width, p_curr.1 + avg_perp.1 * half_width);
            let right_pt = (p_curr.0 - avg_perp.0 * half_width, p_curr.1 - avg_perp.1 * half_width);
            (left_pt, right_pt)
        };

        left_points.push(left_pt);
        right_points.push(right_pt);
    }

    // Build quads for each edge
    let mut quads: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        quads.push(vec![
            left_points[i],
            left_points[j],
            right_points[j],
            right_points[i],
        ]);
    }

    // Fill all quads using union
    let mut mask = Array2::<u8>::zeros((height, width));

    for quad in &quads {
        // Get bounding box
        let poly_min_y = quad.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let poly_max_y = quad.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let y_start = ((poly_min_y - min_y as f64).max(0.0) as usize).min(height);
        let y_end = ((poly_max_y - min_y as f64 + 1.0).max(0.0) as usize).min(height);

        // Build edges
        let mut edges: Vec<(f64, f64, f64, f64)> = Vec::new();
        for i in 0..quad.len() {
            let (mut x1, mut y1) = quad[i];
            let (mut x2, mut y2) = quad[(i + 1) % quad.len()];

            if (y1 - y2).abs() < 1e-10 {
                continue;
            }

            if y1 > y2 {
                std::mem::swap(&mut x1, &mut x2);
                std::mem::swap(&mut y1, &mut y2);
            }

            edges.push((x1, y1, x2, y2));
        }

        // Scanline fill
        for y in y_start..y_end {
            let screen_y = (y as i32 + min_y) as f64 + 0.5;

            let mut intersections: Vec<f64> = Vec::new();

            for &(x1, y1, x2, y2) in &edges {
                if y1 <= screen_y && screen_y < y2 {
                    let t = (screen_y - y1) / (y2 - y1);
                    let x_int = x1 + t * (x2 - x1);
                    intersections.push(x_int);
                }
            }

            if intersections.len() < 2 {
                continue;
            }

            intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Fill between pairs
            for pair in intersections.chunks(2) {
                if pair.len() == 2 {
                    let x_start = (pair[0] - min_x as f64).max(0.0) as usize;
                    let x_end = ((pair[1] - min_x as f64) as usize + 1).min(width);
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

// Helper functions for stroke rendering
#[inline]
fn subtract(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}

#[inline]
fn normalize(v: (f64, f64)) -> (f64, f64) {
    let len = (v.0 * v.0 + v.1 * v.1).sqrt();
    if len < 1e-10 {
        (0.0, 0.0)
    } else {
        (v.0 / len, v.1 / len)
    }
}

#[inline]
fn line_intersection(p1: (f64, f64), d1: (f64, f64), p2: (f64, f64), d2: (f64, f64)) -> Option<(f64, f64)> {
    let det = d1.0 * (-d2.1) - d1.1 * (-d2.0);
    if det.abs() < 1e-10 {
        return None;
    }
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let t = (dx * (-d2.1) - dy * (-d2.0)) / det;
    Some((p1.0 + t * d1.0, p1.1 + t * d1.1))
}

/// Interpolate gradient colors for an entire image
/// Takes t-values array and gradient stops, returns RGBA pixels
#[pyfunction]
fn interpolate_gradient_colors<'py>(
    py: Python<'py>,
    t: numpy::PyReadonlyArray2<'py, f32>,
    offsets: Vec<f32>,
    colors: Vec<(u8, u8, u8, u8)>,
    opacity: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;

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

            // Find the two stops that surround this t value
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
fn create_linear_gradient_image<'py>(
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
    use ndarray::Array3;

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
fn create_radial_gradient_image<'py>(
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
    use ndarray::Array3;

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

/// Sample points along a cubic bezier curve
/// Returns list of (x, y) tuples from t=1/n_samples to t=1 (excludes t=0)
#[pyfunction]
fn sample_cubic_bezier(
    x0: f64, y0: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
    x3: f64, y3: f64,
    n_samples: usize,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(n_samples);
    let n = n_samples as f64;

    for i in 1..=n_samples {
        let t = i as f64 / n;
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        let x = mt3 * x0 + 3.0 * mt2 * t * x1 + 3.0 * mt * t2 * x2 + t3 * x3;
        let y = mt3 * y0 + 3.0 * mt2 * t * y1 + 3.0 * mt * t2 * y2 + t3 * y3;
        points.push((x, y));
    }

    points
}

/// Sample points along a quadratic bezier curve
/// Returns list of (x, y) tuples from t=1/n_samples to t=1 (excludes t=0)
#[pyfunction]
fn sample_quadratic_bezier(
    x0: f64, y0: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
    n_samples: usize,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(n_samples);
    let n = n_samples as f64;

    for i in 1..=n_samples {
        let t = i as f64 / n;
        let mt = 1.0 - t;

        let x = mt * mt * x0 + 2.0 * mt * t * x1 + t * t * x2;
        let y = mt * mt * y0 + 2.0 * mt * t * y1 + t * t * y2;
        points.push((x, y));
    }

    points
}

/// SVG Path Command types for return values
#[derive(Debug, Clone)]
enum PathCmd {
    M(f64, f64),
    L(f64, f64),
    C(f64, f64, f64, f64, f64, f64),
    Q(f64, f64, f64, f64),
    Z,
}

/// Parse SVG path data into absolute commands
/// Returns a list of tuples representing commands
#[pyfunction]
fn parse_path<'py>(py: Python<'py>, d: &str) -> Bound<'py, PyList> {
    let commands = parse_path_internal(d);
    let result = PyList::empty(py);

    for cmd in commands {
        let tuple = match cmd {
            PathCmd::M(x, y) => ("M", x, y, 0.0, 0.0, 0.0, 0.0),
            PathCmd::L(x, y) => ("L", x, y, 0.0, 0.0, 0.0, 0.0),
            PathCmd::C(x1, y1, x2, y2, x, y) => ("C", x1, y1, x2, y2, x, y),
            PathCmd::Q(x1, y1, x, y) => ("Q", x1, y1, x, y, 0.0, 0.0),
            PathCmd::Z => ("Z", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };
        result.append(tuple).unwrap();
    }

    result
}

/// Internal path parsing function
fn parse_path_internal(d: &str) -> Vec<PathCmd> {
    let mut commands = Vec::new();
    let mut current_x: f64 = 0.0;
    let mut current_y: f64 = 0.0;
    let mut start_x: f64 = 0.0;
    let mut start_y: f64 = 0.0;
    let mut last_control: Option<(f64, f64)> = None;
    let mut last_cmd: Option<char> = None;

    // Tokenize the path
    let tokens = tokenize_path(d);
    let mut i = 0;

    while i < tokens.len() {
        let token = &tokens[i];

        // Determine if this is a command or implicit continuation
        let cmd = if token.len() == 1 && "MmZzLlHhVvCcSsQqTtAa".contains(&token[..]) {
            i += 1;
            token.chars().next().unwrap()
        } else {
            // Implicit command
            match last_cmd {
                Some('M') => 'L',
                Some('m') => 'l',
                Some(c) => c,
                None => { i += 1; continue; }
            }
        };

        let is_relative = cmd.is_lowercase();
        let cmd_upper = cmd.to_ascii_uppercase();

        match cmd_upper {
            'M' => {
                // Moveto
                let nums = get_nums(&tokens, &mut i, 2);
                if nums.len() < 2 { continue; }

                let (mut x, mut y) = (nums[0], nums[1]);
                if is_relative {
                    x += current_x;
                    y += current_y;
                }

                commands.push(PathCmd::M(x, y));
                current_x = x;
                current_y = y;
                start_x = x;
                start_y = y;
                last_control = None;

                // Additional pairs are lineto
                loop {
                    let nums = get_nums(&tokens, &mut i, 2);
                    if nums.len() < 2 { break; }
                    let (mut x, mut y) = (nums[0], nums[1]);
                    if is_relative {
                        x += current_x;
                        y += current_y;
                    }
                    commands.push(PathCmd::L(x, y));
                    current_x = x;
                    current_y = y;
                }
            }
            'Z' => {
                commands.push(PathCmd::Z);
                current_x = start_x;
                current_y = start_y;
                last_control = None;
            }
            'L' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 2);
                    if nums.len() < 2 { break; }
                    let (mut x, mut y) = (nums[0], nums[1]);
                    if is_relative {
                        x += current_x;
                        y += current_y;
                    }
                    commands.push(PathCmd::L(x, y));
                    current_x = x;
                    current_y = y;
                }
                last_control = None;
            }
            'H' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 1);
                    if nums.is_empty() { break; }
                    let mut x = nums[0];
                    if is_relative {
                        x += current_x;
                    }
                    commands.push(PathCmd::L(x, current_y));
                    current_x = x;
                }
                last_control = None;
            }
            'V' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 1);
                    if nums.is_empty() { break; }
                    let mut y = nums[0];
                    if is_relative {
                        y += current_y;
                    }
                    commands.push(PathCmd::L(current_x, y));
                    current_y = y;
                }
                last_control = None;
            }
            'C' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 6);
                    if nums.len() < 6 { break; }
                    let (mut x1, mut y1, mut x2, mut y2, mut x, mut y) =
                        (nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]);
                    if is_relative {
                        x1 += current_x;
                        y1 += current_y;
                        x2 += current_x;
                        y2 += current_y;
                        x += current_x;
                        y += current_y;
                    }
                    commands.push(PathCmd::C(x1, y1, x2, y2, x, y));
                    last_control = Some((x2, y2));
                    current_x = x;
                    current_y = y;
                }
            }
            'S' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 4);
                    if nums.len() < 4 { break; }
                    let (mut x2, mut y2, mut x, mut y) = (nums[0], nums[1], nums[2], nums[3]);
                    if is_relative {
                        x2 += current_x;
                        y2 += current_y;
                        x += current_x;
                        y += current_y;
                    }

                    // Calculate first control point as reflection
                    let (x1, y1) = if let Some((lx, ly)) = last_control {
                        if matches!(last_cmd, Some('C') | Some('c') | Some('S') | Some('s')) {
                            (2.0 * current_x - lx, 2.0 * current_y - ly)
                        } else {
                            (current_x, current_y)
                        }
                    } else {
                        (current_x, current_y)
                    };

                    commands.push(PathCmd::C(x1, y1, x2, y2, x, y));
                    last_control = Some((x2, y2));
                    current_x = x;
                    current_y = y;
                }
            }
            'Q' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 4);
                    if nums.len() < 4 { break; }
                    let (mut x1, mut y1, mut x, mut y) = (nums[0], nums[1], nums[2], nums[3]);
                    if is_relative {
                        x1 += current_x;
                        y1 += current_y;
                        x += current_x;
                        y += current_y;
                    }
                    commands.push(PathCmd::Q(x1, y1, x, y));
                    last_control = Some((x1, y1));
                    current_x = x;
                    current_y = y;
                }
            }
            'T' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 2);
                    if nums.len() < 2 { break; }
                    let (mut x, mut y) = (nums[0], nums[1]);
                    if is_relative {
                        x += current_x;
                        y += current_y;
                    }

                    // Calculate control point as reflection
                    let (x1, y1) = if let Some((lx, ly)) = last_control {
                        if matches!(last_cmd, Some('Q') | Some('q') | Some('T') | Some('t')) {
                            (2.0 * current_x - lx, 2.0 * current_y - ly)
                        } else {
                            (current_x, current_y)
                        }
                    } else {
                        (current_x, current_y)
                    };

                    commands.push(PathCmd::Q(x1, y1, x, y));
                    last_control = Some((x1, y1));
                    current_x = x;
                    current_y = y;
                }
            }
            'A' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 7);
                    if nums.len() < 7 { break; }
                    let (rx, ry, x_rot, large_arc, sweep, mut x, mut y) =
                        (nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], nums[6]);
                    if is_relative {
                        x += current_x;
                        y += current_y;
                    }

                    // Convert arc to bezier curves
                    let arc_cmds = arc_to_bezier(
                        current_x, current_y,
                        rx, ry, x_rot,
                        large_arc as i32, sweep as i32,
                        x, y
                    );
                    commands.extend(arc_cmds);
                    current_x = x;
                    current_y = y;
                }
                last_control = None;
            }
            _ => {}
        }

        last_cmd = Some(cmd);
    }

    commands
}

/// Tokenize path data into commands and numbers
fn tokenize_path(d: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = d.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i] as char;

        // Skip whitespace and commas
        if c.is_whitespace() || c == ',' {
            i += 1;
            continue;
        }

        // Check for command
        if "MmZzLlHhVvCcSsQqTtAa".contains(c) {
            tokens.push(c.to_string());
            i += 1;
            continue;
        }

        // Parse number
        if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() {
            let start = i;

            // Optional sign
            if bytes[i] as char == '-' || bytes[i] as char == '+' {
                i += 1;
            }

            // Integer part
            while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                i += 1;
            }

            // Decimal part
            if i < bytes.len() && bytes[i] as char == '.' {
                i += 1;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
            }

            // Scientific notation
            if i < bytes.len() && (bytes[i] as char == 'e' || bytes[i] as char == 'E') {
                i += 1;
                if i < bytes.len() && (bytes[i] as char == '-' || bytes[i] as char == '+') {
                    i += 1;
                }
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
            }

            if i > start {
                tokens.push(d[start..i].to_string());
            }
            continue;
        }

        i += 1;
    }

    tokens
}

/// Get numbers from tokens
fn get_nums(tokens: &[String], i: &mut usize, count: usize) -> Vec<f64> {
    let mut nums = Vec::with_capacity(count);

    while nums.len() < count && *i < tokens.len() {
        if let Ok(n) = tokens[*i].parse::<f64>() {
            nums.push(n);
            *i += 1;
        } else {
            break;
        }
    }

    nums
}

/// Convert SVG arc to cubic bezier curves
fn arc_to_bezier(x1: f64, y1: f64, rx: f64, ry: f64,
                 phi: f64, large_arc: i32, sweep: i32,
                 x2: f64, y2: f64) -> Vec<PathCmd> {
    let mut commands = Vec::new();

    // Handle degenerate cases
    if (x1 - x2).abs() < 1e-10 && (y1 - y2).abs() < 1e-10 {
        return commands;
    }

    if rx == 0.0 || ry == 0.0 {
        return vec![PathCmd::L(x2, y2)];
    }

    let mut rx = rx.abs();
    let mut ry = ry.abs();

    // Convert angle to radians
    let phi_rad = phi.to_radians();
    let cos_phi = phi_rad.cos();
    let sin_phi = phi_rad.sin();

    // Step 1: Compute (x1', y1')
    let dx = (x1 - x2) / 2.0;
    let dy = (y1 - y2) / 2.0;
    let x1p = cos_phi * dx + sin_phi * dy;
    let y1p = -sin_phi * dx + cos_phi * dy;

    // Correct radii if too small
    let lambda_sq = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if lambda_sq > 1.0 {
        let lambda_val = lambda_sq.sqrt();
        rx *= lambda_val;
        ry *= lambda_val;
    }

    // Step 2: Compute (cx', cy')
    let num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
    let den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
    let sq = if den > 0.0 { (num / den).max(0.0).sqrt() } else { 0.0 };

    let sq = if large_arc == sweep { -sq } else { sq };

    let cxp = sq * rx * y1p / ry;
    let cyp = -sq * ry * x1p / rx;

    // Step 3: Compute (cx, cy)
    let cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0;
    let cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0;

    // Step 4: Compute theta1 and dtheta
    fn angle(ux: f64, uy: f64, vx: f64, vy: f64) -> f64 {
        let n = (ux * ux + uy * uy).sqrt() * (vx * vx + vy * vy).sqrt();
        if n == 0.0 { return 0.0; }
        let c = (ux * vx + uy * vy) / n;
        let c = c.max(-1.0).min(1.0);
        let sign = if ux * vy - uy * vx >= 0.0 { 1.0 } else { -1.0 };
        sign * c.acos()
    }

    let theta1 = angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry);
    let mut dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                          (-x1p - cxp) / rx, (-y1p - cyp) / ry);

    if sweep == 0 && dtheta > 0.0 {
        dtheta -= 2.0 * std::f64::consts::PI;
    } else if sweep == 1 && dtheta < 0.0 {
        dtheta += 2.0 * std::f64::consts::PI;
    }

    // Split arc into segments of at most 90 degrees
    let n_segs = ((dtheta.abs() / (std::f64::consts::PI / 2.0)).ceil() as usize).max(1);
    let d_theta = dtheta / n_segs as f64;

    // Approximate each segment with a cubic bezier
    let mut t = theta1;
    for _ in 0..n_segs {
        let t2 = t + d_theta;

        // Control point distance
        let half_d = d_theta / 2.0;
        let tan_half = half_d.tan();
        let alpha = d_theta.sin() * ((4.0 + 3.0 * tan_half * tan_half).sqrt() - 1.0) / 3.0;

        // Start point
        let cos_t = t.cos();
        let sin_t = t.sin();
        let x_start = cx + rx * cos_phi * cos_t - ry * sin_phi * sin_t;
        let y_start = cy + rx * sin_phi * cos_t + ry * cos_phi * sin_t;

        // End point
        let cos_t2 = t2.cos();
        let sin_t2 = t2.sin();
        let x_end = cx + rx * cos_phi * cos_t2 - ry * sin_phi * sin_t2;
        let y_end = cy + rx * sin_phi * cos_t2 + ry * cos_phi * sin_t2;

        // Derivatives
        let dx_start = -rx * cos_phi * sin_t - ry * sin_phi * cos_t;
        let dy_start = -rx * sin_phi * sin_t + ry * cos_phi * cos_t;
        let dx_end = -rx * cos_phi * sin_t2 - ry * sin_phi * cos_t2;
        let dy_end = -rx * sin_phi * sin_t2 + ry * cos_phi * cos_t2;

        // Control points
        let cp1x = x_start + alpha * dx_start;
        let cp1y = y_start + alpha * dy_start;
        let cp2x = x_end - alpha * dx_end;
        let cp2y = y_end - alpha * dy_end;

        commands.push(PathCmd::C(cp1x, cp1y, cp2x, cp2y, x_end, y_end));

        t = t2;
    }

    commands
}

/// Sample elliptical arc points
/// Returns list of (x, y) tuples
#[pyfunction]
fn sample_arc(
    cx: f64, cy: f64,      // Center
    rx: f64, ry: f64,      // Radii
    start_angle: f64,      // Start angle in radians
    end_angle: f64,        // End angle in radians
    rotation: f64,         // X-axis rotation in radians
    n_samples: usize,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(n_samples);
    let cos_rot = rotation.cos();
    let sin_rot = rotation.sin();

    for i in 1..=n_samples {
        let t = i as f64 / n_samples as f64;
        let angle = start_angle + t * (end_angle - start_angle);

        // Point on unrotated ellipse
        let px = rx * angle.cos();
        let py = ry * angle.sin();

        // Apply rotation and translate to center
        let x = cx + px * cos_rot - py * sin_rot;
        let y = cy + px * sin_rot + py * cos_rot;
        points.push((x, y));
    }

    points
}

/// Fill polygon with solid color and composite directly onto destination array
/// This combines mask creation, color fill, and alpha compositing in one step
/// Avoids creating any intermediate PIL images
#[pyfunction]
fn fill_polygon_to_array<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,  // mutable destination RGBA
    points: Vec<(f64, f64)>,
    r: u8, g: u8, b: u8, a: u8,  // fill color with alpha
    fill_rule: u8,  // 0 = nonzero, 1 = evenodd
) {
    let n = points.len();
    if n < 3 || a == 0 {
        return;
    }

    let mut dst_arr = dst.as_array_mut();
    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);

    // Find bounding box (handle negative coordinates properly)
    let raw_min_x = points.iter().map(|p| p.0.floor() as i32).min().unwrap_or(0);
    let raw_max_x = points.iter().map(|p| p.0.ceil() as i32).max().unwrap_or(0);
    let raw_min_y = points.iter().map(|p| p.1.floor() as i32).min().unwrap_or(0);
    let raw_max_y = points.iter().map(|p| p.1.ceil() as i32).max().unwrap_or(0);

    // Clamp to image bounds (must check bounds BEFORE converting to usize)
    let min_x = raw_min_x.max(0).min(dst_w as i32) as usize;
    let max_x = raw_max_x.max(0).min(dst_w as i32) as usize;
    let min_y = raw_min_y.max(0).min(dst_h as i32) as usize;
    let max_y = raw_max_y.max(0).min(dst_h as i32) as usize;

    if min_x >= max_x || min_y >= max_y {
        return;
    }

    // Close the polygon if needed
    let mut pts = points.clone();
    if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
        pts.push(pts[0]);
    }

    // Build edge list (non-horizontal edges only)
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

    // Scanline fill with direct compositing
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

            // Find end of fill span
            while i + 1 < intersections.len() {
                let inside = if fill_rule == 1 {
                    winding % 2 != 0  // evenodd
                } else {
                    winding != 0  // nonzero
                };
                if !inside { break; }
                i += 1;
                winding += intersections[i].1;
            }

            let x_end = if i < intersections.len() { intersections[i].0 } else { x_start };

            // Fill pixels in span
            let px_start = (x_start.floor() as usize).max(min_x);
            let px_end = (x_end.ceil() as usize).min(max_x);

            for x in px_start..px_end {
                // Alpha composite the fill color onto destination
                if src_a == 255 {
                    // Fully opaque - just overwrite
                    dst_arr[[y, x, 0]] = r;
                    dst_arr[[y, x, 1]] = g;
                    dst_arr[[y, x, 2]] = b;
                    dst_arr[[y, x, 3]] = 255;
                } else {
                    // Alpha blend
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
/// Handles multiple contours (outer + holes) in one pass
#[pyfunction]
fn fill_multi_polygon_to_array<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    all_points: Vec<Vec<(f64, f64)>>,
    r: u8, g: u8, b: u8, a: u8,
    fill_rule: u8,
) {
    if all_points.is_empty() || a == 0 {
        return;
    }

    let mut dst_arr = dst.as_array_mut();
    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);

    // Find bounding box across all polygons (using i32 to handle negatives)
    let mut raw_min_x = i32::MAX;
    let mut raw_max_x = i32::MIN;
    let mut raw_min_y = i32::MAX;
    let mut raw_max_y = i32::MIN;

    // Build all edges
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

    // Check if we found any valid polygons
    if raw_min_x == i32::MAX || raw_max_x == i32::MIN {
        return;
    }

    // Clamp to image bounds (must check bounds BEFORE converting to usize)
    let global_min_x = raw_min_x.max(0).min(dst_w as i32) as usize;
    let global_max_x = raw_max_x.max(0).min(dst_w as i32) as usize;
    let global_min_y = raw_min_y.max(0).min(dst_h as i32) as usize;
    let global_max_y = raw_max_y.max(0).min(dst_h as i32) as usize;

    if global_min_x >= global_max_x || global_min_y >= global_max_y {
        return;
    }

    let src_a = a as u32;
    let inv_src_a = 255 - src_a;

    // Scanline fill - use same algorithm as fill_multi_polygon_nonzero
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

        // Fill using winding count (same as fill_multi_polygon_nonzero)
        let mut winding: i32 = 0;
        let mut prev_x: Option<f64> = None;

        for (x_int, direction) in intersections {
            // Check if we should fill from prev_x to current x
            let should_fill = if fill_rule == 1 {
                // Even-odd rule: fill when winding is odd
                winding % 2 != 0
            } else {
                // Nonzero rule: fill when winding is not zero
                winding != 0
            };

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

/// Alpha composite source onto destination in-place at given offset
/// Uses Porter-Duff over operator: out = src + dst * (1 - src_alpha)
#[pyfunction]
fn alpha_composite_inplace<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,  // mutable destination RGBA
    src: numpy::PyReadonlyArray3<'py, u8>,  // source RGBA
    offset_x: i32,
    offset_y: i32,
) {
    let src_arr = src.as_array();
    let mut dst_arr = dst.as_array_mut();

    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);
    let (src_h, src_w, _) = (src_arr.shape()[0], src_arr.shape()[1], src_arr.shape()[2]);

    // Calculate overlap region
    let start_x = offset_x.max(0) as usize;
    let start_y = offset_y.max(0) as usize;
    let end_x = ((offset_x + src_w as i32) as usize).min(dst_w);
    let end_y = ((offset_y + src_h as i32) as usize).min(dst_h);

    let src_start_x = (-offset_x).max(0) as usize;
    let src_start_y = (-offset_y).max(0) as usize;

    for dy in start_y..end_y {
        let sy = src_start_y + (dy - start_y);
        if sy >= src_h { break; }

        for dx in start_x..end_x {
            let sx = src_start_x + (dx - start_x);
            if sx >= src_w { break; }

            let src_a = src_arr[[sy, sx, 3]] as u32;
            if src_a == 0 { continue; }  // Skip fully transparent

            if src_a == 255 {
                // Fully opaque source - just copy
                dst_arr[[dy, dx, 0]] = src_arr[[sy, sx, 0]];
                dst_arr[[dy, dx, 1]] = src_arr[[sy, sx, 1]];
                dst_arr[[dy, dx, 2]] = src_arr[[sy, sx, 2]];
                dst_arr[[dy, dx, 3]] = 255;
            } else {
                // Alpha blending
                let dst_a = dst_arr[[dy, dx, 3]] as u32;
                let inv_src_a = 255 - src_a;

                // out_a = src_a + dst_a * (1 - src_a/255)
                let out_a = src_a + (dst_a * inv_src_a / 255);

                if out_a == 0 {
                    dst_arr[[dy, dx, 0]] = 0;
                    dst_arr[[dy, dx, 1]] = 0;
                    dst_arr[[dy, dx, 2]] = 0;
                    dst_arr[[dy, dx, 3]] = 0;
                } else {
                    // out_rgb = (src_rgb * src_a + dst_rgb * dst_a * (1 - src_a/255)) / out_a
                    for c in 0..3 {
                        let src_c = src_arr[[sy, sx, c]] as u32;
                        let dst_c = dst_arr[[dy, dx, c]] as u32;
                        let out_c = (src_c * src_a + dst_c * dst_a * inv_src_a / 255) / out_a;
                        dst_arr[[dy, dx, c]] = out_c.min(255) as u8;
                    }
                    dst_arr[[dy, dx, 3]] = out_a.min(255) as u8;
                }
            }
        }
    }
}

/// Resize RGBA image using box filter (area averaging for downscale)
#[pyfunction]
fn resize_rgba<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    new_width: usize,
    new_height: usize,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;

    let src_arr = src.as_array();
    let (src_h, src_w, _) = (src_arr.shape()[0], src_arr.shape()[1], src_arr.shape()[2]);

    let mut dst = Array3::<u8>::zeros((new_height, new_width, 4));

    // Fast path: exact 4x downscale (most common for antialias=4)
    if src_w == new_width * 4 && src_h == new_height * 4 {
        for dy in 0..new_height {
            let sy = dy * 4;
            for dx in 0..new_width {
                let sx = dx * 4;

                // Sum 16 pixels (4x4 block) with premultiplied alpha
                let mut sum_r = 0u32;
                let mut sum_g = 0u32;
                let mut sum_b = 0u32;
                let mut sum_a = 0u32;

                for oy in 0..4 {
                    for ox in 0..4 {
                        let a = src_arr[[sy + oy, sx + ox, 3]] as u32;
                        sum_r += src_arr[[sy + oy, sx + ox, 0]] as u32 * a;
                        sum_g += src_arr[[sy + oy, sx + ox, 1]] as u32 * a;
                        sum_b += src_arr[[sy + oy, sx + ox, 2]] as u32 * a;
                        sum_a += a;
                    }
                }

                if sum_a > 0 {
                    dst[[dy, dx, 0]] = (sum_r / sum_a).min(255) as u8;
                    dst[[dy, dx, 1]] = (sum_g / sum_a).min(255) as u8;
                    dst[[dy, dx, 2]] = (sum_b / sum_a).min(255) as u8;
                    dst[[dy, dx, 3]] = (sum_a / 16) as u8;
                }
            }
        }
        return dst.into_pyarray(py);
    }

    // Fast path: exact 2x downscale (for antialias=2)
    if src_w == new_width * 2 && src_h == new_height * 2 {
        for dy in 0..new_height {
            let sy = dy * 2;
            for dx in 0..new_width {
                let sx = dx * 2;

                let a00 = src_arr[[sy, sx, 3]] as u32;
                let a01 = src_arr[[sy, sx + 1, 3]] as u32;
                let a10 = src_arr[[sy + 1, sx, 3]] as u32;
                let a11 = src_arr[[sy + 1, sx + 1, 3]] as u32;
                let sum_a = a00 + a01 + a10 + a11;

                if sum_a > 0 {
                    let sum_r = src_arr[[sy, sx, 0]] as u32 * a00
                              + src_arr[[sy, sx + 1, 0]] as u32 * a01
                              + src_arr[[sy + 1, sx, 0]] as u32 * a10
                              + src_arr[[sy + 1, sx + 1, 0]] as u32 * a11;
                    let sum_g = src_arr[[sy, sx, 1]] as u32 * a00
                              + src_arr[[sy, sx + 1, 1]] as u32 * a01
                              + src_arr[[sy + 1, sx, 1]] as u32 * a10
                              + src_arr[[sy + 1, sx + 1, 1]] as u32 * a11;
                    let sum_b = src_arr[[sy, sx, 2]] as u32 * a00
                              + src_arr[[sy, sx + 1, 2]] as u32 * a01
                              + src_arr[[sy + 1, sx, 2]] as u32 * a10
                              + src_arr[[sy + 1, sx + 1, 2]] as u32 * a11;

                    dst[[dy, dx, 0]] = (sum_r / sum_a).min(255) as u8;
                    dst[[dy, dx, 1]] = (sum_g / sum_a).min(255) as u8;
                    dst[[dy, dx, 2]] = (sum_b / sum_a).min(255) as u8;
                    dst[[dy, dx, 3]] = (sum_a / 4) as u8;
                }
            }
        }
        return dst.into_pyarray(py);
    }

    // General case: box filter
    let scale_x = src_w as f32 / new_width as f32;
    let scale_y = src_h as f32 / new_height as f32;

    for dy in 0..new_height {
        let sy_start = (dy as f32 * scale_y) as usize;
        let sy_end = (((dy + 1) as f32 * scale_y) as usize).min(src_h);

        for dx in 0..new_width {
            let sx_start = (dx as f32 * scale_x) as usize;
            let sx_end = (((dx + 1) as f32 * scale_x) as usize).min(src_w);

            let mut sum_r = 0u64;
            let mut sum_g = 0u64;
            let mut sum_b = 0u64;
            let mut sum_a = 0u64;
            let mut count = 0u64;

            for sy in sy_start..sy_end {
                for sx in sx_start..sx_end {
                    let a = src_arr[[sy, sx, 3]] as u64;
                    sum_r += src_arr[[sy, sx, 0]] as u64 * a;
                    sum_g += src_arr[[sy, sx, 1]] as u64 * a;
                    sum_b += src_arr[[sy, sx, 2]] as u64 * a;
                    sum_a += a;
                    count += 1;
                }
            }

            if count > 0 {
                let avg_a = sum_a / count;
                dst[[dy, dx, 3]] = avg_a as u8;

                if avg_a > 0 {
                    dst[[dy, dx, 0]] = ((sum_r / sum_a).min(255)) as u8;
                    dst[[dy, dx, 1]] = ((sum_g / sum_a).min(255)) as u8;
                    dst[[dy, dx, 2]] = ((sum_b / sum_a).min(255)) as u8;
                }
            }
        }
    }

    dst.into_pyarray(py)
}

// ============================================================================
// SVG Filter Primitives
// ============================================================================

/// feFlood - fill entire region with solid color
#[pyfunction]
fn fe_flood<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    r: u8, g: u8, b: u8, a: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let mut pixels = Array3::<u8>::zeros((height, width, 4));
    for y in 0..height {
        for x in 0..width {
            pixels[[y, x, 0]] = r;
            pixels[[y, x, 1]] = g;
            pixels[[y, x, 2]] = b;
            pixels[[y, x, 3]] = a;
        }
    }
    pixels.into_pyarray(py)
}

/// feOffset - offset image by dx, dy
#[pyfunction]
fn fe_offset<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    dx: i32,
    dy: i32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let src_arr = src.as_array();
    let (h, w, _) = (src_arr.shape()[0], src_arr.shape()[1], src_arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        let src_y = y as i32 - dy;
        if src_y < 0 || src_y >= h as i32 { continue; }
        for x in 0..w {
            let src_x = x as i32 - dx;
            if src_x < 0 || src_x >= w as i32 { continue; }
            for c in 0..4 {
                dst[[y, x, c]] = src_arr[[src_y as usize, src_x as usize, c]];
            }
        }
    }
    dst.into_pyarray(py)
}

// Helper functions for HSL blend modes
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };

    let h = if (max - r).abs() < 1e-6 {
        let mut h = (g - b) / d;
        if g < b { h += 6.0; }
        h
    } else if (max - g).abs() < 1e-6 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };

    (h / 6.0, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-6 {
        return (l, l, l);
    }

    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;

    fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
        if t < 0.0 { t += 1.0; }
        if t > 1.0 { t -= 1.0; }
        if t < 1.0/6.0 { return p + (q - p) * 6.0 * t; }
        if t < 1.0/2.0 { return q; }
        if t < 2.0/3.0 { return p + (q - p) * (2.0/3.0 - t) * 6.0; }
        p
    }

    (hue_to_rgb(p, q, h + 1.0/3.0),
     hue_to_rgb(p, q, h),
     hue_to_rgb(p, q, h - 1.0/3.0))
}

fn luminosity(r: f32, g: f32, b: f32) -> f32 {
    0.3 * r + 0.59 * g + 0.11 * b
}

fn set_lum(r: f32, g: f32, b: f32, l: f32) -> (f32, f32, f32) {
    let d = l - luminosity(r, g, b);
    clip_color(r + d, g + d, b + d)
}

fn clip_color(mut r: f32, mut g: f32, mut b: f32) -> (f32, f32, f32) {
    let l = luminosity(r, g, b);
    let n = r.min(g).min(b);
    let x = r.max(g).max(b);

    if n < 0.0 {
        let d = l - n;
        if d.abs() > 1e-6 {
            r = l + (r - l) * l / d;
            g = l + (g - l) * l / d;
            b = l + (b - l) * l / d;
        }
    }
    if x > 1.0 {
        let d = x - l;
        if d.abs() > 1e-6 {
            r = l + (r - l) * (1.0 - l) / d;
            g = l + (g - l) * (1.0 - l) / d;
            b = l + (b - l) * (1.0 - l) / d;
        }
    }
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

fn saturation(r: f32, g: f32, b: f32) -> f32 {
    r.max(g).max(b) - r.min(g).min(b)
}

fn set_sat(r: f32, g: f32, b: f32, s: f32) -> (f32, f32, f32) {
    // Sort channels by value
    let mut vals = [(r, 0), (g, 1), (b, 2)];
    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let (min_v, _) = vals[0];
    let (mid_v, _) = vals[1];
    let (max_v, _) = vals[2];

    let mut result = [0.0f32; 3];

    if (max_v - min_v).abs() > 1e-6 {
        result[vals[1].1] = (mid_v - min_v) * s / (max_v - min_v);
        result[vals[2].1] = s;
    }
    result[vals[0].1] = 0.0;

    (result[0], result[1], result[2])
}

/// feBlend - blend two images using blend mode
/// mode: 0=normal, 1=multiply, 2=screen, 3=darken, 4=lighten, 5=overlay,
///       6=color-dodge, 7=color-burn, 8=hard-light, 9=soft-light, 10=difference, 11=exclusion,
///       12=hue, 13=saturation, 14=color, 15=luminosity
#[pyfunction]
fn fe_blend<'py>(
    py: Python<'py>,
    in1: numpy::PyReadonlyArray3<'py, u8>,
    in2: numpy::PyReadonlyArray3<'py, u8>,
    mode: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr1 = in1.as_array();
    let arr2 = in2.as_array();
    let (h, w, _) = (arr1.shape()[0], arr1.shape()[1], arr1.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a1 = arr1[[y, x, 3]] as f32 / 255.0;
            let a2 = arr2[[y, x, 3]] as f32 / 255.0;

            let r1 = arr1[[y, x, 0]] as f32 / 255.0;
            let g1 = arr1[[y, x, 1]] as f32 / 255.0;
            let b1 = arr1[[y, x, 2]] as f32 / 255.0;
            let r2 = arr2[[y, x, 0]] as f32 / 255.0;
            let g2 = arr2[[y, x, 1]] as f32 / 255.0;
            let b2 = arr2[[y, x, 2]] as f32 / 255.0;

            // HSL-based blend modes (12-15)
            let (br, bg, bb) = match mode {
                12 => {  // hue: hue from in1, saturation and luminosity from in2
                    let (h1, _, _) = rgb_to_hsl(r1, g1, b1);
                    let (_, s2, _) = rgb_to_hsl(r2, g2, b2);
                    let l2 = luminosity(r2, g2, b2);
                    let (r, g, b) = hsl_to_rgb(h1, s2, 0.5);
                    set_lum(r, g, b, l2)
                }
                13 => {  // saturation: saturation from in1, hue and luminosity from in2
                    let s1 = saturation(r1, g1, b1);
                    let (h2, _, _) = rgb_to_hsl(r2, g2, b2);
                    let l2 = luminosity(r2, g2, b2);
                    let (r, g, b) = set_sat(r2, g2, b2, s1);
                    set_lum(r, g, b, l2)
                }
                14 => {  // color: hue and saturation from in1, luminosity from in2
                    let l2 = luminosity(r2, g2, b2);
                    set_lum(r1, g1, b1, l2)
                }
                15 => {  // luminosity: luminosity from in1, hue and saturation from in2
                    let l1 = luminosity(r1, g1, b1);
                    set_lum(r2, g2, b2, l1)
                }
                _ => (0.0, 0.0, 0.0),  // handled per-channel below
            };

            if mode >= 12 && mode <= 15 {
                // HSL modes - apply compositing
                let out_r = br * a1 + r2 * a2 * (1.0 - a1);
                let out_g = bg * a1 + g2 * a2 * (1.0 - a1);
                let out_b = bb * a1 + b2 * a2 * (1.0 - a1);
                dst[[y, x, 0]] = (out_r * 255.0).clamp(0.0, 255.0) as u8;
                dst[[y, x, 1]] = (out_g * 255.0).clamp(0.0, 255.0) as u8;
                dst[[y, x, 2]] = (out_b * 255.0).clamp(0.0, 255.0) as u8;
            } else {
                // Per-channel modes
                for c in 0..3 {
                    let c1 = arr1[[y, x, c]] as f32 / 255.0;
                    let c2 = arr2[[y, x, c]] as f32 / 255.0;

                    let blended = match mode {
                        1 => c1 * c2,  // multiply
                        2 => 1.0 - (1.0 - c1) * (1.0 - c2),  // screen
                        3 => c1.min(c2),  // darken
                        4 => c1.max(c2),  // lighten
                        5 => {  // overlay
                            if c2 < 0.5 { 2.0 * c1 * c2 }
                            else { 1.0 - 2.0 * (1.0 - c1) * (1.0 - c2) }
                        }
                        6 => {  // color-dodge
                            if c1 >= 1.0 { 1.0 }
                            else { (c2 / (1.0 - c1)).min(1.0) }
                        }
                        7 => {  // color-burn
                            if c1 <= 0.0 { 0.0 }
                            else { 1.0 - ((1.0 - c2) / c1).min(1.0) }
                        }
                        8 => {  // hard-light
                            if c1 < 0.5 { 2.0 * c1 * c2 }
                            else { 1.0 - 2.0 * (1.0 - c1) * (1.0 - c2) }
                        }
                        9 => {  // soft-light
                            if c1 < 0.5 { c2 - (1.0 - 2.0 * c1) * c2 * (1.0 - c2) }
                            else {
                                let d = if c2 <= 0.25 { ((16.0 * c2 - 12.0) * c2 + 4.0) * c2 }
                                        else { c2.sqrt() };
                                c2 + (2.0 * c1 - 1.0) * (d - c2)
                            }
                        }
                        10 => (c1 - c2).abs(),  // difference
                        11 => c1 + c2 - 2.0 * c1 * c2,  // exclusion
                        _ => c1,  // normal - just use c1
                    };

                    // Alpha composite the blended result
                    let out = blended * a1 + c2 * a2 * (1.0 - a1);
                    dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
                }
            }

            // Alpha: standard over
            let out_a = a1 + a2 * (1.0 - a1);
            dst[[y, x, 3]] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    dst.into_pyarray(py)
}

/// feComposite - Porter-Duff compositing
/// operator: 0=over, 1=in, 2=out, 3=atop, 4=xor, 5=arithmetic
#[pyfunction]
fn fe_composite<'py>(
    py: Python<'py>,
    in1: numpy::PyReadonlyArray3<'py, u8>,
    in2: numpy::PyReadonlyArray3<'py, u8>,
    operator: u8,
    k1: f32, k2: f32, k3: f32, k4: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr1 = in1.as_array();
    let arr2 = in2.as_array();
    let (h, w, _) = (arr1.shape()[0], arr1.shape()[1], arr1.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a1 = arr1[[y, x, 3]] as f32 / 255.0;
            let a2 = arr2[[y, x, 3]] as f32 / 255.0;

            let (fa, fb) = match operator {
                0 => (1.0, 1.0 - a1),  // over
                1 => (a2, 0.0),  // in
                2 => (1.0 - a2, 0.0),  // out
                3 => (a2, 1.0 - a1),  // atop
                4 => (1.0 - a2, 1.0 - a1),  // xor
                5 => (0.0, 0.0),  // arithmetic (handled separately)
                _ => (1.0, 1.0 - a1),  // default: over
            };

            for c in 0..4 {
                let c1 = arr1[[y, x, c]] as f32 / 255.0;
                let c2 = arr2[[y, x, c]] as f32 / 255.0;

                let out = if operator == 5 {
                    // arithmetic: result = k1*i1*i2 + k2*i1 + k3*i2 + k4
                    k1 * c1 * c2 + k2 * c1 + k3 * c2 + k4
                } else {
                    c1 * fa + c2 * fb
                };

                dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst.into_pyarray(py)
}

/// feMerge - merge multiple layers (simple alpha composite stack)
#[pyfunction]
fn fe_merge<'py>(
    py: Python<'py>,
    layers: Vec<numpy::PyReadonlyArray3<'py, u8>>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;

    if layers.is_empty() {
        return Array3::<u8>::zeros((1, 1, 4)).into_pyarray(py);
    }

    let first = layers[0].as_array();
    let (h, w, _) = (first.shape()[0], first.shape()[1], first.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    // Copy first layer
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                dst[[y, x, c]] = first[[y, x, c]];
            }
        }
    }

    // Composite remaining layers on top
    for layer in layers.iter().skip(1) {
        let src = layer.as_array();
        for y in 0..h {
            for x in 0..w {
                let src_a = src[[y, x, 3]] as f32 / 255.0;
                if src_a == 0.0 { continue; }

                let dst_a = dst[[y, x, 3]] as f32 / 255.0;
                let out_a = src_a + dst_a * (1.0 - src_a);

                if out_a > 0.0 {
                    for c in 0..3 {
                        let src_c = src[[y, x, c]] as f32 / 255.0;
                        let dst_c = dst[[y, x, c]] as f32 / 255.0;
                        let out_c = (src_c * src_a + dst_c * dst_a * (1.0 - src_a)) / out_a;
                        dst[[y, x, c]] = (out_c * 255.0).clamp(0.0, 255.0) as u8;
                    }
                    dst[[y, x, 3]] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    dst.into_pyarray(py)
}

/// feColorMatrix - apply color transformation matrix
/// type: 0=matrix, 1=saturate, 2=hueRotate, 3=luminanceToAlpha
#[pyfunction]
fn fe_color_matrix<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    matrix_type: u8,
    values: Vec<f32>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    // Build 5x4 matrix based on type
    let mut m = [[0.0f32; 5]; 4];

    match matrix_type {
        0 => {
            // matrix: values is 20 floats for 5x4 matrix (row-major, rows are RGBA output)
            // Default to identity if values is empty or incomplete
            if values.is_empty() {
                m[0] = [1.0, 0.0, 0.0, 0.0, 0.0];
                m[1] = [0.0, 1.0, 0.0, 0.0, 0.0];
                m[2] = [0.0, 0.0, 1.0, 0.0, 0.0];
                m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
            } else {
                for i in 0..4 {
                    for j in 0..5 {
                        let idx = i * 5 + j;
                        // Default to identity matrix values for missing entries
                        let default = if i == j { 1.0 } else { 0.0 };
                        m[i][j] = if idx < values.len() { values[idx] } else { default };
                    }
                }
            }
        }
        1 => {
            // saturate: single value 0-1
            let s = if !values.is_empty() { values[0] } else { 1.0 };
            // Saturate matrix
            m[0] = [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[1] = [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[2] = [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        2 => {
            // hueRotate: angle in degrees
            let angle = if !values.is_empty() { values[0].to_radians() } else { 0.0 };
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            m[0] = [0.213 + cos_a * 0.787 - sin_a * 0.213,
                    0.715 - cos_a * 0.715 - sin_a * 0.715,
                    0.072 - cos_a * 0.072 + sin_a * 0.928, 0.0, 0.0];
            m[1] = [0.213 - cos_a * 0.213 + sin_a * 0.143,
                    0.715 + cos_a * 0.285 + sin_a * 0.140,
                    0.072 - cos_a * 0.072 - sin_a * 0.283, 0.0, 0.0];
            m[2] = [0.213 - cos_a * 0.213 - sin_a * 0.787,
                    0.715 - cos_a * 0.715 + sin_a * 0.715,
                    0.072 + cos_a * 0.928 + sin_a * 0.072, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        3 => {
            // luminanceToAlpha
            m[0] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[1] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[2] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[3] = [0.2126, 0.7152, 0.0722, 0.0, 0.0];
        }
        _ => {
            // Identity matrix
            m[0] = [1.0, 0.0, 0.0, 0.0, 0.0];
            m[1] = [0.0, 1.0, 0.0, 0.0, 0.0];
            m[2] = [0.0, 0.0, 1.0, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
    }

    for y in 0..h {
        for x in 0..w {
            let r = arr[[y, x, 0]] as f32 / 255.0;
            let g = arr[[y, x, 1]] as f32 / 255.0;
            let b = arr[[y, x, 2]] as f32 / 255.0;
            let a = arr[[y, x, 3]] as f32 / 255.0;

            for c in 0..4 {
                let out = m[c][0] * r + m[c][1] * g + m[c][2] * b + m[c][3] * a + m[c][4];
                dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    dst.into_pyarray(py)
}

/// feComponentTransfer - apply transfer function to each channel
/// func_type per channel: 0=identity, 1=table, 2=discrete, 3=linear, 4=gamma
#[pyfunction]
fn fe_component_transfer<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    func_r: (u8, Vec<f32>, f32, f32, f32, f32, f32),  // (type, table, slope, intercept, amplitude, exponent, offset)
    func_g: (u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_b: (u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_a: (u8, Vec<f32>, f32, f32, f32, f32, f32),
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    fn apply_transfer(val: f32, func: &(u8, Vec<f32>, f32, f32, f32, f32, f32)) -> f32 {
        let (func_type, table, slope, intercept, amplitude, exponent, offset) = func;
        match func_type {
            0 => val,  // identity
            1 => {  // table
                if table.len() < 2 { return val; }
                let n = table.len() - 1;
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                let frac = val * n as f32 - k as f32;
                table[k] * (1.0 - frac) + table[k + 1] * frac
            }
            2 => {  // discrete
                if table.is_empty() { return val; }
                let n = table.len();
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                table[k]
            }
            3 => slope * val + intercept,  // linear
            4 => amplitude * val.powf(*exponent) + offset,  // gamma
            _ => val,
        }
    }

    let funcs = [&func_r, &func_g, &func_b, &func_a];

    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let val = arr[[y, x, c]] as f32 / 255.0;
                let out = apply_transfer(val, funcs[c]);
                dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    dst.into_pyarray(py)
}

/// Van Herk-Gil-Werman algorithm for 1D sliding window min/max
/// Achieves O(1) per element instead of O(r) per element
#[inline]
fn vhg_sliding_minmax(data: &[u8], radius: usize, is_min: bool) -> Vec<u8> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let window = 2 * radius + 1;

    // Edge case: window larger than data
    if window >= n {
        let result_val = if is_min {
            *data.iter().min().unwrap_or(&0)
        } else {
            *data.iter().max().unwrap_or(&0)
        };
        return vec![result_val; n];
    }

    let mut result = vec![0u8; n];

    // Allocate prefix and suffix arrays
    let num_blocks = (n + window - 1) / window;
    let mut prefix = vec![0u8; n];
    let mut suffix = vec![0u8; n];

    // Compute prefix and suffix min/max for each block
    for block in 0..num_blocks {
        let block_start = block * window;
        let block_end = ((block + 1) * window).min(n);

        // Suffix: scan backward from block_end
        let mut val = if is_min { 255u8 } else { 0u8 };
        for i in (block_start..block_end).rev() {
            if is_min {
                val = val.min(data[i]);
            } else {
                val = val.max(data[i]);
            }
            suffix[i] = val;
        }

        // Prefix: scan forward from block_start
        val = if is_min { 255u8 } else { 0u8 };
        for i in block_start..block_end {
            if is_min {
                val = val.min(data[i]);
            } else {
                val = val.max(data[i]);
            }
            prefix[i] = val;
        }
    }

    // Compute result using suffix and prefix
    for i in 0..n {
        let left = if i >= radius { i - radius } else { 0 };
        let right = if i + radius < n { i + radius } else { n - 1 };

        // The answer for window [left, right] is combine(suffix[left], prefix[right])
        // But we need to be careful when they span block boundaries
        if is_min {
            result[i] = suffix[left].min(prefix[right]);
        } else {
            result[i] = suffix[left].max(prefix[right]);
        }
    }

    result
}

/// feMorphology - erode or dilate using separable VHG algorithm
/// operator: 0=erode, 1=dilate
#[pyfunction]
fn fe_morphology<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    operator: u8,
    radius_x: f32,
    radius_y: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);

    let rx = radius_x.round() as usize;
    let ry = radius_y.round() as usize;
    let is_erode = operator == 0;

    if rx == 0 && ry == 0 {
        // No morphology - just copy
        let mut dst = Array3::<u8>::zeros((h, w, 4));
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    dst[[y, x, c]] = arr[[y, x, c]];
                }
            }
        }
        return dst.into_pyarray(py);
    }

    // Separable morphology: horizontal pass then vertical pass
    // This reduces O(w*h*rx*ry) to O(w*h*(rx+ry)) with VHG giving O(w*h)

    // Horizontal pass (process each row)
    let mut temp = Array3::<u8>::zeros((h, w, 4));

    if rx > 0 {
        for y in 0..h {
            for c in 0..4 {
                // Extract row for this channel
                let row: Vec<u8> = (0..w).map(|x| arr[[y, x, c]]).collect();
                let result = vhg_sliding_minmax(&row, rx, is_erode);
                for x in 0..w {
                    temp[[y, x, c]] = result[x];
                }
            }
        }
    } else {
        // No horizontal morphology, just copy
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    temp[[y, x, c]] = arr[[y, x, c]];
                }
            }
        }
    }

    // Vertical pass (process each column)
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    if ry > 0 {
        for x in 0..w {
            for c in 0..4 {
                // Extract column for this channel
                let col: Vec<u8> = (0..h).map(|y| temp[[y, x, c]]).collect();
                let result = vhg_sliding_minmax(&col, ry, is_erode);
                for y in 0..h {
                    dst[[y, x, c]] = result[y];
                }
            }
        }
    } else {
        // No vertical morphology, just copy from temp
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    dst[[y, x, c]] = temp[[y, x, c]];
                }
            }
        }
    }

    dst.into_pyarray(py)
}

/// feConvolveMatrix - apply convolution kernel (optimized)
#[pyfunction]
fn fe_convolve_matrix<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    order_x: usize,
    order_y: usize,
    kernel: Vec<f32>,
    divisor: f32,
    bias: f32,
    target_x: usize,
    target_y: usize,
    edge_mode: u8,  // 0=duplicate, 1=wrap, 2=none
    preserve_alpha: bool,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    // Precompute: kernel / divisor and bias * 255
    let div = if divisor.abs() < 1e-10 { 1.0 } else { divisor };
    let scaled_kernel: Vec<f32> = kernel.iter().map(|k| k / div).collect();
    let bias_255 = bias * 255.0;

    let h_i = h as i32;
    let w_i = w as i32;
    let target_y_i = target_y as i32;
    let target_x_i = target_x as i32;

    let channels = if preserve_alpha { 3 } else { 4 };

    // Check for empty kernel
    if scaled_kernel.is_empty() || order_x == 0 || order_y == 0 {
        // Just copy source to destination
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    dst[[y, x, c]] = arr[[y, x, c]];
                }
            }
        }
        return dst.into_pyarray(py);
    }

    // Edge mode 0 (duplicate) is most common - optimize separately
    if edge_mode == 0 {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    let sy = (y_i + ky as i32 - target_y_i).clamp(0, h_i - 1) as usize;

                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }
                        let sx = (x_i + kx as i32 - target_x_i).clamp(0, w_i - 1) as usize;
                        let kernel_val = scaled_kernel[kernel_idx];

                        // Process all channels together for better cache usage
                        for c in 0..channels {
                            sum[c] += arr[[sy, sx, c]] as f32 * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    dst[[y, x, c]] = (sum[c] + bias_255).clamp(0.0, 255.0) as u8;
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = arr[[y, x, 3]];
                }
            }
        }
    } else {
        // General case for wrap and none edge modes
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }

                        let sy = y_i + ky as i32 - target_y_i;
                        let sx = x_i + kx as i32 - target_x_i;

                        let (sy, sx) = match edge_mode {
                            1 => (sy.rem_euclid(h_i), sx.rem_euclid(w_i)),  // wrap
                            _ => {
                                if sy < 0 || sy >= h_i || sx < 0 || sx >= w_i {
                                    continue;
                                }
                                (sy, sx)
                            }
                        };

                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += arr[[sy as usize, sx as usize, c]] as f32 * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    dst[[y, x, c]] = (sum[c] + bias_255).clamp(0.0, 255.0) as u8;
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = arr[[y, x, 3]];
                }
            }
        }
    }

    dst.into_pyarray(py)
}

/// feTurbulence - generate Perlin noise
/// type: 0=turbulence, 1=fractalNoise
#[pyfunction]
fn fe_turbulence<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: usize,
    seed: i32,
    noise_type: u8,  // 0=turbulence, 1=fractalNoise
    stitch_tiles: bool,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let mut pixels = Array3::<u8>::zeros((height, width, 4));

    // Simple Perlin noise implementation
    // Use seed for deterministic random gradients
    let gradients = generate_gradients(seed);

    for y in 0..height {
        for x in 0..width {
            for c in 0..4 {
                let mut noise = 0.0f64;
                let mut amplitude = 1.0f64;
                let mut freq_x = base_freq_x;
                let mut freq_y = base_freq_y;

                for _ in 0..num_octaves {
                    let nx = x as f64 * freq_x;
                    let ny = y as f64 * freq_y;

                    let n = perlin_noise(nx, ny, c, &gradients);

                    if noise_type == 0 {
                        noise += n.abs() * amplitude;  // turbulence
                    } else {
                        noise += n * amplitude;  // fractalNoise
                    }

                    amplitude *= 0.5;
                    freq_x *= 2.0;
                    freq_y *= 2.0;
                }

                // Map to 0-255
                let val = if noise_type == 0 {
                    noise  // already 0-1 for turbulence
                } else {
                    (noise + 1.0) * 0.5  // -1 to 1 -> 0 to 1 for fractalNoise
                };

                pixels[[y, x, c]] = (val * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    pixels.into_pyarray(py)
}

fn generate_gradients(seed: i32) -> [[f64; 2]; 256] {
    let mut gradients = [[0.0f64; 2]; 256];
    let mut rng = seed as u32;

    for i in 0..256 {
        // Simple LCG random number generator
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let angle = (rng as f64 / u32::MAX as f64) * std::f64::consts::PI * 2.0;
        gradients[i] = [angle.cos(), angle.sin()];
    }

    gradients
}

fn perlin_noise(x: f64, y: f64, channel: usize, gradients: &[[f64; 2]; 256]) -> f64 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    // Fade curves
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);

    // Hash coordinates
    let hash = |x: i32, y: i32, c: usize| -> usize {
        (((x.wrapping_mul(1619) ^ y.wrapping_mul(31337) ^ (c as i32 * 6971)) & 0xFF) as usize)
    };

    let g00 = &gradients[hash(x0, y0, channel)];
    let g10 = &gradients[hash(x1, y0, channel)];
    let g01 = &gradients[hash(x0, y1, channel)];
    let g11 = &gradients[hash(x1, y1, channel)];

    // Dot products
    let n00 = g00[0] * fx + g00[1] * fy;
    let n10 = g10[0] * (fx - 1.0) + g10[1] * fy;
    let n01 = g01[0] * fx + g01[1] * (fy - 1.0);
    let n11 = g11[0] * (fx - 1.0) + g11[1] * (fy - 1.0);

    // Interpolate
    let nx0 = n00 + u * (n10 - n00);
    let nx1 = n01 + u * (n11 - n01);

    nx0 + v * (nx1 - nx0)
}

/// feDisplacementMap - displace pixels using map
/// x_channel/y_channel: 0=R, 1=G, 2=B, 3=A
#[pyfunction]
fn fe_displacement_map<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    map: numpy::PyReadonlyArray3<'py, u8>,
    scale: f32,
    x_channel: u8,
    y_channel: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let src_arr = src.as_array();
    let map_arr = map.as_array();
    let (h, w, _) = (src_arr.shape()[0], src_arr.shape()[1], src_arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            // Get displacement values from map
            let dx_val = map_arr[[y, x, x_channel as usize]] as f32 / 255.0 - 0.5;
            let dy_val = map_arr[[y, x, y_channel as usize]] as f32 / 255.0 - 0.5;

            let src_x = x as f32 + dx_val * scale;
            let src_y = y as f32 + dy_val * scale;

            // Bilinear interpolation
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
                        src_arr[[py as usize, px as usize, c]] as f32
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

    dst.into_pyarray(py)
}

/// feTile - tile input image to fill region
#[pyfunction]
fn fe_tile<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    out_width: usize,
    out_height: usize,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let src_arr = src.as_array();
    let (src_h, src_w, _) = (src_arr.shape()[0], src_arr.shape()[1], src_arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((out_height, out_width, 4));

    if src_h == 0 || src_w == 0 {
        return dst.into_pyarray(py);
    }

    for y in 0..out_height {
        let src_y = y % src_h;
        for x in 0..out_width {
            let src_x = x % src_w;
            for c in 0..4 {
                dst[[y, x, c]] = src_arr[[src_y, src_x, c]];
            }
        }
    }

    dst.into_pyarray(py)
}

/// feDiffuseLighting - diffuse lighting effect
#[pyfunction]
fn fe_diffuse_lighting<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,  // bump map (uses alpha channel)
    surface_scale: f32,
    diffuse_constant: f32,
    light_color: (u8, u8, u8),
    // Light source parameters
    light_type: u8,  // 0=distant, 1=point, 2=spot
    azimuth: f32, elevation: f32,  // for distant
    light_x: f32, light_y: f32, light_z: f32,  // for point/spot
    points_at_x: f32, points_at_y: f32, points_at_z: f32,  // for spot
    specular_exponent: f32, limiting_cone_angle: f32,  // for spot
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    // Compute light vector for distant light
    let (lx, ly, lz) = if light_type == 0 {
        let az = azimuth.to_radians();
        let el = elevation.to_radians();
        (az.cos() * el.cos(), az.sin() * el.cos(), el.sin())
    } else {
        (0.0, 0.0, 0.0)  // Will be computed per-pixel for point/spot
    };

    for y in 0..h {
        for x in 0..w {
            // Compute normal from bump map (using Sobel-like filter)
            let get_height = |px: i32, py: i32| -> f32 {
                if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                    return 0.0;
                }
                arr[[py as usize, px as usize, 3]] as f32 / 255.0 * surface_scale
            };

            let ix = x as i32;
            let iy = y as i32;

            // Sobel gradients
            let dx = get_height(ix + 1, iy) - get_height(ix - 1, iy);
            let dy = get_height(ix, iy + 1) - get_height(ix, iy - 1);

            // Normal vector (pointing up)
            let nx = -dx;
            let ny = -dy;
            let nz = 1.0f32;
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = (nx / n_len, ny / n_len, nz / n_len);

            // Light vector
            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                // Point or spot light
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - get_height(ix, iy);
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (lx, ly, lz)
            };

            // N dot L
            let n_dot_l = (nx * lx + ny * ly + nz * lz).max(0.0);

            // Apply spot light cone
            let intensity = if light_type == 2 {
                // Spot direction
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);

                // -L dot S
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();

                if l_dot_s < cone_cos {
                    0.0
                } else {
                    l_dot_s.powf(specular_exponent)
                }
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

    dst.into_pyarray(py)
}

/// feSpecularLighting - specular lighting effect
#[pyfunction]
fn fe_specular_lighting<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,  // bump map
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
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    // Clamp specularExponent to [1, 128] per SVG spec
    let specular_exponent = specular_exponent_param.clamp(1.0, 128.0);
    // Clamp specularConstant to >= 0
    let spec_constant = specular_constant.max(0.0);

    // Eye vector (looking down Z axis towards the surface)
    let (ex, ey, ez) = (0.0f32, 0.0, 1.0);

    // Pre-compute distant light direction
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
                if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                    return 0.0;
                }
                arr[[py as usize, px as usize, 3]] as f32 / 255.0 * surface_scale
            };

            let ix = x as i32;
            let iy = y as i32;

            // Compute surface normal from height differences (Sobel-like)
            let dx = get_height(ix + 1, iy) - get_height(ix - 1, iy);
            let dy = get_height(ix, iy + 1) - get_height(ix, iy - 1);

            // Normal vector (pointing towards viewer)
            let nx = -dx;
            let ny = -dy;
            let nz = 1.0f32;
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = if n_len > 1e-6 {
                (nx / n_len, ny / n_len, nz / n_len)
            } else {
                (0.0, 0.0, 1.0)
            };

            // Compute light direction based on light type
            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                // Point light or spotlight
                let z = get_height(ix, iy);
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - z;
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (dist_lx, dist_ly, dist_lz)
            };

            // Half vector between light and eye
            let hx = lx + ex;
            let hy = ly + ey;
            let hz = lz + ez;
            let h_len = (hx * hx + hy * hy + hz * hz).sqrt();
            let (hx, hy, hz) = if h_len > 1e-6 {
                (hx / h_len, hy / h_len, hz / h_len)
            } else {
                (0.0, 0.0, 1.0)
            };

            // N dot H (clamped to positive)
            let n_dot_h = (nx * hx + ny * hy + nz * hz).max(0.0);

            // Spotlight intensity falloff
            let intensity = if light_type == 2 {
                // Spotlight direction (from light to pointsAt)
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);

                // Light ray points from light to pixel, but we need the opposite
                // Dot product of (-L) with S = alignment between spotlight aim and light ray
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);

                // Apply cone angle and falloff
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos {
                    0.0
                } else {
                    l_dot_s.powf(spot_exponent)
                }
            } else {
                1.0
            };

            // Specular component
            let specular = n_dot_h.powf(specular_exponent) * spec_constant * intensity;

            dst[[y, x, 0]] = (light_color.0 as f32 * specular).clamp(0.0, 255.0) as u8;
            dst[[y, x, 1]] = (light_color.1 as f32 * specular).clamp(0.0, 255.0) as u8;
            dst[[y, x, 2]] = (light_color.2 as f32 * specular).clamp(0.0, 255.0) as u8;
            // Alpha = max(R, G, B) for specular lighting
            let max_rgb = dst[[y, x, 0]].max(dst[[y, x, 1]]).max(dst[[y, x, 2]]);
            dst[[y, x, 3]] = max_rgb;
        }
    }

    dst.into_pyarray(py)
}

/// Compute integral image (summed area table) for a single channel
#[inline]
fn compute_integral_image(src: &[f32], w: usize, h: usize, channel: usize) -> Vec<f64> {
    let mut integral = vec![0.0f64; (w + 1) * (h + 1)];
    let iw = w + 1;

    for y in 0..h {
        let mut row_sum = 0.0f64;
        for x in 0..w {
            row_sum += src[(y * w + x) * 4 + channel] as f64;
            integral[(y + 1) * iw + (x + 1)] = row_sum + integral[y * iw + (x + 1)];
        }
    }
    integral
}

/// Query sum of rectangle using integral image
#[inline]
fn integral_query(integral: &[f64], iw: usize, x1: usize, y1: usize, x2: usize, y2: usize) -> f64 {
    integral[y2 * iw + x2] - integral[y1 * iw + x2] - integral[y2 * iw + x1] + integral[y1 * iw + x1]
}

/// Single-pass box blur using integral images (true O(1) per pixel)
#[inline]
fn box_blur_integral(src: &[f32], dst: &mut [f32], w: usize, h: usize, rx: usize, ry: usize) {
    if w == 0 || h == 0 { return; }

    // Compute integral images for each channel
    let integral_r = compute_integral_image(src, w, h, 0);
    let integral_g = compute_integral_image(src, w, h, 1);
    let integral_b = compute_integral_image(src, w, h, 2);
    let integral_a = compute_integral_image(src, w, h, 3);

    let iw = w + 1;

    for y in 0..h {
        // Clamp vertical bounds
        let y1 = if y >= ry { y - ry } else { 0 };
        let y2 = (y + ry + 1).min(h);

        for x in 0..w {
            // Clamp horizontal bounds
            let x1 = if x >= rx { x - rx } else { 0 };
            let x2 = (x + rx + 1).min(w);

            let area = ((x2 - x1) * (y2 - y1)) as f64;
            let inv_area = if area > 0.0 { 1.0 / area } else { 0.0 };

            let idx = (y * w + x) * 4;
            dst[idx]     = (integral_query(&integral_r, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 1] = (integral_query(&integral_g, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 2] = (integral_query(&integral_b, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 3] = (integral_query(&integral_a, iw, x1, y1, x2, y2) * inv_area) as f32;
        }
    }
}

/// feGaussianBlur - optimized Gaussian blur using box blur approximation
#[pyfunction]
fn fe_gaussian_blur<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    std_dev_x: f32,
    std_dev_y: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);

    if std_dev_x < 0.5 && std_dev_y < 0.5 {
        // No blur needed - just copy
        let mut dst = Array3::<u8>::zeros((h, w, 4));
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    dst[[y, x, c]] = arr[[y, x, c]];
                }
            }
        }
        return dst.into_pyarray(py);
    }

    // Clamp stdDev to reasonable range
    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    // Box blur radius for Gaussian approximation (3 passes)
    let box_radius_x = ((12.0 * std_dev_x * std_dev_x / 3.0).sqrt() + 0.5).floor() as usize;
    let box_radius_y = ((12.0 * std_dev_y * std_dev_y / 3.0).sqrt() + 0.5).floor() as usize;

    let total_pixels = h * w * 4;

    // Premultiply alpha - use two buffers for ping-pong
    let mut buf_a = vec![0.0f32; total_pixels];
    let mut buf_b = vec![0.0f32; total_pixels];

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let a = arr[[y, x, 3]] as f32 / 255.0;
            buf_a[idx] = arr[[y, x, 0]] as f32 * a;
            buf_a[idx + 1] = arr[[y, x, 1]] as f32 * a;
            buf_a[idx + 2] = arr[[y, x, 2]] as f32 * a;
            buf_a[idx + 3] = arr[[y, x, 3]] as f32;
        }
    }

    // Apply 3 passes of box blur using integral images (O(1) per pixel)
    let mut current = &mut buf_a;
    let mut next = &mut buf_b;

    if box_radius_x > 0 || box_radius_y > 0 {
        for _ in 0..3 {
            box_blur_integral(current, next, w, h, box_radius_x, box_radius_y);
            std::mem::swap(&mut current, &mut next);
        }
    }

    // Un-premultiply alpha
    let mut dst = Array3::<u8>::zeros((h, w, 4));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let a = current[idx + 3];

            if a > 0.0 {
                let inv_a = 255.0 / a;
                dst[[y, x, 0]] = (current[idx] * inv_a).clamp(0.0, 255.0) as u8;
                dst[[y, x, 1]] = (current[idx + 1] * inv_a).clamp(0.0, 255.0) as u8;
                dst[[y, x, 2]] = (current[idx + 2] * inv_a).clamp(0.0, 255.0) as u8;
                dst[[y, x, 3]] = a.clamp(0.0, 255.0) as u8;
            }
        }
    }

    dst.into_pyarray(py)
}

/// feDropShadow - create drop shadow effect (SVG2 shorthand)
#[pyfunction]
fn fe_drop_shadow<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    dx: f32,
    dy: f32,
    std_dev_x: f32,
    std_dev_y: f32,
    flood_r: u8, flood_g: u8, flood_b: u8, flood_a: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);

    // Create shadow: offset the alpha channel, blur it, colorize, then composite original on top
    let dx_i = dx.round() as i32;
    let dy_i = dy.round() as i32;

    // Clamp stdDev
    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    // Step 1: Create offset alpha as float for blur
    let mut alpha = vec![0.0f32; h * w];
    for y in 0..h {
        let src_y = y as i32 - dy_i;
        if src_y < 0 || src_y >= h as i32 { continue; }
        for x in 0..w {
            let src_x = x as i32 - dx_i;
            if src_x < 0 || src_x >= w as i32 { continue; }
            let a = arr[[src_y as usize, src_x as usize, 3]] as f32;
            alpha[y * w + x] = a * (flood_a as f32 / 255.0);
        }
    }

    // Step 2: Blur alpha with O(1) sliding window
    if std_dev_x >= 0.5 || std_dev_y >= 0.5 {
        let box_radius_x = ((12.0 * std_dev_x * std_dev_x / 3.0).sqrt() + 0.5).floor() as usize;
        let box_radius_y = ((12.0 * std_dev_y * std_dev_y / 3.0).sqrt() + 0.5).floor() as usize;

        for _ in 0..3 {
            // Horizontal pass
            if box_radius_x > 0 {
                let mut next = vec![0.0f32; h * w];
                for y in 0..h {
                    let mut sum = 0.0f32;
                    for i in 0..box_radius_x.min(w) + 1 {
                        sum += alpha[y * w + i];
                    }
                    let mut left = 0i32 - box_radius_x as i32;
                    let mut right = box_radius_x as i32;

                    for x in 0..w {
                        if right < w as i32 && right > x as i32 {
                            sum += alpha[y * w + right as usize];
                        }
                        let count = (right.min(w as i32 - 1) - left.max(0) + 1) as f32;
                        next[y * w + x] = sum / count;
                        if left >= 0 {
                            sum -= alpha[y * w + left as usize];
                        }
                        left += 1;
                        right += 1;
                    }
                }
                alpha = next;
            }

            // Vertical pass
            if box_radius_y > 0 {
                let mut next = vec![0.0f32; h * w];
                for x in 0..w {
                    let mut sum = 0.0f32;
                    for i in 0..box_radius_y.min(h) + 1 {
                        sum += alpha[i * w + x];
                    }
                    let mut top = 0i32 - box_radius_y as i32;
                    let mut bottom = box_radius_y as i32;

                    for y in 0..h {
                        if bottom < h as i32 && bottom > y as i32 {
                            sum += alpha[bottom as usize * w + x];
                        }
                        let count = (bottom.min(h as i32 - 1) - top.max(0) + 1) as f32;
                        next[y * w + x] = sum / count;
                        if top >= 0 {
                            sum -= alpha[top as usize * w + x];
                        }
                        top += 1;
                        bottom += 1;
                    }
                }
                alpha = next;
            }
        }
    }

    // Step 3: Composite original on top of shadow
    let mut dst = Array3::<u8>::zeros((h, w, 4));
    for y in 0..h {
        for x in 0..w {
            let shadow_a = alpha[y * w + x] / 255.0;
            let src_a = arr[[y, x, 3]] as f32 / 255.0;

            let out_a = src_a + shadow_a * (1.0 - src_a);

            if out_a > 0.0 {
                for c in 0..3 {
                    let src_c = arr[[y, x, c]] as f32 / 255.0;
                    let shadow_c = match c { 0 => flood_r, 1 => flood_g, _ => flood_b } as f32 / 255.0;
                    let out_c = (src_c * src_a + shadow_c * shadow_a * (1.0 - src_a)) / out_a;
                    dst[[y, x, c]] = (out_c * 255.0).clamp(0.0, 255.0) as u8;
                }
                dst[[y, x, 3]] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    dst.into_pyarray(py)
}

/// Get SourceAlpha from input image (just the alpha channel as grayscale)
#[pyfunction]
fn get_source_alpha<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a = arr[[y, x, 3]];
            dst[[y, x, 0]] = 0;
            dst[[y, x, 1]] = 0;
            dst[[y, x, 2]] = 0;
            dst[[y, x, 3]] = a;
        }
    }

    dst.into_pyarray(py)
}

/// Fill polygon with anti-aliased edges using subpixel coverage calculation
/// This produces smoother edges by calculating exact pixel coverage
#[pyfunction]
fn fill_polygon_aa_to_array<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    points: Vec<(f64, f64)>,
    r: u8, g: u8, b: u8, a: u8,
    fill_rule: u8,  // 0 = nonzero, 1 = evenodd
) {
    let n = points.len();
    if n < 3 || a == 0 {
        return;
    }

    let mut dst_arr = dst.as_array_mut();
    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);

    // Find bounding box
    let raw_min_x = points.iter().map(|p| p.0.floor() as i32).min().unwrap_or(0);
    let raw_max_x = points.iter().map(|p| p.0.ceil() as i32).max().unwrap_or(0);
    let raw_min_y = points.iter().map(|p| p.1.floor() as i32).min().unwrap_or(0);
    let raw_max_y = points.iter().map(|p| p.1.ceil() as i32).max().unwrap_or(0);

    let min_x = raw_min_x.max(0).min(dst_w as i32) as usize;
    let max_x = raw_max_x.max(0).min(dst_w as i32) as usize;
    let min_y = raw_min_y.max(0).min(dst_h as i32) as usize;
    let max_y = raw_max_y.max(0).min(dst_h as i32) as usize;

    if min_x >= max_x || min_y >= max_y {
        return;
    }

    // Close the polygon if needed
    let mut pts = points.clone();
    if (pts[0].0 - pts[n - 1].0).abs() > 1e-10 || (pts[0].1 - pts[n - 1].1).abs() > 1e-10 {
        pts.push(pts[0]);
    }

    // Build edge list
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

    // For anti-aliasing, we sample at multiple y-positions within each pixel row
    // Using 4 samples per pixel (y + 0.125, 0.375, 0.625, 0.875)
    let samples = [0.125, 0.375, 0.625, 0.875];
    let sample_weight = 0.25;  // 1/4 for each sample

    for y in min_y..max_y {
        // Accumulator for coverage at each x position
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

                // Find end of fill span
                while i + 1 < intersections.len() {
                    let inside = if fill_rule == 1 {
                        winding % 2 != 0
                    } else {
                        winding != 0
                    };
                    if !inside { break; }
                    i += 1;
                    winding += intersections[i].1;
                }

                let x_end = if i < intersections.len() { intersections[i].0 } else { x_start };

                // Add coverage for this sample
                let px_start_f = x_start - min_x as f64;
                let px_end_f = x_end - min_x as f64;

                let px_start_int = px_start_f.floor() as i32;
                let px_end_int = px_end_f.ceil() as i32;

                for px in px_start_int..px_end_int {
                    if px < 0 || px >= coverage.len() as i32 { continue; }
                    let px_u = px as usize;
                    let px_left = px as f64;
                    let px_right = px_left + 1.0;

                    // Calculate how much of this pixel is covered in the x-direction
                    let left_bound = px_left.max(px_start_f);
                    let right_bound = px_right.min(px_end_f);
                    let pixel_coverage = (right_bound - left_bound).max(0.0);

                    coverage[px_u] += pixel_coverage * sample_weight;
                }

                i += 1;
            }
        }

        // Apply coverage to destination
        for (i, &cov) in coverage.iter().enumerate() {
            if cov <= 0.0 { continue; }
            let x = min_x + i;

            // Calculate effective alpha from source alpha and coverage
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
fn fill_multi_polygon_aa_to_array<'py>(
    _py: Python<'py>,
    mut dst: numpy::PyReadwriteArray3<'py, u8>,
    all_points: Vec<Vec<(f64, f64)>>,
    r: u8, g: u8, b: u8, a: u8,
    fill_rule: u8,
) {
    if all_points.is_empty() || a == 0 {
        return;
    }

    let mut dst_arr = dst.as_array_mut();
    let (dst_h, dst_w, _) = (dst_arr.shape()[0], dst_arr.shape()[1], dst_arr.shape()[2]);

    // Find global bounding box
    let mut raw_min_x = i32::MAX;
    let mut raw_max_x = i32::MIN;
    let mut raw_min_y = i32::MAX;
    let mut raw_max_y = i32::MIN;

    // Build all edges
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

    if raw_min_x == i32::MAX || raw_max_x == i32::MIN {
        return;
    }

    let min_x = raw_min_x.max(0).min(dst_w as i32) as usize;
    let max_x = raw_max_x.max(0).min(dst_w as i32) as usize;
    let min_y = raw_min_y.max(0).min(dst_h as i32) as usize;
    let max_y = raw_max_y.max(0).min(dst_h as i32) as usize;

    if min_x >= max_x || min_y >= max_y {
        return;
    }

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
                    let inside = if fill_rule == 1 {
                        winding % 2 != 0
                    } else {
                        winding != 0
                    };
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

/// Convert sRGB to linearRGB color space (for filter operations)
/// SVG filters default to linearRGB color-interpolation-filters
#[pyfunction]
fn srgb_to_linear<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    // Pre-compute lookup table for sRGB to linear conversion
    // This is much faster than computing per-pixel
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let c = i as f32 / 255.0;
        let linear = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
        lut[i] = (linear * 255.0).round() as u8;
    }

    for y in 0..h {
        for x in 0..w {
            // Convert RGB channels, preserve alpha
            dst[[y, x, 0]] = lut[arr[[y, x, 0]] as usize];
            dst[[y, x, 1]] = lut[arr[[y, x, 1]] as usize];
            dst[[y, x, 2]] = lut[arr[[y, x, 2]] as usize];
            dst[[y, x, 3]] = arr[[y, x, 3]];  // Alpha unchanged
        }
    }

    dst.into_pyarray(py)
}

/// Convert linearRGB to sRGB color space (after filter operations)
#[pyfunction]
fn linear_to_srgb<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    use ndarray::Array3;
    let arr = src.as_array();
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    // Pre-compute lookup table for linear to sRGB conversion
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let c = i as f32 / 255.0;
        let srgb = if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        lut[i] = (srgb * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    for y in 0..h {
        for x in 0..w {
            // Convert RGB channels, preserve alpha
            dst[[y, x, 0]] = lut[arr[[y, x, 0]] as usize];
            dst[[y, x, 1]] = lut[arr[[y, x, 1]] as usize];
            dst[[y, x, 2]] = lut[arr[[y, x, 2]] as usize];
            dst[[y, x, 3]] = arr[[y, x, 3]];  // Alpha unchanged
        }
    }

    dst.into_pyarray(py)
}

/// A Python module implemented in Rust for fast SVG rendering operations.
#[pymodule]
fn vectorstag_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_self_intersecting, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygon_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygon_evenodd, m)?)?;
    m.add_function(wrap_pyfunction!(fill_multi_polygon_evenodd, m)?)?;
    m.add_function(wrap_pyfunction!(fill_multi_polygon_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygons_union, m)?)?;
    m.add_function(wrap_pyfunction!(render_stroke_closed_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_gradient_colors, m)?)?;
    m.add_function(wrap_pyfunction!(create_linear_gradient_image, m)?)?;
    m.add_function(wrap_pyfunction!(create_radial_gradient_image, m)?)?;
    m.add_function(wrap_pyfunction!(sample_cubic_bezier, m)?)?;
    m.add_function(wrap_pyfunction!(sample_quadratic_bezier, m)?)?;
    m.add_function(wrap_pyfunction!(sample_arc, m)?)?;
    m.add_function(wrap_pyfunction!(parse_path, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_composite_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(resize_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygon_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(fill_multi_polygon_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(fill_polygon_aa_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(fill_multi_polygon_aa_to_array, m)?)?;
    // Filter primitives
    m.add_function(wrap_pyfunction!(fe_flood, m)?)?;
    m.add_function(wrap_pyfunction!(fe_offset, m)?)?;
    m.add_function(wrap_pyfunction!(fe_blend, m)?)?;
    m.add_function(wrap_pyfunction!(fe_composite, m)?)?;
    m.add_function(wrap_pyfunction!(fe_merge, m)?)?;
    m.add_function(wrap_pyfunction!(fe_color_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fe_component_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(fe_morphology, m)?)?;
    m.add_function(wrap_pyfunction!(fe_convolve_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fe_turbulence, m)?)?;
    m.add_function(wrap_pyfunction!(fe_displacement_map, m)?)?;
    m.add_function(wrap_pyfunction!(fe_tile, m)?)?;
    m.add_function(wrap_pyfunction!(fe_diffuse_lighting, m)?)?;
    m.add_function(wrap_pyfunction!(fe_specular_lighting, m)?)?;
    m.add_function(wrap_pyfunction!(fe_gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(fe_drop_shadow, m)?)?;
    m.add_function(wrap_pyfunction!(get_source_alpha, m)?)?;
    // Color space conversion
    m.add_function(wrap_pyfunction!(srgb_to_linear, m)?)?;
    m.add_function(wrap_pyfunction!(linear_to_srgb, m)?)?;
    Ok(())
}
