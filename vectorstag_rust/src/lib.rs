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
) -> Bound<'py, PyArray2<u8>> {
    let n = points.len();
    if n < 3 {
        return Array2::<u8>::zeros((height, width)).into_pyarray(py);
    }

    // Compute left and right edge points with miter joins
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
        let alpha = half_d.sin() * ((4.0 + 3.0 * tan_half * tan_half).sqrt() - 1.0) / 3.0;

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

    // Simple box filter (area averaging)
    let scale_x = src_w as f32 / new_width as f32;
    let scale_y = src_h as f32 / new_height as f32;

    for dy in 0..new_height {
        let sy_start = (dy as f32 * scale_y) as usize;
        let sy_end = (((dy + 1) as f32 * scale_y) as usize).min(src_h);

        for dx in 0..new_width {
            let sx_start = (dx as f32 * scale_x) as usize;
            let sx_end = (((dx + 1) as f32 * scale_x) as usize).min(src_w);

            let mut sum = [0u32; 4];
            let mut count = 0u32;

            for sy in sy_start..sy_end {
                for sx in sx_start..sx_end {
                    sum[0] += src_arr[[sy, sx, 0]] as u32;
                    sum[1] += src_arr[[sy, sx, 1]] as u32;
                    sum[2] += src_arr[[sy, sx, 2]] as u32;
                    sum[3] += src_arr[[sy, sx, 3]] as u32;
                    count += 1;
                }
            }

            if count > 0 {
                dst[[dy, dx, 0]] = (sum[0] / count) as u8;
                dst[[dy, dx, 1]] = (sum[1] / count) as u8;
                dst[[dy, dx, 2]] = (sum[2] / count) as u8;
                dst[[dy, dx, 3]] = (sum[3] / count) as u8;
            }
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
    Ok(())
}
