//! Stroke rendering operations for closed polygons

use pyo3::prelude::*;
use numpy::{PyArray2, IntoPyArray};
use ndarray::Array2;

/// Render closed polygon stroke to a mask buffer
/// Computes offset points and fills the stroke region
#[pyfunction]
pub fn render_stroke_closed_polygon<'py>(
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

    let mut quads: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n * 2);

    // 1. Generate Segments (Rectangles)
    for i in 0..n {
        let p1 = points[i];
        let p2 = points[(i + 1) % n];

        let d = normalize(subtract(p2, p1));
        let perp = (-d.1, d.0);

        let p1_l = (p1.0 + perp.0 * half_width, p1.1 + perp.1 * half_width);
        let p1_r = (p1.0 - perp.0 * half_width, p1.1 - perp.1 * half_width);
        let p2_l = (p2.0 + perp.0 * half_width, p2.1 + perp.1 * half_width);
        let p2_r = (p2.0 - perp.0 * half_width, p2.1 - perp.1 * half_width);

        quads.push(vec![p1_l, p2_l, p2_r, p1_r]);
    }

    // 2. Generate Joins at Vertices
    let mut arcs: Vec<(f64, f64, f64, f64, f64)> = Vec::new();

    for i in 0..n {
        let p_prev = points[(i + n - 1) % n];
        let p_curr = points[i];
        let p_next = points[(i + 1) % n];

        let d1 = normalize(subtract(p_curr, p_prev));
        let d2 = normalize(subtract(p_next, p_curr));

        let cross = d1.0 * d2.1 - d1.1 * d2.0;

        if cross.abs() < 0.001 {
            continue;
        }

        let perp1 = (-d1.1, d1.0);
        let perp2 = (-d2.1, d2.0);

        let is_right_turn = cross > 0.0;

        let (outer_p1, outer_p2) = if is_right_turn {
            (
                (p_curr.0 + perp1.0 * half_width, p_curr.1 + perp1.1 * half_width),
                (p_curr.0 + perp2.0 * half_width, p_curr.1 + perp2.1 * half_width)
            )
        } else {
            (
                (p_curr.0 - perp1.0 * half_width, p_curr.1 - perp1.1 * half_width),
                (p_curr.0 - perp2.0 * half_width, p_curr.1 - perp2.1 * half_width)
            )
        };

        if linejoin == "round" {
            let angle1 = if is_right_turn {
                (-perp1.1).atan2(-perp1.0)
            } else {
                perp1.1.atan2(perp1.0)
            };

            let angle2 = if is_right_turn {
                (-perp2.1).atan2(-perp2.0)
            } else {
                perp2.1.atan2(perp2.0)
            };

            let start = angle1;
            let end = angle2;

            let mut sweep = end - start;
            while sweep > std::f64::consts::PI { sweep -= 2.0 * std::f64::consts::PI; }
            while sweep < -std::f64::consts::PI { sweep += 2.0 * std::f64::consts::PI; }

            arcs.push((p_curr.0, p_curr.1, half_width, start, start + sweep));

        } else if linejoin == "bevel" {
            quads.push(vec![p_curr, outer_p1, outer_p2]);
        } else {
            // Miter
            let miter_pt = line_intersection(outer_p1, d1, outer_p2, d2);

            if let Some(mp) = miter_pt {
                let dist = ((mp.0 - p_curr.0).powi(2) + (mp.1 - p_curr.1).powi(2)).sqrt();
                if dist <= miterlimit * half_width {
                    quads.push(vec![p_curr, outer_p1, mp, outer_p2]);
                } else {
                    quads.push(vec![p_curr, outer_p1, outer_p2]);
                }
            } else {
                quads.push(vec![p_curr, outer_p1, outer_p2]);
            }
        }
    }

    // Fill all quads using union
    let mut mask = Array2::<u8>::zeros((height, width));

    for quad in &quads {
        let poly_min_y = quad.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let poly_max_y = quad.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let y_start = ((poly_min_y - min_y as f64).max(0.0) as usize).min(height);
        let y_end = ((poly_max_y - min_y as f64 + 1.0).max(0.0) as usize).min(height);

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

    // Draw arcs for round joins
    for (cx, cy, r, start_angle, end_angle) in arcs {
        let sweep = end_angle - start_angle;
        let n_arc = ((sweep.abs() / (std::f64::consts::PI / 16.0)) as usize).max(8);
        let mut arc_poly: Vec<(f64, f64)> = Vec::with_capacity(n_arc + 2);

        arc_poly.push((cx - min_x as f64, cy - min_y as f64));

        for j in 0..=n_arc {
            let t = j as f64 / n_arc as f64;
            let angle = start_angle + t * sweep;
            let px = cx + r * angle.cos() - min_x as f64;
            let py = cy + r * angle.sin() - min_y as f64;
            arc_poly.push((px, py));
        }

        let poly_min_y = arc_poly.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let poly_max_y = arc_poly.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let y_start = (poly_min_y.max(0.0) as usize).min(height);
        let y_end = ((poly_max_y + 1.0).max(0.0) as usize).min(height);

        for y in y_start..y_end {
            let screen_y = y as f64 + 0.5;
            let mut intersections: Vec<f64> = Vec::new();

            for k in 0..arc_poly.len() {
                let (mut x1, mut y1) = arc_poly[k];
                let (mut x2, mut y2) = arc_poly[(k + 1) % arc_poly.len()];

                if (y1 - y2).abs() < 1e-10 {
                    continue;
                }
                if y1 > y2 {
                    std::mem::swap(&mut x1, &mut x2);
                    std::mem::swap(&mut y1, &mut y2);
                }
                if y1 <= screen_y && screen_y < y2 {
                    let t = (screen_y - y1) / (y2 - y1);
                    intersections.push(x1 + t * (x2 - x1));
                }
            }

            if intersections.len() >= 2 {
                intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for pair in intersections.chunks(2) {
                    if pair.len() == 2 {
                        let x_start = (pair[0].max(0.0) as usize).min(width);
                        let x_end = ((pair[1] as usize) + 1).min(width);
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

/// Register stroke module functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_stroke_closed_polygon, m)?)?;
    Ok(())
}
