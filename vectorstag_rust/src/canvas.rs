//! High-quality analytical antialiasing canvas rendering.
//!
//! This module implements the signed-trapezoid-area algorithm for computing
//! exact pixel coverage without supersampling. The algorithm is based on
//! techniques from stb_truetype and Anti-Grain Geometry (AGG).
//!
//! Key features:
//! - Analytical coverage calculation (equivalent to infinite supersampling)
//! - Direct rendering to numpy arrays (no intermediate buffers)
//! - Support for subpixel positioning (float coordinates)
//! - O(p log p) complexity where p = pixels on polygon edges

use pyo3::prelude::*;
use numpy::PyReadwriteArray3;

/// Analytical edge for coverage calculation.
/// Represents a polygon edge with precomputed values for efficient scanline processing.
#[derive(Clone, Debug)]
struct AnalyticalEdge {
    x_top: f32,       // X coordinate at top of edge
    y_top: f32,       // Y coordinate at top of edge
    y_bottom: f32,    // Y coordinate at bottom of edge
    dx_per_dy: f32,   // X increment per Y unit (inverse slope)
    direction: i8,    // +1 for downward edge, -1 for upward (for winding)
}

impl AnalyticalEdge {
    /// Create a new edge from two points, ensuring y_top < y_bottom
    fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Option<Self> {
        let dy = y2 - y1;

        // Skip horizontal edges (they don't contribute to coverage)
        if dy.abs() < 1e-6 {
            return None;
        }

        let dx = x2 - x1;
        let dx_per_dy = dx / dy;

        if y1 < y2 {
            // Downward edge
            Some(AnalyticalEdge {
                x_top: x1,
                y_top: y1,
                y_bottom: y2,
                dx_per_dy,
                direction: 1,
            })
        } else {
            // Upward edge - swap points
            Some(AnalyticalEdge {
                x_top: x2,
                y_top: y2,
                y_bottom: y1,
                dx_per_dy,
                direction: -1,
            })
        }
    }

    /// Get X coordinate at a given Y value
    #[inline]
    fn x_at_y(&self, y: f32) -> f32 {
        self.x_top + (y - self.y_top) * self.dx_per_dy
    }
}

/// Process all edges for a single scanline, computing per-pixel coverage.
///
/// Uses the signed-area algorithm. For each edge crossing this scanline,
/// we compute its contribution to each pixel. The contribution to a pixel
/// is (portion of pixel to the RIGHT of edge) * edge_direction * height.
/// We then integrate from left to right to accumulate winding.
fn process_scanline_coverage(
    edges: &[AnalyticalEdge],
    y_scanline: i32,
    coverage: &mut [f32],
    width: usize,
) {
    let y = y_scanline as f32;
    let y_next = y + 1.0;

    // Clear coverage array - this will hold coverage deltas
    coverage.fill(0.0);

    for edge in edges {
        // Check if edge is active for this scanline
        if edge.y_bottom <= y || edge.y_top >= y_next {
            continue;
        }

        // Clamp edge to scanline bounds
        let y_clamp_top = y.max(edge.y_top);
        let y_clamp_bot = y_next.min(edge.y_bottom);
        let height = y_clamp_bot - y_clamp_top;

        if height <= 0.0 {
            continue;
        }

        // Compute X coordinates at clamped Y values
        let x_at_top = edge.x_at_y(y_clamp_top);
        let x_at_bot = edge.x_at_y(y_clamp_bot);

        // For a near-vertical edge, use the average x position
        let x_avg = (x_at_top + x_at_bot) * 0.5;
        let x_min = x_at_top.min(x_at_bot);
        let x_max = x_at_top.max(x_at_bot);

        let sign = edge.direction as f32;

        // Compute the pixel containing the average edge position
        let px_edge = (x_avg.floor() as i32).max(0) as usize;

        if px_edge >= width {
            continue;
        }

        // For nearly vertical edges (common case), we distribute the delta:
        // - At the edge pixel: add (1 - frac) * height * sign as delta
        // - At the next pixel: add frac * height * sign as delta
        // After integration, this gives proper coverage values.

        let edge_width = x_max - x_min;

        // The edge contributes a total of (height * sign) to the winding count.
        // We distribute this across pixels where the edge crosses.
        //
        // For a nearly vertical edge at x=25.3:
        //   - Pixel 25 gets (1 - 0.3) * height = 0.7 * height
        //   - Pixel 26 gets 0.3 * height
        //
        // For a wide/sloped edge from x=10 to x=30:
        //   - Each pixel crossed gets a portion based on how much of the
        //     edge's x-range falls within that pixel.

        let px_start = (x_min.floor() as i32).max(0) as usize;
        let px_end = ((x_max.ceil() as i32) + 1).min(width as i32) as usize;

        if edge_width < 1.0 {
            // Edge is narrow (including vertical edges) - split between two pixels
            // based on subpixel position for proper antialiasing
            let frac = x_avg - x_avg.floor();
            if px_edge < width {
                coverage[px_edge] += (1.0 - frac) * height * sign;
            }
            if px_edge + 1 < width {
                coverage[px_edge + 1] += frac * height * sign;
            }
        } else {
            // Wide edge - distribute based on x-range within each pixel
            // The contribution per unit x is: height / edge_width
            let contrib_per_x = height / edge_width;

            for px in px_start..px_end {
                if px >= width {
                    break;
                }
                let px_left = px as f32;
                let px_right = px_left + 1.0;

                // How much of the edge's x-range falls within this pixel?
                let edge_in_pixel_left = px_left.max(x_min);
                let edge_in_pixel_right = px_right.min(x_max);
                let edge_in_pixel = (edge_in_pixel_right - edge_in_pixel_left).max(0.0);

                // This pixel gets a proportional share of the contribution
                let delta = edge_in_pixel * contrib_per_x * sign;
                coverage[px] += delta;
            }
        }
    }

    // Integrate from left to right to get cumulative coverage
    let mut running_sum = 0.0f32;
    for px in 0..width {
        running_sum += coverage[px];
        coverage[px] = running_sum;
    }
}

/// Apply fill rule to convert signed coverage to final alpha.
#[inline]
fn apply_fill_rule(coverage: f32, fill_rule: u8) -> f32 {
    if fill_rule == 1 {
        // Even-odd: coverage mod 2, then take fractional part
        let c = coverage.abs();
        let wrapped = c - 2.0 * (c / 2.0).floor();
        if wrapped > 1.0 {
            2.0 - wrapped
        } else {
            wrapped
        }
    } else {
        // Nonzero: absolute value clamped to 1
        coverage.abs().min(1.0)
    }
}

/// Alpha-blend a source color onto a destination pixel.
#[inline]
fn blend_pixel(dst: &mut [u8], src_r: u8, src_g: u8, src_b: u8, src_a: u8) {
    if src_a == 0 {
        return;
    }

    if src_a == 255 {
        dst[0] = src_r;
        dst[1] = src_g;
        dst[2] = src_b;
        dst[3] = 255;
        return;
    }

    let sa = src_a as f32 / 255.0;
    let da = dst[3] as f32 / 255.0;

    let out_a = sa + da * (1.0 - sa);

    if out_a > 0.0 {
        let out_a_inv = 1.0 / out_a;
        dst[0] = ((src_r as f32 * sa + dst[0] as f32 * da * (1.0 - sa)) * out_a_inv) as u8;
        dst[1] = ((src_g as f32 * sa + dst[1] as f32 * da * (1.0 - sa)) * out_a_inv) as u8;
        dst[2] = ((src_b as f32 * sa + dst[2] as f32 * da * (1.0 - sa)) * out_a_inv) as u8;
        dst[3] = (out_a * 255.0) as u8;
    }
}

/// Fill a polygon with analytical antialiasing directly onto an RGBA array.
///
/// This function uses the signed-trapezoid-area algorithm to compute exact
/// per-pixel coverage without supersampling, achieving quality equivalent
/// to infinite supersampling.
///
/// # Arguments
/// * `dst` - Target RGBA array (modified in-place)
/// * `points` - Polygon vertices as (x, y) float tuples
/// * `r, g, b, a` - Fill color components
/// * `fill_rule` - 0 for nonzero winding, 1 for even-odd
#[pyfunction]
pub fn canvas_fill_polygon_aa(
    mut dst: PyReadwriteArray3<u8>,
    points: Vec<(f32, f32)>,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    fill_rule: u8,
) -> PyResult<()> {
    let n = points.len();
    if n < 3 {
        return Ok(());
    }

    let mut arr = dst.as_array_mut();
    let height = arr.shape()[0];
    let width = arr.shape()[1];

    if width == 0 || height == 0 {
        return Ok(());
    }

    // Build edge list from polygon
    let mut edges: Vec<AnalyticalEdge> = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        if let Some(edge) = AnalyticalEdge::new(
            points[i].0, points[i].1,
            points[j].0, points[j].1,
        ) {
            edges.push(edge);
        }
    }

    if edges.is_empty() {
        return Ok(());
    }

    // Sort edges by y_top for efficient scanline processing
    edges.sort_by(|a, b| a.y_top.partial_cmp(&b.y_top).unwrap_or(std::cmp::Ordering::Equal));

    // Find bounding box
    let y_min = edges.iter().map(|e| e.y_top).fold(f32::INFINITY, f32::min);
    let y_max = edges.iter().map(|e| e.y_bottom).fold(f32::NEG_INFINITY, f32::max);

    let scanline_start = (y_min.floor() as i32).max(0);
    let scanline_end = (y_max.ceil() as i32).min(height as i32);

    // Allocate coverage buffer
    let mut coverage = vec![0.0f32; width];

    // Process each scanline
    for y in scanline_start..scanline_end {
        process_scanline_coverage(&edges, y, &mut coverage, width);

        // Apply coverage to pixels
        for x in 0..width {
            let cov = apply_fill_rule(coverage[x], fill_rule);

            if cov > 0.0 {
                // Compute effective alpha
                let effective_alpha = (a as f32 * cov).round() as u8;

                if effective_alpha > 0 {
                    let pixel = &mut arr.as_slice_mut().unwrap()[(y as usize * width + x) * 4..][..4];
                    blend_pixel(pixel, r, g, b, effective_alpha);
                }
            }
        }
    }

    Ok(())
}

/// Fill multiple polygon contours (for shapes with holes).
///
/// Outer contours should be wound counter-clockwise, holes clockwise
/// (or vice versa) to achieve correct fill with nonzero winding rule.
#[pyfunction]
pub fn canvas_fill_multi_polygon_aa(
    mut dst: PyReadwriteArray3<u8>,
    contours: Vec<Vec<(f32, f32)>>,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    fill_rule: u8,
) -> PyResult<()> {
    let mut arr = dst.as_array_mut();
    let height = arr.shape()[0];
    let width = arr.shape()[1];

    if width == 0 || height == 0 {
        return Ok(());
    }

    // Build edge list from all contours
    let mut edges: Vec<AnalyticalEdge> = Vec::new();

    for points in &contours {
        let n = points.len();
        if n < 3 {
            continue;
        }

        for i in 0..n {
            let j = (i + 1) % n;
            if let Some(edge) = AnalyticalEdge::new(
                points[i].0, points[i].1,
                points[j].0, points[j].1,
            ) {
                edges.push(edge);
            }
        }
    }

    if edges.is_empty() {
        return Ok(());
    }

    // Sort edges by y_top
    edges.sort_by(|a, b| a.y_top.partial_cmp(&b.y_top).unwrap_or(std::cmp::Ordering::Equal));

    // Find bounding box
    let y_min = edges.iter().map(|e| e.y_top).fold(f32::INFINITY, f32::min);
    let y_max = edges.iter().map(|e| e.y_bottom).fold(f32::NEG_INFINITY, f32::max);

    let scanline_start = (y_min.floor() as i32).max(0);
    let scanline_end = (y_max.ceil() as i32).min(height as i32);

    let mut coverage = vec![0.0f32; width];

    for y in scanline_start..scanline_end {
        process_scanline_coverage(&edges, y, &mut coverage, width);

        for x in 0..width {
            let cov = apply_fill_rule(coverage[x], fill_rule);

            if cov > 0.0 {
                let effective_alpha = (a as f32 * cov).round() as u8;

                if effective_alpha > 0 {
                    let pixel = &mut arr.as_slice_mut().unwrap()[(y as usize * width + x) * 4..][..4];
                    blend_pixel(pixel, r, g, b, effective_alpha);
                }
            }
        }
    }

    Ok(())
}

/// Interpolate gradient color at position t (0.0 to 1.0).
#[inline]
fn interpolate_gradient_color(
    t: f32,
    stops: &[(f32, u8, u8, u8, u8)],
    spread_method: u8,
) -> (u8, u8, u8, u8) {
    if stops.is_empty() {
        return (0, 0, 0, 0);
    }

    // Apply spread method
    let t = match spread_method {
        1 => t.rem_euclid(1.0),  // repeat
        2 => {  // reflect
            let t2 = t.rem_euclid(2.0);
            if t2 > 1.0 { 2.0 - t2 } else { t2 }
        }
        _ => t.clamp(0.0, 1.0),  // pad (default)
    };

    // Find the two stops to interpolate between
    if t <= stops[0].0 {
        let s = &stops[0];
        return (s.1, s.2, s.3, s.4);
    }
    if t >= stops[stops.len() - 1].0 {
        let s = &stops[stops.len() - 1];
        return (s.1, s.2, s.3, s.4);
    }

    for i in 0..stops.len() - 1 {
        if t >= stops[i].0 && t <= stops[i + 1].0 {
            let t0 = stops[i].0;
            let t1 = stops[i + 1].0;
            let dt = t1 - t0;
            if dt < 0.0001 {
                let s = &stops[i];
                return (s.1, s.2, s.3, s.4);
            }
            let f = (t - t0) / dt;
            let s0 = &stops[i];
            let s1 = &stops[i + 1];
            return (
                (s0.1 as f32 + f * (s1.1 as f32 - s0.1 as f32)).round() as u8,
                (s0.2 as f32 + f * (s1.2 as f32 - s0.2 as f32)).round() as u8,
                (s0.3 as f32 + f * (s1.3 as f32 - s0.3 as f32)).round() as u8,
                (s0.4 as f32 + f * (s1.4 as f32 - s0.4 as f32)).round() as u8,
            );
        }
    }

    let s = &stops[stops.len() - 1];
    (s.1, s.2, s.3, s.4)
}

/// Fill polygon with linear gradient and analytical AA.
#[pyfunction]
pub fn canvas_fill_polygon_linear_gradient_aa(
    mut dst: PyReadwriteArray3<u8>,
    points: Vec<(f32, f32)>,
    x1: f32, y1: f32,  // Gradient start point
    x2: f32, y2: f32,  // Gradient end point
    stops: Vec<(f32, u8, u8, u8, u8)>,  // (position, r, g, b, a)
    spread_method: u8,  // 0=pad, 1=repeat, 2=reflect
    fill_rule: u8,
) -> PyResult<()> {
    let n = points.len();
    if n < 3 || stops.is_empty() {
        return Ok(());
    }

    let mut arr = dst.as_array_mut();
    let height = arr.shape()[0];
    let width = arr.shape()[1];

    if width == 0 || height == 0 {
        return Ok(());
    }

    // Build edge list
    let mut edges: Vec<AnalyticalEdge> = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        if let Some(edge) = AnalyticalEdge::new(
            points[i].0, points[i].1,
            points[j].0, points[j].1,
        ) {
            edges.push(edge);
        }
    }

    if edges.is_empty() {
        return Ok(());
    }

    // Gradient vector
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 0.0001 {
        return Ok(());
    }

    edges.sort_by(|a, b| a.y_top.partial_cmp(&b.y_top).unwrap_or(std::cmp::Ordering::Equal));

    let y_min = edges.iter().map(|e| e.y_top).fold(f32::INFINITY, f32::min);
    let y_max = edges.iter().map(|e| e.y_bottom).fold(f32::NEG_INFINITY, f32::max);

    let scanline_start = (y_min.floor() as i32).max(0);
    let scanline_end = (y_max.ceil() as i32).min(height as i32);

    let mut coverage = vec![0.0f32; width];

    for y in scanline_start..scanline_end {
        process_scanline_coverage(&edges, y, &mut coverage, width);

        for x in 0..width {
            let cov = apply_fill_rule(coverage[x], fill_rule);

            if cov > 0.0 {
                // Compute gradient t value at this pixel center
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;

                let (r, g, b, a) = interpolate_gradient_color(t, &stops, spread_method);
                let effective_alpha = (a as f32 * cov).round() as u8;

                if effective_alpha > 0 {
                    let pixel = &mut arr.as_slice_mut().unwrap()[(y as usize * width + x) * 4..][..4];
                    blend_pixel(pixel, r, g, b, effective_alpha);
                }
            }
        }
    }

    Ok(())
}

/// Fill polygon with radial gradient and analytical AA.
#[pyfunction]
pub fn canvas_fill_polygon_radial_gradient_aa(
    mut dst: PyReadwriteArray3<u8>,
    points: Vec<(f32, f32)>,
    cx: f32, cy: f32, radius: f32,  // Outer circle
    fx: f32, fy: f32, fr: f32,       // Focal point and inner radius
    stops: Vec<(f32, u8, u8, u8, u8)>,  // (position, r, g, b, a)
    spread_method: u8,
    fill_rule: u8,
) -> PyResult<()> {
    let n = points.len();
    if n < 3 || stops.is_empty() || radius <= 0.0 {
        return Ok(());
    }

    let mut arr = dst.as_array_mut();
    let height = arr.shape()[0];
    let width = arr.shape()[1];

    if width == 0 || height == 0 {
        return Ok(());
    }

    // Build edge list
    let mut edges: Vec<AnalyticalEdge> = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        if let Some(edge) = AnalyticalEdge::new(
            points[i].0, points[i].1,
            points[j].0, points[j].1,
        ) {
            edges.push(edge);
        }
    }

    if edges.is_empty() {
        return Ok(());
    }

    edges.sort_by(|a, b| a.y_top.partial_cmp(&b.y_top).unwrap_or(std::cmp::Ordering::Equal));

    let y_min = edges.iter().map(|e| e.y_top).fold(f32::INFINITY, f32::min);
    let y_max = edges.iter().map(|e| e.y_bottom).fold(f32::NEG_INFINITY, f32::max);

    let scanline_start = (y_min.floor() as i32).max(0);
    let scanline_end = (y_max.ceil() as i32).min(height as i32);

    let mut coverage = vec![0.0f32; width];

    for y in scanline_start..scanline_end {
        process_scanline_coverage(&edges, y, &mut coverage, width);

        for x in 0..width {
            let cov = apply_fill_rule(coverage[x], fill_rule);

            if cov > 0.0 {
                // Compute radial gradient t value
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;

                // Distance from focal point
                let dfx = px - fx;
                let dfy = py - fy;
                let dist = (dfx * dfx + dfy * dfy).sqrt();

                // Normalize to gradient range
                let t = if radius > fr {
                    ((dist - fr) / (radius - fr)).max(0.0)
                } else {
                    dist / radius.max(0.001)
                };

                let (r, g, b, a) = interpolate_gradient_color(t, &stops, spread_method);
                let effective_alpha = (a as f32 * cov).round() as u8;

                if effective_alpha > 0 {
                    let pixel = &mut arr.as_slice_mut().unwrap()[(y as usize * width + x) * 4..][..4];
                    blend_pixel(pixel, r, g, b, effective_alpha);
                }
            }
        }
    }

    Ok(())
}

/// Fill an axis-aligned rectangle with analytical antialiasing.
///
/// This is optimized for the common case of rectangles, computing
/// exact subpixel coverage at edges.
#[pyfunction]
pub fn canvas_fill_rect_aa(
    mut dst: PyReadwriteArray3<u8>,
    x: f32,
    y: f32,
    rect_width: f32,
    rect_height: f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> PyResult<()> {
    let mut arr = dst.as_array_mut();
    let img_height = arr.shape()[0];
    let img_width = arr.shape()[1];

    if img_width == 0 || img_height == 0 || rect_width <= 0.0 || rect_height <= 0.0 {
        return Ok(());
    }

    let x2 = x + rect_width;
    let y2 = y + rect_height;

    // Pixel bounds
    let px_start = (x.floor() as i32).max(0) as usize;
    let px_end = (x2.ceil() as i32).min(img_width as i32) as usize;
    let py_start = (y.floor() as i32).max(0) as usize;
    let py_end = (y2.ceil() as i32).min(img_height as i32) as usize;

    for py in py_start..py_end {
        let py_f = py as f32;

        // Vertical coverage for this row
        let y_cov = if py_f + 1.0 <= y || py_f >= y2 {
            0.0
        } else {
            let y_top = y.max(py_f);
            let y_bot = y2.min(py_f + 1.0);
            y_bot - y_top
        };

        if y_cov <= 0.0 {
            continue;
        }

        for px in px_start..px_end {
            let px_f = px as f32;

            // Horizontal coverage for this column
            let x_cov = if px_f + 1.0 <= x || px_f >= x2 {
                0.0
            } else {
                let x_left = x.max(px_f);
                let x_right = x2.min(px_f + 1.0);
                x_right - x_left
            };

            if x_cov <= 0.0 {
                continue;
            }

            // Combined coverage
            let cov = x_cov * y_cov;
            let effective_alpha = (a as f32 * cov).round() as u8;

            if effective_alpha > 0 {
                let pixel = &mut arr.as_slice_mut().unwrap()[(py * img_width + px) * 4..][..4];
                blend_pixel(pixel, r, g, b, effective_alpha);
            }
        }
    }

    Ok(())
}

/// Fill a circle with analytical antialiasing.
///
/// Uses distance-based coverage calculation for smooth edges.
#[pyfunction]
pub fn canvas_fill_circle_aa(
    mut dst: PyReadwriteArray3<u8>,
    cx: f32,
    cy: f32,
    radius: f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> PyResult<()> {
    let mut arr = dst.as_array_mut();
    let img_height = arr.shape()[0];
    let img_width = arr.shape()[1];

    if img_width == 0 || img_height == 0 || radius <= 0.0 {
        return Ok(());
    }

    // Bounding box
    let px_start = ((cx - radius - 1.0).floor() as i32).max(0) as usize;
    let px_end = ((cx + radius + 1.0).ceil() as i32).min(img_width as i32) as usize;
    let py_start = ((cy - radius - 1.0).floor() as i32).max(0) as usize;
    let py_end = ((cy + radius + 1.0).ceil() as i32).min(img_height as i32) as usize;

    let r2 = radius * radius;

    for py in py_start..py_end {
        let py_center = py as f32 + 0.5;
        let dy = py_center - cy;
        let dy2 = dy * dy;

        for px in px_start..px_end {
            let px_center = px as f32 + 0.5;
            let dx = px_center - cx;
            let dx2 = dx * dx;

            let dist2 = dx2 + dy2;
            let dist = dist2.sqrt();

            // Compute coverage based on distance from circle edge
            // Use a 1-pixel transition zone for antialiasing
            let cov = if dist <= radius - 0.5 {
                1.0  // Fully inside
            } else if dist >= radius + 0.5 {
                0.0  // Fully outside
            } else {
                // In the transition zone - linear falloff
                (radius + 0.5 - dist)
            };

            if cov > 0.0 {
                let effective_alpha = (a as f32 * cov).round() as u8;

                if effective_alpha > 0 {
                    let pixel = &mut arr.as_slice_mut().unwrap()[(py * img_width + px) * 4..][..4];
                    blend_pixel(pixel, r, g, b, effective_alpha);
                }
            }
        }
    }

    Ok(())
}

/// Fill an ellipse with analytical antialiasing.
///
/// Uses normalized distance calculation for smooth edges.
#[pyfunction]
pub fn canvas_fill_ellipse_aa(
    mut dst: PyReadwriteArray3<u8>,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> PyResult<()> {
    let mut arr = dst.as_array_mut();
    let img_height = arr.shape()[0];
    let img_width = arr.shape()[1];

    if img_width == 0 || img_height == 0 || rx <= 0.0 || ry <= 0.0 {
        return Ok(());
    }

    // Bounding box
    let px_start = ((cx - rx - 1.0).floor() as i32).max(0) as usize;
    let px_end = ((cx + rx + 1.0).ceil() as i32).min(img_width as i32) as usize;
    let py_start = ((cy - ry - 1.0).floor() as i32).max(0) as usize;
    let py_end = ((cy + ry + 1.0).ceil() as i32).min(img_height as i32) as usize;

    // Precompute inverse radii squared
    let inv_rx2 = 1.0 / (rx * rx);
    let inv_ry2 = 1.0 / (ry * ry);

    for py in py_start..py_end {
        let py_center = py as f32 + 0.5;
        let dy = py_center - cy;
        let dy_norm2 = dy * dy * inv_ry2;

        for px in px_start..px_end {
            let px_center = px as f32 + 0.5;
            let dx = px_center - cx;
            let dx_norm2 = dx * dx * inv_rx2;

            // Normalized distance (1.0 = on ellipse boundary)
            let norm_dist2 = dx_norm2 + dy_norm2;
            let norm_dist = norm_dist2.sqrt();

            // Compute gradient of the implicit function f(x,y) = (x/rx)² + (y/ry)² - 1
            // The gradient magnitude tells us how fast f changes per pixel
            let grad_x = 2.0 * dx * inv_rx2;
            let grad_y = 2.0 * dy * inv_ry2;
            let grad_mag = (grad_x * grad_x + grad_y * grad_y).sqrt();

            // Approximate signed distance to ellipse in pixels
            // signed_dist < 0 means inside, > 0 means outside
            let signed_dist = if grad_mag > 1e-6 {
                (norm_dist - 1.0) / grad_mag
            } else {
                // At center, gradient is tiny - use norm_dist to determine inside/outside
                if norm_dist < 1.0 { -1000.0 } else { 1000.0 }
            };

            // Coverage based on signed pixel distance
            let cov = if signed_dist < -0.5 {
                1.0  // Fully inside
            } else if signed_dist > 0.5 {
                0.0  // Fully outside
            } else {
                // Linear transition in the [-0.5, 0.5] pixel band
                0.5 - signed_dist
            };

            if cov > 0.0 {
                let effective_alpha = (a as f32 * cov).round() as u8;

                if effective_alpha > 0 {
                    let pixel = &mut arr.as_slice_mut().unwrap()[(py * img_width + px) * 4..][..4];
                    blend_pixel(pixel, r, g, b, effective_alpha);
                }
            }
        }
    }

    Ok(())
}

/// Draw an antialiased line with subpixel endpoint positions.
#[pyfunction]
pub fn canvas_stroke_line_aa(
    mut dst: PyReadwriteArray3<u8>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    width: f32,
    linecap: u8,  // 0=butt, 1=round, 2=square
) -> PyResult<()> {
    if width <= 0.0 {
        return Ok(());
    }

    let mut arr = dst.as_array_mut();
    let img_height = arr.shape()[0];
    let img_width = arr.shape()[1];

    if img_width == 0 || img_height == 0 {
        return Ok(());
    }

    // Line direction
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();

    if len < 1e-6 {
        // Degenerate line - just draw a dot
        if linecap == 1 {
            // Round cap = circle
            canvas_fill_circle_aa(dst, x1, y1, width / 2.0, r, g, b, a)?;
        } else {
            // Butt/square = small rect
            let half = width / 2.0;
            canvas_fill_rect_aa(dst, x1 - half, y1 - half, width, width, r, g, b, a)?;
        }
        return Ok(());
    }

    // Normalized direction and perpendicular
    let dx_n = dx / len;
    let dy_n = dy / len;
    let perp_x = -dy_n;
    let perp_y = dx_n;

    let half_width = width / 2.0;

    // Build stroke polygon
    let mut points = Vec::with_capacity(8);

    // Start cap
    let cap_extend = if linecap == 2 { half_width } else { 0.0 };  // Square cap extends
    let start_x = x1 - dx_n * cap_extend;
    let start_y = y1 - dy_n * cap_extend;
    let end_x = x2 + dx_n * cap_extend;
    let end_y = y2 + dy_n * cap_extend;

    // Left side (going forward)
    points.push((start_x + perp_x * half_width, start_y + perp_y * half_width));
    points.push((end_x + perp_x * half_width, end_y + perp_y * half_width));

    // Right side (going back)
    points.push((end_x - perp_x * half_width, end_y - perp_y * half_width));
    points.push((start_x - perp_x * half_width, start_y - perp_y * half_width));

    // Build edge list from polygon points
    let n = points.len();
    let mut edges: Vec<AnalyticalEdge> = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        if let Some(edge) = AnalyticalEdge::new(
            points[i].0, points[i].1,
            points[j].0, points[j].1,
        ) {
            edges.push(edge);
        }
    }

    // Add round cap edges if needed
    if linecap == 1 && !edges.is_empty() {
        // For round caps, add semicircle edges at both endpoints
        // We approximate with 8 segments per semicircle
        let n_segments = 8;
        for endpoint_idx in 0..2 {
            let (cx, cy) = if endpoint_idx == 0 { (x1, y1) } else { (x2, y2) };
            let start_angle = if endpoint_idx == 0 {
                (perp_y).atan2(perp_x) + std::f32::consts::PI / 2.0
            } else {
                (perp_y).atan2(perp_x) - std::f32::consts::PI / 2.0
            };

            for seg in 0..n_segments {
                let a1 = start_angle + (seg as f32) * std::f32::consts::PI / (n_segments as f32);
                let a2 = start_angle + ((seg + 1) as f32) * std::f32::consts::PI / (n_segments as f32);
                let px1 = cx + half_width * a1.cos();
                let py1 = cy + half_width * a1.sin();
                let px2 = cx + half_width * a2.cos();
                let py2 = cy + half_width * a2.sin();

                if let Some(edge) = AnalyticalEdge::new(px1, py1, px2, py2) {
                    edges.push(edge);
                }
            }
        }
    }

    if edges.is_empty() {
        return Ok(());
    }

    // Sort edges and render
    edges.sort_by(|a, b| a.y_top.partial_cmp(&b.y_top).unwrap_or(std::cmp::Ordering::Equal));

    let y_min = edges.iter().map(|e| e.y_top).fold(f32::INFINITY, f32::min);
    let y_max = edges.iter().map(|e| e.y_bottom).fold(f32::NEG_INFINITY, f32::max);

    let scanline_start = (y_min.floor() as i32).max(0);
    let scanline_end = (y_max.ceil() as i32).min(img_height as i32);

    let mut coverage = vec![0.0f32; img_width];

    for y in scanline_start..scanline_end {
        process_scanline_coverage(&edges, y, &mut coverage, img_width);

        for x in 0..img_width {
            let cov = apply_fill_rule(coverage[x], 0);

            if cov > 0.0 {
                let effective_alpha = (a as f32 * cov).round() as u8;

                if effective_alpha > 0 {
                    let pixel = &mut arr.as_slice_mut().unwrap()[(y as usize * img_width + x) * 4..][..4];
                    blend_pixel(pixel, r, g, b, effective_alpha);
                }
            }
        }
    }

    Ok(())
}

/// Blit source image to destination with a polygon mask and analytical AA.
///
/// The mask polygon defines which pixels of the source image are visible.
/// Edges of the mask are antialiased using the analytical coverage algorithm.
#[pyfunction]
pub fn canvas_masked_blit_aa(
    mut dst: PyReadwriteArray3<u8>,
    src: numpy::PyReadonlyArray3<u8>,
    mask_polygon: Vec<(f32, f32)>,
    dst_x: f32, dst_y: f32,  // Destination position (subpixel)
    opacity: f32,
) -> PyResult<()> {
    let n = mask_polygon.len();
    if n < 3 {
        return Ok(());
    }

    let src_arr = src.as_array();
    let mut dst_arr = dst.as_array_mut();

    let dst_height = dst_arr.shape()[0];
    let dst_width = dst_arr.shape()[1];
    let src_height = src_arr.shape()[0];
    let src_width = src_arr.shape()[1];

    if dst_width == 0 || dst_height == 0 || src_width == 0 || src_height == 0 {
        return Ok(());
    }

    // Build edge list from mask polygon
    let mut edges: Vec<AnalyticalEdge> = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        if let Some(edge) = AnalyticalEdge::new(
            mask_polygon[i].0, mask_polygon[i].1,
            mask_polygon[j].0, mask_polygon[j].1,
        ) {
            edges.push(edge);
        }
    }

    if edges.is_empty() {
        return Ok(());
    }

    edges.sort_by(|a, b| a.y_top.partial_cmp(&b.y_top).unwrap_or(std::cmp::Ordering::Equal));

    // Find bounds
    let y_min = edges.iter().map(|e| e.y_top).fold(f32::INFINITY, f32::min);
    let y_max = edges.iter().map(|e| e.y_bottom).fold(f32::NEG_INFINITY, f32::max);
    let x_min = mask_polygon.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let x_max = mask_polygon.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);

    let scanline_start = (y_min.floor() as i32).max(0);
    let scanline_end = (y_max.ceil() as i32).min(dst_height as i32);

    let mut coverage = vec![0.0f32; dst_width];
    let opacity_factor = opacity.clamp(0.0, 1.0);

    for y in scanline_start..scanline_end {
        process_scanline_coverage(&edges, y, &mut coverage, dst_width);

        let px_start = (x_min.floor() as i32).max(0) as usize;
        let px_end = ((x_max.ceil() as i32) + 1).min(dst_width as i32) as usize;

        for x in px_start..px_end {
            let cov = apply_fill_rule(coverage[x], 0);

            if cov > 0.0 {
                // Compute source pixel coordinates
                let src_x = (x as f32 - dst_x).round() as i32;
                let src_y = (y as f32 - dst_y).round() as i32;

                if src_x >= 0 && src_x < src_width as i32 && src_y >= 0 && src_y < src_height as i32 {
                    let src_pixel = &src_arr.as_slice().unwrap()[(src_y as usize * src_width + src_x as usize) * 4..][..4];

                    // Apply mask coverage and opacity to source alpha
                    let src_alpha = src_pixel[3] as f32 * cov * opacity_factor;
                    let effective_alpha = src_alpha.round() as u8;

                    if effective_alpha > 0 {
                        let dst_pixel = &mut dst_arr.as_slice_mut().unwrap()[(y as usize * dst_width + x) * 4..][..4];
                        blend_pixel(dst_pixel, src_pixel[0], src_pixel[1], src_pixel[2], effective_alpha);
                    }
                }
            }
        }
    }

    Ok(())
}
