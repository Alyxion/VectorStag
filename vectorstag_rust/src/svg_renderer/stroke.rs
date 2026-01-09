//! Stroke rendering for SVG shapes.

use super::types::*;

/// Check if all points in a path are effectively at the same location (zero-length path)
pub fn is_zero_length_path(points: &[(f64, f64)]) -> bool {
    if points.is_empty() {
        return true;
    }
    let (x0, y0) = points[0];
    const EPSILON: f64 = 0.001;
    points.iter().all(|(x, y)| (x - x0).abs() < EPSILON && (y - y0).abs() < EPSILON)
}

/// Render a stroke along a set of points
pub fn render_stroke(
    ctx: &mut RenderContext,
    points: &[(f64, f64)],
    color: Color,
    width: f64,
    linecap: LineCap,
    linejoin: LineJoin,
    closed: bool,
    miter_limit: f64,
) {
    if color.a == 0 {
        return;
    }

    let half_width = width / 2.0;

    // Handle zero-length paths (single point or all points identical)
    if points.len() == 1 || (points.len() >= 2 && is_zero_length_path(points)) {
        let (cx, cy) = points[0];
        match linecap {
            LineCap::Round => {
                draw_circle(ctx, cx, cy, half_width, color);
            }
            LineCap::Square => {
                let square = vec![
                    (cx - half_width, cy - half_width),
                    (cx + half_width, cy - half_width),
                    (cx + half_width, cy + half_width),
                    (cx - half_width, cy + half_width),
                ];
                ctx.fill_polygon(&square, color, FillRule::NonZero);
            }
            LineCap::Butt => {
                // Butt caps on zero-length paths render nothing
            }
        }
        return;
    }

    if points.len() < 2 {
        return;
    }

    // Draw stroke as thick line segments
    let n = if closed { points.len() } else { points.len() - 1 };
    for i in 0..n {
        let j = (i + 1) % points.len();
        let (x1, y1) = points[i];
        let (x2, y2) = points[j];

        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 0.001 {
            continue;
        }

        let perp_x = -dy / len * half_width;
        let perp_y = dx / len * half_width;

        let quad = vec![
            (x1 + perp_x, y1 + perp_y),
            (x2 + perp_x, y2 + perp_y),
            (x2 - perp_x, y2 - perp_y),
            (x1 - perp_x, y1 - perp_y),
        ];

        ctx.fill_polygon(&quad, color, FillRule::NonZero);
    }

    // Draw linejoins at internal vertices
    if points.len() > 2 {
        let start = if closed { 0 } else { 1 };
        let end = if closed { points.len() } else { points.len() - 1 };
        for i in start..end {
            let prev_idx = if i == 0 { points.len() - 1 } else { i - 1 };
            let next_idx = (i + 1) % points.len();

            let (cx, cy) = points[i];
            let (px, py) = points[prev_idx];
            let (nx, ny) = points[next_idx];

            let d1x = cx - px;
            let d1y = cy - py;
            let d2x = nx - cx;
            let d2y = ny - cy;

            let len1 = (d1x * d1x + d1y * d1y).sqrt();
            let len2 = (d2x * d2x + d2y * d2y).sqrt();

            if len1 < 0.001 || len2 < 0.001 {
                continue;
            }

            // Perpendicular directions (normalized)
            let perp1_x = -d1y / len1;
            let perp1_y = d1x / len1;
            let perp2_x = -d2y / len2;
            let perp2_y = d2x / len2;

            // Cross product determines turn direction
            // Positive = left turn (convex on left side)
            // Negative = right turn (convex on right side)
            let cross = d1x * d2y - d1y * d2x;

            // Edge points at this vertex for both segments
            let p1_plus = (cx + perp1_x * half_width, cy + perp1_y * half_width);
            let p1_minus = (cx - perp1_x * half_width, cy - perp1_y * half_width);
            let p2_plus = (cx + perp2_x * half_width, cy + perp2_y * half_width);
            let p2_minus = (cx - perp2_x * half_width, cy - perp2_y * half_width);

            // Normalize direction vectors
            let d1_norm = (d1x / len1, d1y / len1);
            let d2_norm = (d2x / len2, d2y / len2);

            // perp = (-d.y, d.x) is 90° CCW rotation, so +perp is LEFT of path, -perp is RIGHT
            // cross > 0 means left turn (counter-clockwise), outer/convex is on the RIGHT (-perp side)
            // cross < 0 means right turn (clockwise), outer/convex is on the LEFT (+perp side)
            let is_left_turn = cross > 0.0;

            let (outer_p1, outer_p2) = if is_left_turn {
                (p1_minus, p2_minus)  // Right side = -perp
            } else {
                (p1_plus, p2_plus)    // Left side = +perp
            };

            let (inner_p1, inner_p2) = if is_left_turn {
                (p1_plus, p2_plus)    // Left side = +perp
            } else {
                (p1_minus, p2_minus)  // Right side = -perp
            };

            // Always fill the inner (concave) gap
            let inner_bevel = vec![(cx, cy), inner_p1, inner_p2];
            ctx.fill_polygon(&inner_bevel, color, FillRule::NonZero);

            match linejoin {
                LineJoin::Round => {
                    draw_circle(ctx, cx, cy, half_width, color);
                }
                LineJoin::Miter => {
                    if cross.abs() > 0.001 {
                        // Find miter point using line intersection
                        // Extend outer_p1 forward along d1, outer_p2 backward along -d2
                        let neg_d2 = (-d2_norm.0, -d2_norm.1);
                        let miter_pt = line_intersection(outer_p1, d1_norm, outer_p2, neg_d2);

                        if let Some(mp) = miter_pt {
                            let dist = ((mp.0 - cx).powi(2) + (mp.1 - cy).powi(2)).sqrt();
                            if dist <= miter_limit * half_width {
                                // Draw miter quad
                                let miter_quad = vec![(cx, cy), outer_p1, mp, outer_p2];
                                ctx.fill_polygon(&miter_quad, color, FillRule::NonZero);
                            } else {
                                // Bevel fallback
                                let bevel = vec![(cx, cy), outer_p1, outer_p2];
                                ctx.fill_polygon(&bevel, color, FillRule::NonZero);
                            }
                        } else {
                            let bevel = vec![(cx, cy), outer_p1, outer_p2];
                            ctx.fill_polygon(&bevel, color, FillRule::NonZero);
                        }
                    } else {
                        // Nearly parallel - just use bevel
                        let bevel = vec![(cx, cy), outer_p1, outer_p2];
                        ctx.fill_polygon(&bevel, color, FillRule::NonZero);
                    }
                }
                LineJoin::Bevel => {
                    let bevel = vec![(cx, cy), outer_p1, outer_p2];
                    ctx.fill_polygon(&bevel, color, FillRule::NonZero);
                }
            }
        }
    }

    // Draw linecaps on open paths
    if !closed {
        match linecap {
            LineCap::Round => {
                let (x1, y1) = points[0];
                let (x2, y2) = points[points.len() - 1];
                draw_circle(ctx, x1, y1, half_width, color);
                draw_circle(ctx, x2, y2, half_width, color);
            }
            LineCap::Square => {
                if points.len() >= 2 {
                    // Start cap
                    let (x1, y1) = points[0];
                    let (x2, y2) = points[1];
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len > 0.001 {
                        let ext_x = -dx / len * half_width;
                        let ext_y = -dy / len * half_width;
                        let perp_x = -dy / len * half_width;
                        let perp_y = dx / len * half_width;
                        let cap = vec![
                            (x1 + perp_x, y1 + perp_y),
                            (x1 - perp_x, y1 - perp_y),
                            (x1 + ext_x - perp_x, y1 + ext_y - perp_y),
                            (x1 + ext_x + perp_x, y1 + ext_y + perp_y),
                        ];
                        ctx.fill_polygon(&cap, color, FillRule::NonZero);
                    }
                    // End cap
                    let (x1, y1) = points[points.len() - 2];
                    let (x2, y2) = points[points.len() - 1];
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len > 0.001 {
                        let ext_x = dx / len * half_width;
                        let ext_y = dy / len * half_width;
                        let perp_x = -dy / len * half_width;
                        let perp_y = dx / len * half_width;
                        let cap = vec![
                            (x2 + perp_x, y2 + perp_y),
                            (x2 - perp_x, y2 - perp_y),
                            (x2 + ext_x - perp_x, y2 + ext_y - perp_y),
                            (x2 + ext_x + perp_x, y2 + ext_y + perp_y),
                        ];
                        ctx.fill_polygon(&cap, color, FillRule::NonZero);
                    }
                }
            }
            LineCap::Butt => {}
        }
    }
}

/// Draw a filled circle
pub fn draw_circle(ctx: &mut RenderContext, cx: f64, cy: f64, radius: f64, color: Color) {
    let segments = 16;
    let mut circle: Vec<(f64, f64)> = Vec::with_capacity(segments);
    for i in 0..segments {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
        circle.push((cx + radius * angle.cos(), cy + radius * angle.sin()));
    }
    ctx.fill_polygon(&circle, color, FillRule::NonZero);
}

/// Find intersection of two lines defined by point and direction
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
