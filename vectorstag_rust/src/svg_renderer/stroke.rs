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

            match linejoin {
                LineJoin::Round => {
                    draw_circle(ctx, cx, cy, half_width, color);
                }
                LineJoin::Miter => {
                    // For miter joins, we need to handle convex and concave sides differently
                    // The convex side gets a miter point, the concave side gets a bevel

                    if cross.abs() > 0.001 {
                        // SVG miter limit: miter_ratio = 1/sin(θ/2) where θ is interior angle
                        let dot = d1x * d2x + d1y * d2y;
                        let cos_alpha = dot / (len1 * len2);
                        let sin_half_theta_sq = (1.0 + cos_alpha) / 2.0;

                        let miter_limit_sq = miter_limit * miter_limit;
                        let use_miter = sin_half_theta_sq >= 1.0 / miter_limit_sq && sin_half_theta_sq > 0.0001;

                        // Calculate miter point position using line intersection
                        let dx_perp = (perp2_x - perp1_x) * half_width;
                        let dy_perp = (perp2_y - perp1_y) * half_width;
                        let t = (dx_perp * d2y - dy_perp * d2x) / cross;

                        if cross > 0.0 {
                            // Left turn - convex on the - side (outer), concave on the + side (inner)
                            // Fill concave gap on + side with bevel
                            let bevel = vec![(cx, cy), p1_plus, p2_plus];
                            ctx.fill_polygon(&bevel, color, FillRule::NonZero);

                            // Miter on - side if within limit
                            if use_miter {
                                let miter_x = cx - perp1_x * half_width - t * d1x;
                                let miter_y = cy - perp1_y * half_width - t * d1y;
                                let tri1 = vec![p1_minus, (miter_x, miter_y), (cx, cy)];
                                let tri2 = vec![(miter_x, miter_y), p2_minus, (cx, cy)];
                                ctx.fill_polygon(&tri1, color, FillRule::NonZero);
                                ctx.fill_polygon(&tri2, color, FillRule::NonZero);
                            } else {
                                // Bevel fallback
                                let bevel = vec![(cx, cy), p1_minus, p2_minus];
                                ctx.fill_polygon(&bevel, color, FillRule::NonZero);
                            }
                        } else {
                            // Right turn - convex on the + side (outer), concave on the - side (inner)
                            // Fill concave gap on - side with bevel
                            let bevel = vec![(cx, cy), p1_minus, p2_minus];
                            ctx.fill_polygon(&bevel, color, FillRule::NonZero);

                            // Miter on + side if within limit
                            if use_miter {
                                let miter_x = cx + perp1_x * half_width + t * d1x;
                                let miter_y = cy + perp1_y * half_width + t * d1y;
                                let tri1 = vec![p1_plus, (miter_x, miter_y), (cx, cy)];
                                let tri2 = vec![(miter_x, miter_y), p2_plus, (cx, cy)];
                                ctx.fill_polygon(&tri1, color, FillRule::NonZero);
                                ctx.fill_polygon(&tri2, color, FillRule::NonZero);
                            } else {
                                // Bevel fallback
                                let bevel = vec![(cx, cy), p1_plus, p2_plus];
                                ctx.fill_polygon(&bevel, color, FillRule::NonZero);
                            }
                        }
                    } else {
                        // Nearly parallel - just use bevel on both sides
                        let bevel1 = vec![(cx, cy), p1_plus, p2_plus];
                        let bevel2 = vec![(cx, cy), p1_minus, p2_minus];
                        ctx.fill_polygon(&bevel1, color, FillRule::NonZero);
                        ctx.fill_polygon(&bevel2, color, FillRule::NonZero);
                    }
                }
                LineJoin::Bevel => {
                    // Bevel join - triangles from center to edge points on both sides
                    let bevel1 = vec![(cx, cy), p1_plus, p2_plus];
                    let bevel2 = vec![(cx, cy), p1_minus, p2_minus];
                    ctx.fill_polygon(&bevel1, color, FillRule::NonZero);
                    ctx.fill_polygon(&bevel2, color, FillRule::NonZero);
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
