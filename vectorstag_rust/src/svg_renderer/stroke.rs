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

    // Draw round linejoins at internal vertices
    if matches!(linejoin, LineJoin::Round) && points.len() > 2 {
        let start = if closed { 0 } else { 1 };
        let end = if closed { points.len() } else { points.len() - 1 };
        for i in start..end {
            let (cx, cy) = points[i];
            draw_circle(ctx, cx, cy, half_width, color);
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
