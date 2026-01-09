//! Basic shape rendering (rect, circle, ellipse, line, polyline, polygon).

use roxmltree::Node;
use super::types::*;
use super::parsing::{parse_points, parse_length_percent, parse_radius_percent};
use super::stroke::render_stroke;
use super::markers::render_markers;

/// Render a rectangle element
pub fn render_rect(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    if !style.visibility { return; }
    ctx.increment_shapes();

    // Get viewport dimensions for percent calculations
    let vp_w = ctx.viewport_width;
    let vp_h = ctx.viewport_height;

    // Parse with percent support
    let x: f64 = node.attribute("x")
        .map(|s| parse_length_percent(s, vp_w, 0.0))
        .unwrap_or(0.0);
    let y: f64 = node.attribute("y")
        .map(|s| parse_length_percent(s, vp_h, 0.0))
        .unwrap_or(0.0);
    let w: f64 = node.attribute("width")
        .map(|s| parse_length_percent(s, vp_w, 0.0))
        .unwrap_or(0.0);
    let h: f64 = node.attribute("height")
        .map(|s| parse_length_percent(s, vp_h, 0.0))
        .unwrap_or(0.0);

    if w <= 0.0 || h <= 0.0 {
        return;
    }

    // Parse rx/ry for rounded corners
    let mut rx: f64 = node.attribute("rx")
        .map(|s| parse_length_percent(s, vp_w, 0.0))
        .unwrap_or(0.0);
    let mut ry: f64 = node.attribute("ry")
        .map(|s| parse_length_percent(s, vp_h, 0.0))
        .unwrap_or(0.0);

    // Per SVG spec: if only rx or ry is specified, the other defaults to it
    if rx > 0.0 && ry == 0.0 { ry = rx; }
    if ry > 0.0 && rx == 0.0 { rx = ry; }

    // Clamp to half width/height
    rx = rx.min(w / 2.0);
    ry = ry.min(h / 2.0);

    let corners = if rx > 0.0 && ry > 0.0 {
        // Rounded rectangle
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let segments = 8;

        // Top-right corner
        for i in 0..=segments {
            let angle = std::f64::consts::PI * 1.5 + (std::f64::consts::PI / 2.0) * (i as f64 / segments as f64);
            let px = x + w - rx + rx * angle.cos();
            let py = y + ry + ry * angle.sin();
            pts.push(transform.apply(px, py));
        }
        // Bottom-right corner
        for i in 0..=segments {
            let angle = (std::f64::consts::PI / 2.0) * (i as f64 / segments as f64);
            let px = x + w - rx + rx * angle.cos();
            let py = y + h - ry + ry * angle.sin();
            pts.push(transform.apply(px, py));
        }
        // Bottom-left corner
        for i in 0..=segments {
            let angle = std::f64::consts::PI / 2.0 + (std::f64::consts::PI / 2.0) * (i as f64 / segments as f64);
            let px = x + rx + rx * angle.cos();
            let py = y + h - ry + ry * angle.sin();
            pts.push(transform.apply(px, py));
        }
        // Top-left corner
        for i in 0..=segments {
            let angle = std::f64::consts::PI + (std::f64::consts::PI / 2.0) * (i as f64 / segments as f64);
            let px = x + rx + rx * angle.cos();
            let py = y + ry + ry * angle.sin();
            pts.push(transform.apply(px, py));
        }
        pts
    } else {
        // Regular rectangle
        vec![
            transform.apply(x, y),
            transform.apply(x + w, y),
            transform.apply(x + w, y + h),
            transform.apply(x, y + h),
        ]
    };

    if let Some(ref fill) = style.fill {
        match fill {
            Paint::Color(color) => {
                let mut c = *color;
                c.a = (c.a as f64 * style.fill_opacity * style.opacity) as u8;
                ctx.fill_polygon(&corners, c, style.fill_rule);
            }
            Paint::Ref(id) => {
                let opacity = style.fill_opacity * style.opacity;
                if let Some(gradient) = ctx.gradients.get(id).cloned() {
                    ctx.fill_polygon_gradient(&corners, &gradient, transform, style.fill_rule, opacity);
                } else if let Some(pattern) = ctx.patterns.get(id).cloned() {
                    ctx.fill_polygon_pattern(&corners, &pattern, style.fill_rule, opacity);
                }
            }
            Paint::None => {}
        }
    }

    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                render_stroke(ctx, &corners, c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, true, style.stroke_miterlimit);
            }
        }
    }
}

/// Render a circle element
pub fn render_circle(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    if !style.visibility { return; }
    ctx.increment_shapes();

    // Get viewport dimensions for percent calculations
    let vp_w = ctx.viewport_width;
    let vp_h = ctx.viewport_height;

    // Parse cx, cy with percent support (relative to viewport width/height)
    let cx: f64 = node.attribute("cx")
        .map(|s| parse_length_percent(s, vp_w, 0.0))
        .unwrap_or(0.0);
    let cy: f64 = node.attribute("cy")
        .map(|s| parse_length_percent(s, vp_h, 0.0))
        .unwrap_or(0.0);
    // Parse r with percent support (relative to sqrt((w^2 + h^2)/2))
    // SVG: negative r means error - don't render
    let r: f64 = node.attribute("r")
        .map(|s| parse_radius_percent(s, vp_w, vp_h, 0.0))
        .unwrap_or(0.0);

    if r <= 0.0 {
        return;
    }

    let n = 32;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        points.push(transform.apply(x, y));
    }

    if let Some(ref fill) = style.fill {
        match fill {
            Paint::Color(color) => {
                let mut c = *color;
                c.a = (c.a as f64 * style.fill_opacity * style.opacity) as u8;
                ctx.fill_polygon(&points, c, style.fill_rule);
            }
            Paint::Ref(id) => {
                let opacity = style.fill_opacity * style.opacity;
                if let Some(gradient) = ctx.gradients.get(id).cloned() {
                    ctx.fill_polygon_gradient(&points, &gradient, transform, style.fill_rule, opacity);
                } else if let Some(pattern) = ctx.patterns.get(id).cloned() {
                    ctx.fill_polygon_pattern(&points, &pattern, style.fill_rule, opacity);
                }
            }
            Paint::None => {}
        }
    }

    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                render_stroke(ctx, &points, c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, true, style.stroke_miterlimit);
            }
        }
    }
}

/// Render an ellipse element
pub fn render_ellipse(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    if !style.visibility { return; }
    ctx.increment_shapes();

    // Get viewport dimensions for percent calculations
    let vp_w = ctx.viewport_width;
    let vp_h = ctx.viewport_height;

    // Parse with percent support
    let cx: f64 = node.attribute("cx")
        .map(|s| parse_length_percent(s, vp_w, 0.0))
        .unwrap_or(0.0);
    let cy: f64 = node.attribute("cy")
        .map(|s| parse_length_percent(s, vp_h, 0.0))
        .unwrap_or(0.0);

    // SVG 2: Handle missing and negative radii
    let rx_attr = node.attribute("rx");
    let ry_attr = node.attribute("ry");
    let rx_raw: f64 = rx_attr
        .map(|s| parse_length_percent(s, vp_w, 0.0))
        .unwrap_or(0.0);
    let ry_raw: f64 = ry_attr
        .map(|s| parse_length_percent(s, vp_h, 0.0))
        .unwrap_or(0.0);

    // If both are negative, don't render
    if rx_raw < 0.0 && ry_raw < 0.0 {
        return;
    }

    // Use absolute values for remaining calculations
    let mut rx = rx_raw.abs();
    let mut ry = ry_raw.abs();

    // If one radius is missing, use the other
    if rx_attr.is_none() && ry_attr.is_some() {
        rx = ry;
    } else if ry_attr.is_none() && rx_attr.is_some() {
        ry = rx;
    }

    if rx == 0.0 || ry == 0.0 {
        return;
    }

    let n = 32;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let x = cx + rx * angle.cos();
        let y = cy + ry * angle.sin();
        points.push(transform.apply(x, y));
    }

    if let Some(ref fill) = style.fill {
        match fill {
            Paint::Color(color) => {
                let mut c = *color;
                c.a = (c.a as f64 * style.fill_opacity * style.opacity) as u8;
                ctx.fill_polygon(&points, c, style.fill_rule);
            }
            Paint::Ref(id) => {
                let opacity = style.fill_opacity * style.opacity;
                if let Some(gradient) = ctx.gradients.get(id).cloned() {
                    ctx.fill_polygon_gradient(&points, &gradient, transform, style.fill_rule, opacity);
                } else if let Some(pattern) = ctx.patterns.get(id).cloned() {
                    ctx.fill_polygon_pattern(&points, &pattern, style.fill_rule, opacity);
                }
            }
            Paint::None => {}
        }
    }

    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                render_stroke(ctx, &points, c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, true, style.stroke_miterlimit);
            }
        }
    }
}

/// Render a line element
pub fn render_line(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style, root: &Node) {
    if !ctx.can_render_more() { return; }
    if !style.visibility { return; }
    ctx.increment_shapes();

    let x1: f64 = node.attribute("x1").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y1: f64 = node.attribute("y1").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let x2: f64 = node.attribute("x2").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y2: f64 = node.attribute("y2").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let p1 = transform.apply(x1, y1);
    let p2 = transform.apply(x2, y2);
    let points = vec![p1, p2];

    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                render_stroke(ctx, &points, c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, false, style.stroke_miterlimit);
            }
        }
    }

    // Render markers
    let has_markers = style.marker_start.is_some() || style.marker_mid.is_some() || style.marker_end.is_some();
    if has_markers {
        render_markers(ctx, &points, style, transform, root);
    }
}

/// Render a polyline element
pub fn render_polyline(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style, root: &Node) {
    if !ctx.can_render_more() { return; }
    if !style.visibility { return; }
    ctx.increment_shapes();

    let points = parse_points(node.attribute("points").unwrap_or(""), transform);

    if points.len() < 2 {
        return;
    }

    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                render_stroke(ctx, &points, c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, false, style.stroke_miterlimit);
            }
        }
    }

    // Render markers
    let has_markers = style.marker_start.is_some() || style.marker_mid.is_some() || style.marker_end.is_some();
    if has_markers && points.len() >= 2 {
        render_markers(ctx, &points, style, transform, root);
    }
}

/// Render a polygon element
pub fn render_polygon_elem(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style, root: &Node) {
    if !ctx.can_render_more() { return; }
    if !style.visibility { return; }
    ctx.increment_shapes();

    let points = parse_points(node.attribute("points").unwrap_or(""), transform);

    if points.len() < 3 {
        return;
    }

    if let Some(ref fill) = style.fill {
        match fill {
            Paint::Color(color) => {
                let mut c = *color;
                c.a = (c.a as f64 * style.fill_opacity * style.opacity) as u8;
                ctx.fill_polygon(&points, c, style.fill_rule);
            }
            Paint::Ref(id) => {
                let opacity = style.fill_opacity * style.opacity;
                if let Some(gradient) = ctx.gradients.get(id).cloned() {
                    ctx.fill_polygon_gradient(&points, &gradient, transform, style.fill_rule, opacity);
                } else if let Some(pattern) = ctx.patterns.get(id).cloned() {
                    ctx.fill_polygon_pattern(&points, &pattern, style.fill_rule, opacity);
                }
            }
            Paint::None => {}
        }
    }

    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                render_stroke(ctx, &points, c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, true, style.stroke_miterlimit);
            }
        }
    }

    // Render markers
    let has_markers = style.marker_start.is_some() || style.marker_mid.is_some() || style.marker_end.is_some();
    if has_markers && points.len() >= 2 {
        render_markers(ctx, &points, style, transform, root);
    }
}
