//! Complex element rendering (path, text, image).

use roxmltree::Node;
use super::types::*;
use super::parsing::{parse_length, decode_data_url};
use super::path_utils::{path_to_polygons, commands_to_polygons};
use super::stroke::render_stroke;
use super::markers::render_markers;

/// Render a path element with optional markers
pub fn render_path_with_markers(
    ctx: &mut RenderContext,
    d: &str,
    transform: &Transform,
    style: &Style,
    root: &Node,
) {
    if !ctx.can_render_more() {
        return;
    }

    let polygons = path_to_polygons(d, transform);

    for poly in &polygons {
        if poly.len() > MAX_POLYGON_POINTS {
            continue;
        }

        // Fill
        if let Some(ref fill) = style.fill {
            match fill {
                Paint::Color(color) => {
                    let mut c = *color;
                    c.a = (c.a as f64 * style.fill_opacity * style.opacity) as u8;
                    ctx.fill_polygon(poly, c, style.fill_rule);
                }
                Paint::Gradient(id) => {
                    if let Some(gradient) = ctx.gradients.get(id).cloned() {
                        let opacity = style.fill_opacity * style.opacity;
                        ctx.fill_polygon_gradient(poly, &gradient, transform, style.fill_rule, opacity);
                    }
                }
                Paint::None => {}
            }
        }
    }

    ctx.increment_shapes();

    // Stroke
    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                for poly in &polygons {
                    if poly.len() > MAX_POLYGON_POINTS {
                        continue;
                    }
                    render_stroke(ctx, poly, c, style.stroke_width * transform.a.abs(),
                        style.stroke_linecap, style.stroke_linejoin, false);
                }
            }
        }
    }

    // Render markers on each polygon
    let has_markers = style.marker_start.is_some() || style.marker_mid.is_some() || style.marker_end.is_some();
    if has_markers {
        for poly in &polygons {
            if poly.len() >= 2 && poly.len() <= MAX_POLYGON_POINTS {
                render_markers(ctx, poly, style, transform, root);
            }
        }
    }
}

/// Render a text element
pub fn render_text_element(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();

    let x: f64 = node.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y: f64 = node.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let text_content = collect_text_content(node);
    if text_content.is_empty() {
        return;
    }

    let font_family = &style.font_family;
    let font_size = style.font_size;
    let font_weight = style.font_weight;
    let italic = style.font_style == "italic" || style.font_style == "oblique";
    let text_anchor = &style.text_anchor;

    let glyph_paths = crate::text::layout_text(
        &text_content,
        x,
        y,
        font_family,
        font_size,
        font_weight,
        italic,
        text_anchor,
        &ctx.font_manager,
    );

    for glyph_commands in glyph_paths {
        let polygons = commands_to_polygons(&glyph_commands, transform);

        for poly in &polygons {
            if poly.len() < 3 {
                continue;
            }

            if let Some(ref fill) = style.fill {
                match fill {
                    Paint::Color(color) => {
                        let mut c = *color;
                        c.a = (c.a as f64 * style.fill_opacity * style.opacity) as u8;
                        ctx.fill_polygon(poly, c, style.fill_rule);
                    }
                    Paint::Gradient(id) => {
                        if let Some(gradient) = ctx.gradients.get(id).cloned() {
                            let opacity = style.fill_opacity * style.opacity;
                            ctx.fill_polygon_gradient(poly, &gradient, transform, style.fill_rule, opacity);
                        }
                    }
                    Paint::None => {}
                }
            }
        }
    }
}

/// Collect text content from a text element (including tspan children)
pub fn collect_text_content(node: &Node) -> String {
    let mut content = String::new();

    for child in node.children() {
        if child.is_text() {
            if let Some(text) = child.text() {
                content.push_str(text);
            }
        } else if child.is_element() && child.tag_name().name() == "tspan" {
            content.push_str(&collect_text_content(&child));
        }
    }

    content
}

/// Render an image element (embedded or external)
pub fn render_image_element(ctx: &mut RenderContext, node: &Node, transform: &Transform, _style: &Style) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();

    let x: f64 = node.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y: f64 = node.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let width: f64 = node.attribute("width").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
    let height: f64 = node.attribute("height").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);

    if width <= 0.0 || height <= 0.0 {
        return;
    }

    let href = node.attribute("href")
        .or_else(|| node.attribute(("http://www.w3.org/1999/xlink", "href")));

    let href = match href {
        Some(h) => h,
        None => return,
    };

    // Only handle data: URLs for now
    if !href.starts_with("data:") {
        return;
    }

    let img_data = match decode_data_url(href) {
        Some(data) => data,
        None => return,
    };

    let img = match image::load_from_memory(&img_data) {
        Ok(img) => img.to_rgba8(),
        Err(_) => return,
    };

    let (dst_x1, dst_y1) = transform.apply(x, y);
    let (dst_x2, dst_y2) = transform.apply(x + width, y + height);

    let dst_x = dst_x1.min(dst_x2) as i32;
    let dst_y = dst_y1.min(dst_y2) as i32;
    let dst_w = (dst_x2 - dst_x1).abs() as u32;
    let dst_h = (dst_y2 - dst_y1).abs() as u32;

    if dst_w == 0 || dst_h == 0 {
        return;
    }

    let resized = image::imageops::resize(&img, dst_w, dst_h, image::imageops::FilterType::Lanczos3);

    let canvas_w = ctx.width as i32;
    let canvas_h = ctx.height as i32;

    for (img_y, row) in resized.enumerate_rows() {
        let canvas_y = dst_y + img_y as i32;
        if canvas_y < 0 || canvas_y >= canvas_h {
            continue;
        }

        for (img_x, _, pixel) in row {
            let canvas_x = dst_x + img_x as i32;
            if canvas_x < 0 || canvas_x >= canvas_w {
                continue;
            }

            let [r, g, b, a] = pixel.0;
            if a == 0 {
                continue;
            }

            let idx = (canvas_y as usize * ctx.width + canvas_x as usize) * 4;
            if idx + 3 >= ctx.buffer.len() {
                continue;
            }

            let sa = a as f32 / 255.0;
            let da = ctx.buffer[idx + 3] as f32 / 255.0;
            let out_a = sa + da * (1.0 - sa);

            if out_a > 0.0 {
                let blend = |s: u8, d: u8| -> u8 {
                    ((s as f32 * sa + d as f32 * da * (1.0 - sa)) / out_a) as u8
                };
                ctx.buffer[idx] = blend(r, ctx.buffer[idx]);
                ctx.buffer[idx + 1] = blend(g, ctx.buffer[idx + 1]);
                ctx.buffer[idx + 2] = blend(b, ctx.buffer[idx + 2]);
                ctx.buffer[idx + 3] = (out_a * 255.0) as u8;
            }
        }
    }
}
