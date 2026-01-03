//! Main node rendering logic.

use roxmltree::Node;
use super::types::*;
use super::parsing::{parse_transform, parse_style};
use super::defs::collect_defs;
use super::shapes::*;
use super::elements::*;
use super::path_utils::find_element_by_id;
use super::preserve_aspect_ratio::{parse_preserve_aspect_ratio, compute_viewbox_transform};

/// Render a node and its children
pub fn render_node(
    ctx: &mut RenderContext,
    node: &Node,
    parent_transform: &Transform,
    parent_style: &Style,
    depth: usize,
    root: &Node,
) {
    // Prevent infinite recursion
    if depth > MAX_DEPTH {
        return;
    }

    // Skip non-element nodes
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();

    // Skip defs (just collect gradients)
    if tag == "defs" {
        collect_defs(ctx, node);
        return;
    }

    // Skip elements that shouldn't be rendered directly
    match tag {
        "metadata" | "title" | "desc" | "style" | "script" => return,
        "filter" | "clipPath" | "mask" | "pattern" | "symbol" | "marker" => return,
        "font" | "font-face" | "glyph" | "missing-glyph" => return,
        "foreignObject" => return,
        _ => {}
    }

    // Parse transform
    let local_transform = node.attribute("transform")
        .map(parse_transform)
        .unwrap_or_default();
    let transform = parent_transform.multiply(&local_transform);

    // Parse style
    let style = parse_style(node, parent_style);

    // Skip elements with display: none
    if !style.display {
        return;
    }

    // Skip elements with visibility: hidden
    if !style.visibility {
        return;
    }

    // Handle clip-path attribute
    let mut prev_clip: Option<Vec<Vec<(f64, f64)>>> = None;
    let mut prev_clip_bbox: Option<(f64, f64, f64, f64)> = None;

    if let Some(clip_attr) = style.display.then(|| node.attribute("clip-path")).flatten() {
        if clip_attr.starts_with("url(#") {
            let id = clip_attr.trim_start_matches("url(#").trim_end_matches(')');
            if let Some(clip_def) = ctx.clip_paths.get(id).cloned() {
                prev_clip = ctx.active_clip.clone();
                prev_clip_bbox = ctx.active_clip_bbox;

                let new_polygons = if clip_def.user_space {
                    clip_def.polygons.clone()
                } else {
                    clip_def.polygons.clone()
                };

                let mut min_x = f64::INFINITY;
                let mut min_y = f64::INFINITY;
                let mut max_x = f64::NEG_INFINITY;
                let mut max_y = f64::NEG_INFINITY;

                for poly in &new_polygons {
                    for point in poly {
                        if point.0 < min_x { min_x = point.0; }
                        if point.0 > max_x { max_x = point.0; }
                        if point.1 < min_y { min_y = point.1; }
                        if point.1 > max_y { max_y = point.1; }
                    }
                }

                ctx.active_clip = Some(new_polygons);
                ctx.active_clip_bbox = Some((min_x, min_y, max_x, max_y));
            }
        }
    }

    // Render based on element type
    match tag {
        "g" => {
            for child in node.children() {
                render_node(ctx, &child, &transform, &style, depth + 1, root);
            }
        }
        "svg" => {
            let x = node.attribute("x")
                .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
                .unwrap_or(0.0);
            let y = node.attribute("y")
                .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
                .unwrap_or(0.0);

            let nested_transform = if x != 0.0 || y != 0.0 {
                transform.multiply(&Transform::translate(x, y))
            } else {
                transform
            };

            for child in node.children() {
                render_node(ctx, &child, &nested_transform, &style, depth + 1, root);
            }
        }
        "switch" => {
            for child in node.children() {
                if !child.is_element() {
                    continue;
                }

                if child.attribute("requiredExtensions").is_some() {
                    continue;
                }

                if child.attribute("requiredFeatures").is_some() {
                    continue;
                }

                if let Some(lang) = child.attribute("systemLanguage") {
                    if !lang.starts_with("en") {
                        continue;
                    }
                }

                render_node(ctx, &child, &transform, &style, depth + 1, root);
                break;
            }
        }
        "path" => {
            if let Some(d) = node.attribute("d") {
                render_path_with_markers(ctx, d, &transform, &style, root);
            }
        }
        "rect" => {
            render_rect(ctx, node, &transform, &style);
        }
        "circle" => {
            render_circle(ctx, node, &transform, &style);
        }
        "ellipse" => {
            render_ellipse(ctx, node, &transform, &style);
        }
        "line" => {
            render_line(ctx, node, &transform, &style, root);
        }
        "polyline" => {
            render_polyline(ctx, node, &transform, &style, root);
        }
        "polygon" => {
            render_polygon_elem(ctx, node, &transform, &style, root);
        }
        "text" => {
            render_text_element(ctx, node, &transform, &style);
        }
        "tspan" | "textPath" => {
            // Handled within render_text_element
        }
        "image" => {
            render_image_element(ctx, node, &transform, &style);
        }
        "use" => {
            let href = node.attribute("href")
                .or_else(|| node.attribute(("http://www.w3.org/1999/xlink", "href")));

            if let Some(href) = href {
                let target_id = href.trim_start_matches('#');

                if let Some(target) = find_element_by_id(root, target_id) {
                    let x = node.attribute("x")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let y = node.attribute("y")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);

                    let use_transform = if x != 0.0 || y != 0.0 {
                        let translate = Transform {
                            a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: x, f: y,
                        };
                        transform.multiply(&translate)
                    } else {
                        transform
                    };

                    let target_tag = target.tag_name().name();
                    if target_tag == "symbol" {
                        render_symbol(ctx, &target, node, &use_transform, &style, depth, root);
                    } else {
                        render_node(ctx, &target, &use_transform, &style, depth + 1, root);
                    }
                }
            }
        }
        "a" => {
            for child in node.children() {
                render_node(ctx, &child, &transform, &style, depth + 1, root);
            }
        }
        _ => {
            // Unknown element - skip
        }
    }

    // Restore previous clip path
    ctx.active_clip = prev_clip;
    ctx.active_clip_bbox = prev_clip_bbox;
}

/// Render a symbol element (used by <use>)
fn render_symbol(
    ctx: &mut RenderContext,
    target: &Node,
    use_node: &Node,
    use_transform: &Transform,
    style: &Style,
    depth: usize,
    root: &Node,
) {
    let use_width: f64 = use_node.attribute("width")
        .and_then(|s| s.trim_end_matches("px").parse().ok())
        .unwrap_or(0.0);
    let use_height: f64 = use_node.attribute("height")
        .and_then(|s| s.trim_end_matches("px").parse().ok())
        .unwrap_or(0.0);

    if let Some(viewbox_str) = target.attribute("viewBox") {
        let parts: Vec<f64> = viewbox_str
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if parts.len() == 4 {
            let (vb_x, vb_y, vb_w, vb_h) = (parts[0], parts[1], parts[2], parts[3]);

            let (viewport_w, viewport_h) = root.attribute("viewBox")
                .and_then(|vb| {
                    let p: Vec<f64> = vb.split(|c: char| c == ',' || c.is_whitespace())
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if p.len() == 4 { Some((p[2], p[3])) } else { None }
                })
                .unwrap_or((vb_w, vb_h));

            let target_w = if use_width > 0.0 { use_width } else { viewport_w };
            let target_h = if use_height > 0.0 { use_height } else { viewport_h };

            let par = target.attribute("preserveAspectRatio")
                .map(parse_preserve_aspect_ratio)
                .unwrap_or_default();

            let viewbox_transform = compute_viewbox_transform(
                vb_x, vb_y, vb_w, vb_h,
                target_w, target_h,
                par
            );

            let symbol_transform = use_transform.multiply(&viewbox_transform);

            for child in target.children() {
                render_node(ctx, &child, &symbol_transform, style, depth + 1, root);
            }
        } else {
            for child in target.children() {
                render_node(ctx, &child, use_transform, style, depth + 1, root);
            }
        }
    } else {
        for child in target.children() {
            render_node(ctx, &child, use_transform, style, depth + 1, root);
        }
    }
}
