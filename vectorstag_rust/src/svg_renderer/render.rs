//! Main node rendering logic.

use roxmltree::Node;
use super::types::*;
use super::parsing::{parse_transform, parse_style, parse_viewbox, parse_length, parse_length_percent, parse_transform_origin, get_element_bbox, apply_transform_origin};
use super::defs::collect_defs;
use super::shapes::*;
use super::elements::*;
use super::path_utils::find_element_by_id;
use super::preserve_aspect_ratio::{parse_preserve_aspect_ratio, compute_viewbox_transform};
use super::filter::apply_filter;

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

    // Parse transform with transform-origin support
    let mut local_transform = node.attribute("transform")
        .map(parse_transform)
        .unwrap_or_default();

    // Apply transform-origin if present
    if let Some(origin_str) = node.attribute("transform-origin") {
        let origin = parse_transform_origin(origin_str);
        let bbox = get_element_bbox(node);
        local_transform = apply_transform_origin(&local_transform, &origin, bbox);
    }

    let transform = parent_transform.multiply(&local_transform);

    // Parse style
    let style = parse_style(node, parent_style);

    // Skip elements with display: none (removes element and all children)
    if !style.display {
        return;
    }

    // Note: visibility:hidden is handled per-element, not here
    // Container elements (g, svg, etc.) still process children even if hidden
    // Only leaf elements (path, rect, etc.) check visibility before rendering

    // Check for filter attribute
    let filter_id = node.attribute("filter")
        .or_else(|| {
            // Also check style attribute for filter
            node.attribute("style").and_then(|s| {
                for part in s.split(';') {
                    if let Some(colon) = part.find(':') {
                        let prop = part[..colon].trim();
                        let val = part[colon + 1..].trim();
                        if prop == "filter" {
                            return Some(val);
                        }
                    }
                }
                None
            })
        })
        .and_then(|s| {
            if s.starts_with("url(#") {
                Some(s.trim_start_matches("url(#").trim_end_matches(')'))
            } else {
                None
            }
        });

    // If element has a filter, render to temporary buffer and apply filter
    if let Some(filter_id) = filter_id {
        if let Some(filter_def) = ctx.filters.get(filter_id).cloned() {
            render_with_filter(ctx, node, parent_transform, parent_style, depth, root, &filter_def);
            return;
        }
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
            // Handle nested SVG with proper viewport and viewBox support
            // Use parent viewport for percent values
            let parent_vp_w = ctx.viewport_width;
            let parent_vp_h = ctx.viewport_height;

            let x = node.attribute("x")
                .map(|s| parse_length_percent(s, parent_vp_w, 0.0))
                .unwrap_or(0.0);
            let y = node.attribute("y")
                .map(|s| parse_length_percent(s, parent_vp_h, 0.0))
                .unwrap_or(0.0);

            // Get viewport size (width/height of nested SVG)
            let width = node.attribute("width")
                .map(|s| parse_length_percent(s, parent_vp_w, 0.0))
                .unwrap_or(0.0);
            let height = node.attribute("height")
                .map(|s| parse_length_percent(s, parent_vp_h, 0.0))
                .unwrap_or(0.0);

            // Parse viewBox if present
            let viewbox = node.attribute("viewBox").and_then(parse_viewbox);

            // Start with x/y translation
            let mut nested_transform = transform.multiply(&Transform::translate(x, y));

            // Determine the new viewport dimensions for percent calculations
            // Use viewBox dimensions if present, otherwise use width/height
            let (new_viewport_w, new_viewport_h) = if let Some((_, _, vb_w, vb_h)) = viewbox {
                (vb_w, vb_h)
            } else if width > 0.0 && height > 0.0 {
                (width, height)
            } else {
                // Use parent viewport if nothing specified
                (ctx.viewport_width, ctx.viewport_height)
            };

            // If viewBox is present and we have a valid viewport size, apply viewBox transform
            if let Some((vb_x, vb_y, vb_w, vb_h)) = viewbox {
                if width > 0.0 && height > 0.0 && vb_w > 0.0 && vb_h > 0.0 {
                    // Parse preserveAspectRatio
                    let par = node.attribute("preserveAspectRatio")
                        .map(parse_preserve_aspect_ratio)
                        .unwrap_or_default();

                    let viewbox_transform = compute_viewbox_transform(
                        vb_x, vb_y, vb_w, vb_h,
                        width, height,
                        par
                    );

                    nested_transform = nested_transform.multiply(&viewbox_transform);
                }
            }

            // Set up viewport clipping if width/height are specified
            // Check overflow attribute (default is "hidden" for nested SVG)
            let overflow = node.attribute("overflow").unwrap_or("hidden");
            let should_clip = (overflow == "hidden" || overflow == "scroll") && width > 0.0 && height > 0.0;

            // Save previous state
            let prev_clip = ctx.active_clip.clone();
            let prev_clip_bbox = ctx.active_clip_bbox;
            let prev_viewport_w = ctx.viewport_width;
            let prev_viewport_h = ctx.viewport_height;

            // Update viewport dimensions for children
            ctx.viewport_width = new_viewport_w;
            ctx.viewport_height = new_viewport_h;

            if should_clip {
                // Create clip rectangle in screen coordinates
                let (x1, y1) = transform.apply(x, y);
                let (x2, y2) = transform.apply(x + width, y);
                let (x3, y3) = transform.apply(x + width, y + height);
                let (x4, y4) = transform.apply(x, y + height);

                let clip_polygon = vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4)];

                // Calculate bbox
                let min_x = x1.min(x2).min(x3).min(x4);
                let max_x = x1.max(x2).max(x3).max(x4);
                let min_y = y1.min(y2).min(y3).min(y4);
                let max_y = y1.max(y2).max(y3).max(y4);

                ctx.active_clip = Some(vec![clip_polygon]);
                ctx.active_clip_bbox = Some((min_x, min_y, max_x, max_y));
            }

            for child in node.children() {
                render_node(ctx, &child, &nested_transform, &style, depth + 1, root);
            }

            // Restore previous state
            ctx.viewport_width = prev_viewport_w;
            ctx.viewport_height = prev_viewport_h;
            if should_clip {
                ctx.active_clip = prev_clip;
                ctx.active_clip_bbox = prev_clip_bbox;
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

/// Render a node with a filter applied
fn render_with_filter(
    ctx: &mut RenderContext,
    node: &Node,
    parent_transform: &Transform,
    parent_style: &Style,
    depth: usize,
    root: &Node,
    filter_def: &FilterDef,
) {
    let width = ctx.width;
    let height = ctx.height;

    // Calculate scale factor from transform for scaling filter parameters
    // The transform includes both viewBox->output scale and antialiasing
    let scale_x = (parent_transform.a * parent_transform.a + parent_transform.b * parent_transform.b).sqrt();
    let scale_y = (parent_transform.c * parent_transform.c + parent_transform.d * parent_transform.d).sqrt();
    let scale = (scale_x + scale_y) / 2.0;

    // Save current buffer
    let original_buffer = std::mem::take(&mut ctx.buffer);

    // Create transparent temporary buffer
    ctx.buffer = vec![0u8; width * height * 4];

    // Create a modified node context without the filter attribute
    // We render the element normally (the filter attribute is already extracted)
    render_node_without_filter(ctx, node, parent_transform, parent_style, depth, root);

    // Get element bbox for filter units
    let bbox = get_element_bbox(node);

    // Apply filter to the temporary buffer with scale factor
    let filtered = apply_filter(filter_def, &ctx.buffer, &original_buffer, width, height, parent_transform, bbox);

    // Restore original buffer
    ctx.buffer = original_buffer;

    // Composite filtered result onto original buffer
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let src_r = filtered[idx];
            let src_g = filtered[idx + 1];
            let src_b = filtered[idx + 2];
            let src_a = filtered[idx + 3];

            if src_a == 0 {
                continue;
            }

            let src_a_f = src_a as f32 / 255.0;

            if src_a == 255 {
                ctx.buffer[idx] = src_r;
                ctx.buffer[idx + 1] = src_g;
                ctx.buffer[idx + 2] = src_b;
                ctx.buffer[idx + 3] = 255;
            } else {
                let dst_a_f = ctx.buffer[idx + 3] as f32 / 255.0;
                let out_a = src_a_f + dst_a_f * (1.0 - src_a_f);

                if out_a > 0.0 {
                    let blend = |src: u8, dst: u8| -> u8 {
                        let s = src as f32;
                        let d = dst as f32;
                        ((s * src_a_f + d * dst_a_f * (1.0 - src_a_f)) / out_a) as u8
                    };

                    ctx.buffer[idx] = blend(src_r, ctx.buffer[idx]);
                    ctx.buffer[idx + 1] = blend(src_g, ctx.buffer[idx + 1]);
                    ctx.buffer[idx + 2] = blend(src_b, ctx.buffer[idx + 2]);
                    ctx.buffer[idx + 3] = (out_a * 255.0) as u8;
                }
            }
        }
    }
}

/// Render a node without applying its filter attribute (used internally by render_with_filter)
fn render_node_without_filter(
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

    // Parse transform with transform-origin support
    let mut local_transform = node.attribute("transform")
        .map(parse_transform)
        .unwrap_or_default();

    // Apply transform-origin if present
    if let Some(origin_str) = node.attribute("transform-origin") {
        let origin = parse_transform_origin(origin_str);
        let bbox = get_element_bbox(node);
        local_transform = apply_transform_origin(&local_transform, &origin, bbox);
    }

    let transform = parent_transform.multiply(&local_transform);

    // Parse style
    let style = parse_style(node, parent_style);

    // Skip elements with display: none
    if !style.display {
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

                let new_polygons = clip_def.polygons.clone();

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

    // Render based on element type (same as render_node but without filter check)
    match tag {
        "g" => {
            for child in node.children() {
                render_node(ctx, &child, &transform, &style, depth + 1, root);
            }
        }
        "svg" => {
            // Handle nested SVG
            let parent_vp_w = ctx.viewport_width;
            let parent_vp_h = ctx.viewport_height;

            let x = node.attribute("x")
                .map(|s| parse_length_percent(s, parent_vp_w, 0.0))
                .unwrap_or(0.0);
            let y = node.attribute("y")
                .map(|s| parse_length_percent(s, parent_vp_h, 0.0))
                .unwrap_or(0.0);

            let width = node.attribute("width")
                .map(|s| parse_length_percent(s, parent_vp_w, 0.0))
                .unwrap_or(0.0);
            let height = node.attribute("height")
                .map(|s| parse_length_percent(s, parent_vp_h, 0.0))
                .unwrap_or(0.0);

            let viewbox = node.attribute("viewBox").and_then(parse_viewbox);
            let mut nested_transform = transform.multiply(&Transform::translate(x, y));

            let (new_viewport_w, new_viewport_h) = if let Some((_, _, vb_w, vb_h)) = viewbox {
                (vb_w, vb_h)
            } else if width > 0.0 && height > 0.0 {
                (width, height)
            } else {
                (ctx.viewport_width, ctx.viewport_height)
            };

            if let Some((vb_x, vb_y, vb_w, vb_h)) = viewbox {
                if width > 0.0 && height > 0.0 && vb_w > 0.0 && vb_h > 0.0 {
                    let par = node.attribute("preserveAspectRatio")
                        .map(parse_preserve_aspect_ratio)
                        .unwrap_or_default();

                    let viewbox_transform = compute_viewbox_transform(
                        vb_x, vb_y, vb_w, vb_h,
                        width, height,
                        par
                    );

                    nested_transform = nested_transform.multiply(&viewbox_transform);
                }
            }

            let prev_viewport_w = ctx.viewport_width;
            let prev_viewport_h = ctx.viewport_height;
            ctx.viewport_width = new_viewport_w;
            ctx.viewport_height = new_viewport_h;

            for child in node.children() {
                render_node(ctx, &child, &nested_transform, &style, depth + 1, root);
            }

            ctx.viewport_width = prev_viewport_w;
            ctx.viewport_height = prev_viewport_h;
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
        "image" => {
            render_image_element(ctx, node, &transform, &style);
        }
        "use" => {
            let href = node.attribute("href")
                .or_else(|| node.attribute(("http://www.w3.org/1999/xlink", "href")));

            if let Some(href) = href {
                let target_id = href.trim_start_matches('#');

                if let Some(target) = find_element_by_id(root, target_id) {
                    let ux = node.attribute("x")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let uy = node.attribute("y")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);

                    let use_transform = if ux != 0.0 || uy != 0.0 {
                        transform.multiply(&Transform::translate(ux, uy))
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
        _ => {}
    }

    // Restore previous clip path
    ctx.active_clip = prev_clip;
    ctx.active_clip_bbox = prev_clip_bbox;
}
