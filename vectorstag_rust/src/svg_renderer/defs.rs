//! Collection of SVG definitions (gradients, markers, clipPaths, masks).

use roxmltree::Node;
use super::types::*;
use super::parsing::*;

/// Collect definitions from a defs element
pub fn collect_defs(ctx: &mut RenderContext, node: &Node) {
    for child in node.children() {
        if !child.is_element() {
            continue;
        }

        let tag = child.tag_name().name();
        if tag == "linearGradient" || tag == "radialGradient" {
            collect_gradient(ctx, &child);
        } else if tag == "pattern" {
            collect_pattern(ctx, &child);
        } else if tag == "marker" {
            collect_marker(ctx, &child);
        }
    }
}

fn collect_pattern(ctx: &mut RenderContext, node: &Node) {
    let id = match node.attribute("id") {
        Some(id) => id,
        None => return,
    };

    let x: f64 = node.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y: f64 = node.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let width: f64 = node.attribute("width").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let height: f64 = node.attribute("height").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let user_space = node.attribute("patternUnits") == Some("userSpaceOnUse");

    let mut rects = Vec::new();
    for child in node.children() {
        if !child.is_element() {
            continue;
        }
        if child.tag_name().name() != "rect" {
            continue;
        }

        let rx: f64 = child.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let ry: f64 = child.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let rw: f64 = child.attribute("width").and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let rh: f64 = child.attribute("height").and_then(|s| s.parse().ok()).unwrap_or(0.0);

        let fill = child.attribute("fill").and_then(parse_color).unwrap_or(Color::from_rgba(0, 0, 0, 0));
        let opacity: f64 = child.attribute("opacity").and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let fill_opacity: f64 = child.attribute("fill-opacity").and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let mut color = fill;
        color.a = (color.a as f64 * opacity * fill_opacity) as u8;

        rects.push(PatternRect { x: rx, y: ry, width: rw, height: rh, color });
    }

    ctx.patterns.insert(id.to_string(), PatternDef {
        id: id.to_string(),
        x,
        y,
        width,
        height,
        user_space,
        rects,
    });
}

/// Collect a gradient definition
fn collect_gradient(ctx: &mut RenderContext, node: &Node) {
    let id = match node.attribute("id") {
        Some(id) => id,
        None => return,
    };

    let tag = node.tag_name().name();
    let is_radial = tag == "radialGradient";

    let mut grad = GradientDef {
        id: id.to_string(),
        is_radial,
        x1: node.attribute("x1").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(0.0),
        y1: node.attribute("y1").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(0.0),
        x2: node.attribute("x2").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(100.0),
        y2: node.attribute("y2").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(0.0),
        cx: node.attribute("cx").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
        cy: node.attribute("cy").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
        r: node.attribute("r").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
        fx: node.attribute("fx").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
        fy: node.attribute("fy").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
        stops: Vec::new(),
        user_space: node.attribute("gradientUnits") == Some("userSpaceOnUse"),
        transform: node.attribute("gradientTransform")
            .map(parse_transform)
            .unwrap_or_default(),
    };

    // Collect stops
    for stop in node.children() {
        if stop.is_element() && stop.tag_name().name() == "stop" {
            let offset: f64 = stop.attribute("offset")
                .and_then(|s| s.trim_end_matches('%').parse().ok())
                .map(|v: f64| if v > 1.0 { v / 100.0 } else { v })
                .unwrap_or(0.0);

            let mut color = Color::from_rgba(0, 0, 0, 255);
            let mut opacity = 1.0f64;

            if let Some(style) = stop.attribute("style") {
                for part in style.split(';') {
                    if let Some(colon) = part.find(':') {
                        let prop = part[..colon].trim();
                        let val = part[colon + 1..].trim();
                        if prop == "stop-color" {
                            if let Some(c) = parse_color(val) {
                                color = c;
                            }
                        } else if prop == "stop-opacity" {
                            if let Ok(o) = val.parse() {
                                opacity = o;
                            }
                        }
                    }
                }
            }

            if let Some(c) = stop.attribute("stop-color").and_then(parse_color) {
                color = c;
            }
            if let Some(o) = stop.attribute("stop-opacity").and_then(|s| s.parse().ok()) {
                opacity = o;
            }

            let a = (color.a as f64 * opacity) as u8;
            grad.stops.push((offset, color.r, color.g, color.b, a));
        }
    }

    ctx.gradients.insert(id.to_string(), grad);
}

/// Collect a marker definition
fn collect_marker(ctx: &mut RenderContext, node: &Node) {
    let id = match node.attribute("id") {
        Some(id) => id,
        None => return,
    };

    let ref_x = node.attribute("refX")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let ref_y = node.attribute("refY")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let marker_width = node.attribute("markerWidth")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let marker_height = node.attribute("markerHeight")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);

    let orient = match node.attribute("orient") {
        Some("auto") => MarkerOrient::Auto,
        Some("auto-start-reverse") => MarkerOrient::AutoStartReverse,
        Some(s) => {
            let angle_str = s.trim_end_matches("deg");
            angle_str.parse::<f64>()
                .map(|deg| MarkerOrient::Angle(deg.to_radians()))
                .unwrap_or(MarkerOrient::Angle(0.0))
        }
        None => MarkerOrient::Angle(0.0),
    };

    let viewbox = node.attribute("viewBox").and_then(parse_viewbox);
    let stroke_width_units = node.attribute("markerUnits") != Some("userSpaceOnUse");

    let marker = MarkerDef {
        id: id.to_string(),
        ref_x,
        ref_y,
        marker_width,
        marker_height,
        orient,
        viewbox,
        stroke_width_units,
        children_xml: String::new(),
    };

    ctx.markers.insert(id.to_string(), marker);
}

/// Recursively collect all gradients from the entire document tree
pub fn collect_all_gradients(ctx: &mut RenderContext, node: &Node) {
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();

    if tag == "linearGradient" || tag == "radialGradient" {
        collect_gradient(ctx, node);
        return;
    }

    for child in node.children() {
        collect_all_gradients(ctx, &child);
    }
}

pub fn collect_all_patterns(ctx: &mut RenderContext, node: &Node) {
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();
    if tag == "pattern" {
        collect_pattern(ctx, node);
        return;
    }

    for child in node.children() {
        collect_all_patterns(ctx, &child);
    }
}

/// Recursively collect all markers from the entire document tree
pub fn collect_all_markers(ctx: &mut RenderContext, node: &Node) {
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();

    if tag == "marker" {
        collect_marker(ctx, node);
        return;
    }

    for child in node.children() {
        collect_all_markers(ctx, &child);
    }
}

/// Collect all clipPath and mask definitions from the document
pub fn collect_clip_paths_and_masks<'a>(ctx: &mut RenderContext, node: &Node<'a, '_>, transform: &Transform, root: &Node<'a, '_>) {
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();

    if tag == "clipPath" {
        collect_clip_path(ctx, node, transform, root);
        return;
    }

    if tag == "mask" {
        collect_mask(ctx, node);
        return;
    }

    for child in node.children() {
        collect_clip_paths_and_masks(ctx, &child, transform, root);
    }
}

/// Collect a clipPath definition
fn collect_clip_path<'a>(ctx: &mut RenderContext, node: &Node<'a, '_>, transform: &Transform, root: &Node<'a, '_>) {
    let id = match node.attribute("id") {
        Some(id) => id,
        None => return,
    };

    let user_space = node.attribute("clipPathUnits") == Some("userSpaceOnUse");
    let mut polygons: Vec<Vec<(f64, f64)>> = Vec::new();

    for child in node.children() {
        if !child.is_element() {
            continue;
        }
        let child_tag = child.tag_name().name();
        match child_tag {
            "path" => {
                if let Some(d) = child.attribute("d") {
                    let child_transform = child.attribute("transform")
                        .map(parse_transform)
                        .unwrap_or_default();
                    let combined = if user_space {
                        transform.multiply(&child_transform)
                    } else {
                        child_transform
                    };
                    let polys = super::path_utils::path_to_polygons(d, &combined);
                    polygons.extend(polys);
                }
            }
            "rect" => {
                let x: f64 = child.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f64 = child.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let w: f64 = child.attribute("width").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let h: f64 = child.attribute("height").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let child_transform = child.attribute("transform")
                    .map(parse_transform)
                    .unwrap_or_default();
                let combined = if user_space {
                    transform.multiply(&child_transform)
                } else {
                    child_transform
                };
                let rect_poly = vec![
                    combined.apply(x, y),
                    combined.apply(x + w, y),
                    combined.apply(x + w, y + h),
                    combined.apply(x, y + h),
                ];
                polygons.push(rect_poly);
            }
            "circle" => {
                let cx: f64 = child.attribute("cx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let cy: f64 = child.attribute("cy").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let r: f64 = child.attribute("r").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let child_transform = child.attribute("transform")
                    .map(parse_transform)
                    .unwrap_or_default();
                let combined = if user_space {
                    transform.multiply(&child_transform)
                } else {
                    child_transform
                };
                let segments = 32;
                let mut circle: Vec<(f64, f64)> = Vec::with_capacity(segments);
                for i in 0..segments {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
                    let px = cx + r * angle.cos();
                    let py = cy + r * angle.sin();
                    circle.push(combined.apply(px, py));
                }
                polygons.push(circle);
            }
            "ellipse" => {
                let cx: f64 = child.attribute("cx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let cy: f64 = child.attribute("cy").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let rx: f64 = child.attribute("rx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let ry: f64 = child.attribute("ry").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let child_transform = child.attribute("transform")
                    .map(parse_transform)
                    .unwrap_or_default();
                let combined = if user_space {
                    transform.multiply(&child_transform)
                } else {
                    child_transform
                };
                let segments = 32;
                let mut ellipse: Vec<(f64, f64)> = Vec::with_capacity(segments);
                for i in 0..segments {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
                    let px = cx + rx * angle.cos();
                    let py = cy + ry * angle.sin();
                    ellipse.push(combined.apply(px, py));
                }
                polygons.push(ellipse);
            }
            "use" => {
                // Handle <use> elements that reference other shapes
                let href = child.attribute("href")
                    .or_else(|| child.attribute(("http://www.w3.org/1999/xlink", "href")));

                if let Some(href) = href {
                    let target_id = href.trim_start_matches('#');
                    if let Some(target) = super::path_utils::find_element_by_id(root, target_id) {
                        // Get x/y offset from use element
                        let use_x: f64 = child.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                        let use_y: f64 = child.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);

                        let use_transform = child.attribute("transform")
                            .map(parse_transform)
                            .unwrap_or_default();

                        // Combine use transform with translation
                        let translation = Transform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: use_x, f: use_y };
                        let use_combined = use_transform.multiply(&translation);

                        let combined = if user_space {
                            transform.multiply(&use_combined)
                        } else {
                            use_combined
                        };

                        // Extract polygons from the referenced element
                        let target_tag = target.tag_name().name();
                        match target_tag {
                            "path" => {
                                if let Some(d) = target.attribute("d") {
                                    let target_transform = target.attribute("transform")
                                        .map(parse_transform)
                                        .unwrap_or_default();
                                    let final_transform = combined.multiply(&target_transform);
                                    let polys = super::path_utils::path_to_polygons(d, &final_transform);
                                    polygons.extend(polys);
                                }
                            }
                            "rect" => {
                                let x: f64 = target.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let y: f64 = target.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let w: f64 = target.attribute("width").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let h: f64 = target.attribute("height").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let target_transform = target.attribute("transform")
                                    .map(parse_transform)
                                    .unwrap_or_default();
                                let final_transform = combined.multiply(&target_transform);
                                let rect_poly = vec![
                                    final_transform.apply(x, y),
                                    final_transform.apply(x + w, y),
                                    final_transform.apply(x + w, y + h),
                                    final_transform.apply(x, y + h),
                                ];
                                polygons.push(rect_poly);
                            }
                            "circle" => {
                                let cx: f64 = target.attribute("cx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let cy: f64 = target.attribute("cy").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let r: f64 = target.attribute("r").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let target_transform = target.attribute("transform")
                                    .map(parse_transform)
                                    .unwrap_or_default();
                                let final_transform = combined.multiply(&target_transform);
                                let segments = 32;
                                let mut circle: Vec<(f64, f64)> = Vec::with_capacity(segments);
                                for i in 0..segments {
                                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
                                    let px = cx + r * angle.cos();
                                    let py = cy + r * angle.sin();
                                    circle.push(final_transform.apply(px, py));
                                }
                                polygons.push(circle);
                            }
                            "ellipse" => {
                                let cx: f64 = target.attribute("cx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let cy: f64 = target.attribute("cy").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let rx: f64 = target.attribute("rx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let ry: f64 = target.attribute("ry").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let target_transform = target.attribute("transform")
                                    .map(parse_transform)
                                    .unwrap_or_default();
                                let final_transform = combined.multiply(&target_transform);
                                let segments = 32;
                                let mut ellipse: Vec<(f64, f64)> = Vec::with_capacity(segments);
                                for i in 0..segments {
                                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
                                    let px = cx + rx * angle.cos();
                                    let py = cy + ry * angle.sin();
                                    ellipse.push(final_transform.apply(px, py));
                                }
                                polygons.push(ellipse);
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }

    ctx.clip_paths.insert(id.to_string(), ClipPathDef {
        id: id.to_string(),
        polygons,
        user_space,
    });
}

/// Collect a mask definition
fn collect_mask(ctx: &mut RenderContext, node: &Node) {
    let id = match node.attribute("id") {
        Some(id) => id,
        None => return,
    };

    let x: f64 = node.attribute("x").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(-10.0);
    let y: f64 = node.attribute("y").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(-10.0);
    let width: f64 = node.attribute("width").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(120.0);
    let height: f64 = node.attribute("height").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(120.0);

    ctx.masks.insert(id.to_string(), MaskDef { id: id.to_string(), x, y, width, height });
}

/// Check if a point is inside a polygon using ray casting algorithm
#[allow(dead_code)]
pub fn point_in_polygon(x: f64, y: f64, polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];

        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Check if a point is inside any of the clip path's polygons
#[allow(dead_code)]
pub fn point_in_clip_path(x: f64, y: f64, clip_path: &ClipPathDef) -> bool {
    for polygon in &clip_path.polygons {
        if point_in_polygon(x, y, polygon) {
            return true;
        }
    }
    false
}
