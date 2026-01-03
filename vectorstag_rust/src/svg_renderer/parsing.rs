//! Parsing functions for SVG attributes.

use roxmltree::Node;
use super::types::*;
pub use super::types::Transform;

/// Convert HSL to RGB
/// h: hue in degrees (0-360)
/// s: saturation (0-1)
/// l: lightness (0-1)
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    // Normalize hue to 0-360 range
    let h = ((h % 360.0) + 360.0) % 360.0;
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);

    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let r = ((r1 + m) * 255.0).round() as u8;
    let g = ((g1 + m) * 255.0).round() as u8;
    let b = ((b1 + m) * 255.0).round() as u8;

    (r, g, b)
}

/// Parse color from string
pub fn parse_color(s: &str) -> Option<Color> {
    let s = s.trim();

    if s == "none" || s == "transparent" {
        return None;
    }

    if s == "currentColor" {
        return Some(Color::from_rgba(0, 0, 0, 255));
    }

    if s.starts_with('#') {
        let hex = &s[1..];
        return match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                Some(Color::from_rgba(r, g, b, 255))
            }
            4 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                let a = u8::from_str_radix(&hex[3..4], 16).ok()? * 17;
                Some(Color::from_rgba(r, g, b, a))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Color::from_rgba(r, g, b, 255))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                Some(Color::from_rgba(r, g, b, a))
            }
            _ => None,
        };
    }

    if s.starts_with("rgb(") {
        let inner = s.trim_start_matches("rgb(").trim_end_matches(')');
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 3 {
            let r: u8 = parts[0].trim().parse().ok()?;
            let g: u8 = parts[1].trim().parse().ok()?;
            let b: u8 = parts[2].trim().parse().ok()?;
            return Some(Color::from_rgba(r, g, b, 255));
        }
    }

    if s.starts_with("rgba(") {
        let inner = s.trim_start_matches("rgba(").trim_end_matches(')');
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 4 {
            let r: u8 = parts[0].trim().parse().ok()?;
            let g: u8 = parts[1].trim().parse().ok()?;
            let b: u8 = parts[2].trim().parse().ok()?;
            let a: f32 = parts[3].trim().parse().ok()?;
            return Some(Color::from_rgba(r, g, b, (a * 255.0) as u8));
        }
    }

    // HSL/HSLA colors
    if s.starts_with("hsl(") || s.starts_with("hsla(") {
        let is_hsla = s.starts_with("hsla(");
        let inner = if is_hsla {
            s.trim_start_matches("hsla(").trim_end_matches(')')
        } else {
            s.trim_start_matches("hsl(").trim_end_matches(')')
        };
        let parts: Vec<&str> = inner.split(',').collect();
        let expected_parts = if is_hsla { 4 } else { 3 };
        if parts.len() == expected_parts {
            // Hue in degrees (0-360)
            let h: f64 = parts[0].trim().trim_end_matches("deg").parse().ok()?;
            // Saturation as percentage
            let s_str = parts[1].trim().trim_end_matches('%');
            let s_val: f64 = s_str.parse().ok()?;
            // Lightness as percentage
            let l_str = parts[2].trim().trim_end_matches('%');
            let l_val: f64 = l_str.parse().ok()?;
            // Alpha (optional)
            let a_val: f64 = if is_hsla {
                parts[3].trim().parse().ok()?
            } else {
                1.0
            };

            // Convert HSL to RGB
            let (r, g, b) = hsl_to_rgb(h, s_val / 100.0, l_val / 100.0);
            return Some(Color::from_rgba(r, g, b, (a_val * 255.0) as u8));
        }
    }

    // Named colors
    match s.to_lowercase().as_str() {
        "black" => Some(Color::from_rgba(0, 0, 0, 255)),
        "white" => Some(Color::from_rgba(255, 255, 255, 255)),
        "red" => Some(Color::from_rgba(255, 0, 0, 255)),
        "green" => Some(Color::from_rgba(0, 128, 0, 255)),
        "blue" => Some(Color::from_rgba(0, 0, 255, 255)),
        "yellow" => Some(Color::from_rgba(255, 255, 0, 255)),
        "cyan" => Some(Color::from_rgba(0, 255, 255, 255)),
        "magenta" => Some(Color::from_rgba(255, 0, 255, 255)),
        "gray" | "grey" => Some(Color::from_rgba(128, 128, 128, 255)),
        "orange" => Some(Color::from_rgba(255, 165, 0, 255)),
        "purple" => Some(Color::from_rgba(128, 0, 128, 255)),
        "pink" => Some(Color::from_rgba(255, 192, 203, 255)),
        "brown" => Some(Color::from_rgba(165, 42, 42, 255)),
        "lime" => Some(Color::from_rgba(0, 255, 0, 255)),
        "navy" => Some(Color::from_rgba(0, 0, 128, 255)),
        "teal" => Some(Color::from_rgba(0, 128, 128, 255)),
        "olive" => Some(Color::from_rgba(128, 128, 0, 255)),
        "maroon" => Some(Color::from_rgba(128, 0, 0, 255)),
        "silver" => Some(Color::from_rgba(192, 192, 192, 255)),
        "aqua" => Some(Color::from_rgba(0, 255, 255, 255)),
        "fuchsia" => Some(Color::from_rgba(255, 0, 255, 255)),
        "currentcolor" => Some(Color::from_rgba(0, 0, 0, 255)),
        _ => None,
    }
}

/// Parse paint value (color or gradient reference)
pub fn parse_paint(s: &str) -> Paint {
    let s = s.trim();

    if s == "none" {
        return Paint::None;
    }

    // Handle url() with optional quotes: url(#id), url('#id'), url("#id")
    if s.starts_with("url(") {
        // Extract content between url( and )
        let content = s.trim_start_matches("url(").trim_end_matches(')').trim();
        // Remove quotes if present
        let content = content.trim_matches('\'').trim_matches('"');
        // Check if it's a local reference
        if content.starts_with('#') {
            let id = content.trim_start_matches('#');
            return Paint::Gradient(id.to_string());
        }
    }

    if let Some(color) = parse_color(s) {
        return Paint::Color(color);
    }

    Paint::None
}

/// Parse marker URL reference (e.g., "url(#marker1)")
pub fn parse_marker_url(s: &str) -> Option<String> {
    let s = s.trim();
    if s == "none" {
        return None;
    }
    if s.starts_with("url(#") {
        let id = s.trim_start_matches("url(#").trim_end_matches(')');
        return Some(id.to_string());
    }
    None
}

/// Parse transform attribute
pub fn parse_transform(s: &str) -> Transform {
    let mut result = Transform::default();
    let s = s.trim();

    let mut remaining = s;
    while !remaining.is_empty() {
        remaining = remaining.trim_start();

        if remaining.starts_with("translate(") {
            let end = remaining.find(')').unwrap_or(remaining.len());
            let args = &remaining[10..end];
            let nums: Vec<f64> = args.split(|c| c == ',' || c == ' ')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if !nums.is_empty() {
                let tx = nums[0];
                let ty = if nums.len() >= 2 { nums[1] } else { 0.0 };
                result = result.multiply(&Transform::translate(tx, ty));
            }
            remaining = &remaining[(end + 1).min(remaining.len())..];
        } else if remaining.starts_with("scale(") {
            let end = remaining.find(')').unwrap_or(remaining.len());
            let args = &remaining[6..end];
            let nums: Vec<f64> = args.split(|c| c == ',' || c == ' ')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if !nums.is_empty() {
                let sx = nums[0];
                let sy = if nums.len() >= 2 { nums[1] } else { sx };
                result = result.multiply(&Transform::scale(sx, sy));
            }
            remaining = &remaining[(end + 1).min(remaining.len())..];
        } else if remaining.starts_with("rotate(") {
            let end = remaining.find(')').unwrap_or(remaining.len());
            let args = &remaining[7..end];
            let nums: Vec<f64> = args.split(|c| c == ',' || c == ' ')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if !nums.is_empty() {
                let angle = nums[0] * std::f64::consts::PI / 180.0;
                if nums.len() >= 3 {
                    let cx = nums[1];
                    let cy = nums[2];
                    result = result.multiply(&Transform::translate(cx, cy));
                    result = result.multiply(&Transform::rotate(angle));
                    result = result.multiply(&Transform::translate(-cx, -cy));
                } else {
                    result = result.multiply(&Transform::rotate(angle));
                }
            }
            remaining = &remaining[(end + 1).min(remaining.len())..];
        } else if remaining.starts_with("matrix(") {
            let end = remaining.find(')').unwrap_or(remaining.len());
            let args = &remaining[7..end];
            let nums: Vec<f64> = args.split(|c| c == ',' || c == ' ')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if nums.len() >= 6 {
                let t = Transform {
                    a: nums[0], b: nums[1], c: nums[2],
                    d: nums[3], e: nums[4], f: nums[5],
                };
                result = result.multiply(&t);
            }
            remaining = &remaining[(end + 1).min(remaining.len())..];
        } else if remaining.starts_with("skewX(") {
            let end = remaining.find(')').unwrap_or(remaining.len());
            let args = &remaining[6..end];
            if let Ok(angle) = args.trim().parse::<f64>() {
                let t = Transform {
                    a: 1.0, b: 0.0,
                    c: (angle * std::f64::consts::PI / 180.0).tan(),
                    d: 1.0, e: 0.0, f: 0.0,
                };
                result = result.multiply(&t);
            }
            remaining = &remaining[(end + 1).min(remaining.len())..];
        } else if remaining.starts_with("skewY(") {
            let end = remaining.find(')').unwrap_or(remaining.len());
            let args = &remaining[6..end];
            if let Ok(angle) = args.trim().parse::<f64>() {
                let t = Transform {
                    a: 1.0,
                    b: (angle * std::f64::consts::PI / 180.0).tan(),
                    c: 0.0, d: 1.0, e: 0.0, f: 0.0,
                };
                result = result.multiply(&t);
            }
            remaining = &remaining[(end + 1).min(remaining.len())..];
        } else {
            if let Some(pos) = remaining.find(')') {
                remaining = &remaining[(pos + 1)..];
            } else {
                break;
            }
        }
    }

    result
}

/// Represents parsed transform-origin values
#[derive(Clone, Copy, Debug)]
pub struct TransformOrigin {
    pub x: f64,      // 0.0-1.0 for percentage, or absolute value
    pub y: f64,
    pub x_percent: bool,
    pub y_percent: bool,
}

impl Default for TransformOrigin {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0, x_percent: true, y_percent: true }
    }
}

/// Parse transform-origin attribute
/// Returns (x_value, y_value, x_is_percent, y_is_percent)
pub fn parse_transform_origin(s: &str) -> TransformOrigin {
    let s = s.trim();
    if s.is_empty() {
        return TransformOrigin::default();
    }

    // Split by whitespace
    let parts: Vec<&str> = s.split_whitespace().collect();

    fn parse_keyword_or_value(v: &str, is_x: bool) -> (f64, bool) {
        match v.to_lowercase().as_str() {
            "center" => (0.5, true),
            "left" => (0.0, true),
            "right" => (1.0, true),
            "top" => (0.0, true),
            "bottom" => (1.0, true),
            _ => {
                if v.ends_with('%') {
                    let num = v.trim_end_matches('%').parse::<f64>().unwrap_or(if is_x { 0.0 } else { 0.0 });
                    (num / 100.0, true)
                } else {
                    // Parse as length
                    let val = parse_length(v, 0.0);
                    (val, false)
                }
            }
        }
    }

    let (x, x_pct, y, y_pct) = if parts.len() == 1 {
        // Single value: applies to X, Y defaults to center
        let (x, x_pct) = parse_keyword_or_value(parts[0], true);
        (x, x_pct, 0.5, true)
    } else {
        let (x, x_pct) = parse_keyword_or_value(parts[0], true);
        let (y, y_pct) = parse_keyword_or_value(parts[1], false);
        (x, x_pct, y, y_pct)
    };

    TransformOrigin { x, y, x_percent: x_pct, y_percent: y_pct }
}

/// Get element bounding box from its attributes
pub fn get_element_bbox(node: &Node) -> Option<(f64, f64, f64, f64)> {
    let tag = node.tag_name().name();

    match tag {
        "rect" => {
            let x = node.attribute("x").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let y = node.attribute("y").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let w = node.attribute("width").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let h = node.attribute("height").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            Some((x, y, w, h))
        }
        "circle" => {
            let cx = node.attribute("cx").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let cy = node.attribute("cy").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let r = node.attribute("r").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            Some((cx - r, cy - r, r * 2.0, r * 2.0))
        }
        "ellipse" => {
            let cx = node.attribute("cx").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let cy = node.attribute("cy").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let rx = node.attribute("rx").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let ry = node.attribute("ry").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            Some((cx - rx, cy - ry, rx * 2.0, ry * 2.0))
        }
        "line" => {
            let x1 = node.attribute("x1").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let y1 = node.attribute("y1").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let x2 = node.attribute("x2").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let y2 = node.attribute("y2").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let min_x = x1.min(x2);
            let min_y = y1.min(y2);
            Some((min_x, min_y, (x2 - x1).abs(), (y2 - y1).abs()))
        }
        "image" => {
            let x = node.attribute("x").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let y = node.attribute("y").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let w = node.attribute("width").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let h = node.attribute("height").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            Some((x, y, w, h))
        }
        "text" => {
            // Approximate: use x, y and assume default size
            let x = node.attribute("x").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let y = node.attribute("y").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            // Very rough approximation
            Some((x, y - 12.0, 100.0, 16.0))
        }
        "use" => {
            let x = node.attribute("x").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let y = node.attribute("y").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let w = node.attribute("width").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            let h = node.attribute("height").map(|s| parse_length(s, 0.0)).unwrap_or(0.0);
            if w > 0.0 && h > 0.0 {
                Some((x, y, w, h))
            } else {
                Some((x, y, 0.0, 0.0))
            }
        }
        "g" | "svg" => {
            // Groups: use viewBox if available
            if let Some(vb) = node.attribute("viewBox").and_then(parse_viewbox) {
                Some(vb)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Apply transform-origin to a transform
pub fn apply_transform_origin(transform: &Transform, origin: &TransformOrigin, bbox: Option<(f64, f64, f64, f64)>) -> Transform {
    if let Some((x, y, w, h)) = bbox {
        let ox = if origin.x_percent { x + origin.x * w } else { origin.x };
        let oy = if origin.y_percent { y + origin.y * h } else { origin.y };

        // translate(origin) * transform * translate(-origin)
        Transform::translate(ox, oy)
            .multiply(transform)
            .multiply(&Transform::translate(-ox, -oy))
    } else {
        // No bbox, just use the origin values as absolute
        let ox = origin.x;
        let oy = origin.y;
        Transform::translate(ox, oy)
            .multiply(transform)
            .multiply(&Transform::translate(-ox, -oy))
    }
}

/// Parse style from node attributes and style attribute
pub fn parse_style(node: &Node, parent_style: &Style) -> Style {
    let mut style = parent_style.clone();

    let mut apply_prop = |key: &str, val: &str| {
        match key {
            "fill" => style.fill = Some(parse_paint(val)),
            "stroke" => style.stroke = Some(parse_paint(val)),
            "stroke-width" => style.stroke_width = parse_length(val, 1.0),
            "fill-opacity" => style.fill_opacity = parse_opacity(val, 1.0),
            "stroke-opacity" => style.stroke_opacity = parse_opacity(val, 1.0),
            "opacity" => style.opacity = parse_opacity(val, 1.0),
            "display" => style.display = val != "none",
            "visibility" => style.visibility = val == "visible",
            "fill-rule" => {
                style.fill_rule = match val {
                    "evenodd" => FillRule::EvenOdd,
                    _ => FillRule::NonZero,
                };
            }
            "stroke-linecap" => {
                style.stroke_linecap = match val {
                    "round" => LineCap::Round,
                    "square" => LineCap::Square,
                    _ => LineCap::Butt,
                };
            }
            "stroke-linejoin" => {
                style.stroke_linejoin = match val {
                    "round" => LineJoin::Round,
                    "bevel" => LineJoin::Bevel,
                    _ => LineJoin::Miter,
                };
            }
            "stroke-miterlimit" => {
                if let Ok(m) = val.parse() {
                    style.stroke_miterlimit = m;
                }
            }
            "font-family" => style.font_family = val.trim_matches('\'').trim_matches('"').to_string(),
            "font-size" => style.font_size = parse_length(val, 12.0),
            "font-weight" => style.font_weight = match val {
                "bold" => 700,
                "normal" => 400,
                _ => val.parse().unwrap_or(400),
            },
            "font-style" => style.font_style = val.to_string(),
            "text-anchor" => style.text_anchor = val.to_string(),
            "marker-start" => style.marker_start = parse_marker_url(val),
            "marker-mid" => style.marker_mid = parse_marker_url(val),
            "marker-end" => style.marker_end = parse_marker_url(val),
            "marker" => {
                let m = parse_marker_url(val);
                style.marker_start = m.clone();
                style.marker_mid = m.clone();
                style.marker_end = m;
            }
            _ => {}
        }
    };

    // Parse style attribute
    if let Some(style_attr) = node.attribute("style") {
        for part in style_attr.split(';') {
            let part = part.trim();
            if part.is_empty() { continue; }
            if let Some(colon) = part.find(':') {
                let prop = part[..colon].trim();
                let val = part[colon + 1..].trim();
                apply_prop(prop, val);
            }
        }
    }

    // Parse individual attributes
    for attr in node.attributes() {
        apply_prop(attr.name(), attr.value());
    }

    style
}

/// Parse points attribute (for polygon and polyline)
pub fn parse_points(s: &str, transform: &Transform) -> Vec<(f64, f64)> {
    let nums: Vec<f64> = s.split(|c: char| c == ',' || c.is_whitespace())
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    nums.chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| transform.apply(c[0], c[1]))
        .collect()
}

/// Parse viewBox attribute
pub fn parse_viewbox(s: &str) -> Option<(f64, f64, f64, f64)> {
    let nums: Vec<f64> = s.split(|c: char| c == ',' || c.is_whitespace())
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if nums.len() >= 4 {
        Some((nums[0], nums[1], nums[2], nums[3]))
    } else {
        None
    }
}

/// Parse length value (with optional units)
pub fn parse_length(s: &str, default: f64) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return default;
    }

    let num_str = s.trim_end_matches(|c: char| c.is_alphabetic() || c == '%');
    num_str.parse().unwrap_or(default)
}

/// Parse length value with percent and unit support
/// reference_size is used when the value is a percentage
/// SVG uses 96 DPI for absolute unit conversion
pub fn parse_length_percent(s: &str, reference_size: f64, default: f64) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return default;
    }

    // Percent
    if s.ends_with('%') {
        let num_str = s.trim_end_matches('%');
        if let Ok(pct) = num_str.parse::<f64>() {
            return pct / 100.0 * reference_size;
        }
        return default;
    }

    // Absolute units (at 96 DPI)
    const PX_PER_IN: f64 = 96.0;
    const PX_PER_CM: f64 = PX_PER_IN / 2.54;
    const PX_PER_MM: f64 = PX_PER_IN / 25.4;
    const PX_PER_PT: f64 = PX_PER_IN / 72.0;
    const PX_PER_PC: f64 = PX_PER_IN / 6.0;
    const PX_PER_Q: f64 = PX_PER_MM / 4.0;

    if s.ends_with("mm") {
        let num_str = s.trim_end_matches("mm");
        if let Ok(val) = num_str.parse::<f64>() {
            return val * PX_PER_MM;
        }
    } else if s.ends_with("cm") {
        let num_str = s.trim_end_matches("cm");
        if let Ok(val) = num_str.parse::<f64>() {
            return val * PX_PER_CM;
        }
    } else if s.ends_with("in") {
        let num_str = s.trim_end_matches("in");
        if let Ok(val) = num_str.parse::<f64>() {
            return val * PX_PER_IN;
        }
    } else if s.ends_with("pt") {
        let num_str = s.trim_end_matches("pt");
        if let Ok(val) = num_str.parse::<f64>() {
            return val * PX_PER_PT;
        }
    } else if s.ends_with("pc") {
        let num_str = s.trim_end_matches("pc");
        if let Ok(val) = num_str.parse::<f64>() {
            return val * PX_PER_PC;
        }
    } else if s.ends_with('Q') || s.ends_with('q') {
        let num_str = s.trim_end_matches(|c| c == 'Q' || c == 'q');
        if let Ok(val) = num_str.parse::<f64>() {
            return val * PX_PER_Q;
        }
    } else if s.ends_with("px") {
        let num_str = s.trim_end_matches("px");
        if let Ok(val) = num_str.parse::<f64>() {
            return val;
        }
    }

    // Handle px and other units (strip unit suffix)
    let num_str = s.trim_end_matches(|c: char| c.is_alphabetic());
    num_str.parse().unwrap_or(default)
}

/// Parse radius value with percent and unit support
/// For r in circle, the reference is sqrt((width^2 + height^2)/2)
pub fn parse_radius_percent(s: &str, width: f64, height: f64, default: f64) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return default;
    }

    if s.ends_with('%') {
        let num_str = s.trim_end_matches('%');
        if let Ok(pct) = num_str.parse::<f64>() {
            // SVG spec: percentage of sqrt((width^2 + height^2)/2)
            let reference = ((width * width + height * height) / 2.0).sqrt();
            return pct / 100.0 * reference;
        }
        return default;
    }

    // Use parse_length_percent for unit handling (with 0 reference since not percent)
    parse_length_percent(s, 0.0, default)
}

/// Parse opacity value (can be 0.5 or 50%)
pub fn parse_opacity(s: &str, default: f64) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return default;
    }

    if s.ends_with('%') {
        let num_str = s.trim_end_matches('%');
        if let Ok(pct) = num_str.parse::<f64>() {
            return (pct / 100.0).clamp(0.0, 1.0);
        }
        return default;
    }

    s.parse().unwrap_or(default).min(1.0).max(0.0)
}

/// Decode a data: URL to raw bytes
pub fn decode_data_url(url: &str) -> Option<Vec<u8>> {
    let url = url.strip_prefix("data:")?;

    let comma_pos = url.find(',')?;
    let (metadata, data) = url.split_at(comma_pos);
    let data = &data[1..];

    if metadata.contains(";base64") {
        let clean_data: String = data.chars().filter(|c| !c.is_whitespace()).collect();
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.decode(&clean_data).ok()
    } else {
        None
    }
}
