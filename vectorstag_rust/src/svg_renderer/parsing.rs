//! Parsing functions for SVG attributes.

use roxmltree::Node;
use super::types::*;

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

    if s.starts_with("url(#") {
        let id = s.trim_start_matches("url(#").trim_end_matches(')');
        return Paint::Gradient(id.to_string());
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

/// Parse style from node attributes and style attribute
pub fn parse_style(node: &Node, parent_style: &Style) -> Style {
    let mut style = parent_style.clone();

    let mut apply_prop = |key: &str, val: &str| {
        match key {
            "fill" => style.fill = Some(parse_paint(val)),
            "stroke" => style.stroke = Some(parse_paint(val)),
            "stroke-width" => style.stroke_width = parse_length(val, 1.0),
            "fill-opacity" => style.fill_opacity = val.parse().unwrap_or(1.0),
            "stroke-opacity" => style.stroke_opacity = val.parse().unwrap_or(1.0),
            "opacity" => style.opacity = val.parse().unwrap_or(1.0),
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
