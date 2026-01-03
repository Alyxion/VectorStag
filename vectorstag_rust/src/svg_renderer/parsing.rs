//! Parsing functions for SVG attributes.

use roxmltree::Node;
use super::types::*;
pub use super::types::Transform;

pub const XLINK_NS: &str = "http://www.w3.org/1999/xlink";

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

    // Named colors (full CSS/SVG named color list)
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
        // Extended SVG/CSS named colors
        "aliceblue" => Some(Color::from_rgba(240, 248, 255, 255)),
        "antiquewhite" => Some(Color::from_rgba(250, 235, 215, 255)),
        "aquamarine" => Some(Color::from_rgba(127, 255, 212, 255)),
        "azure" => Some(Color::from_rgba(240, 255, 255, 255)),
        "beige" => Some(Color::from_rgba(245, 245, 220, 255)),
        "bisque" => Some(Color::from_rgba(255, 228, 196, 255)),
        "blanchedalmond" => Some(Color::from_rgba(255, 235, 205, 255)),
        "blueviolet" => Some(Color::from_rgba(138, 43, 226, 255)),
        "burlywood" => Some(Color::from_rgba(222, 184, 135, 255)),
        "cadetblue" => Some(Color::from_rgba(95, 158, 160, 255)),
        "chartreuse" => Some(Color::from_rgba(127, 255, 0, 255)),
        "chocolate" => Some(Color::from_rgba(210, 105, 30, 255)),
        "coral" => Some(Color::from_rgba(255, 127, 80, 255)),
        "cornflowerblue" => Some(Color::from_rgba(100, 149, 237, 255)),
        "cornsilk" => Some(Color::from_rgba(255, 248, 220, 255)),
        "crimson" => Some(Color::from_rgba(220, 20, 60, 255)),
        "darkblue" => Some(Color::from_rgba(0, 0, 139, 255)),
        "darkcyan" => Some(Color::from_rgba(0, 139, 139, 255)),
        "darkgoldenrod" => Some(Color::from_rgba(184, 134, 11, 255)),
        "darkgray" | "darkgrey" => Some(Color::from_rgba(169, 169, 169, 255)),
        "darkgreen" => Some(Color::from_rgba(0, 100, 0, 255)),
        "darkkhaki" => Some(Color::from_rgba(189, 183, 107, 255)),
        "darkmagenta" => Some(Color::from_rgba(139, 0, 139, 255)),
        "darkolivegreen" => Some(Color::from_rgba(85, 107, 47, 255)),
        "darkorange" => Some(Color::from_rgba(255, 140, 0, 255)),
        "darkorchid" => Some(Color::from_rgba(153, 50, 204, 255)),
        "darkred" => Some(Color::from_rgba(139, 0, 0, 255)),
        "darksalmon" => Some(Color::from_rgba(233, 150, 122, 255)),
        "darkseagreen" => Some(Color::from_rgba(143, 188, 143, 255)),
        "darkslateblue" => Some(Color::from_rgba(72, 61, 139, 255)),
        "darkslategray" | "darkslategrey" => Some(Color::from_rgba(47, 79, 79, 255)),
        "darkturquoise" => Some(Color::from_rgba(0, 206, 209, 255)),
        "darkviolet" => Some(Color::from_rgba(148, 0, 211, 255)),
        "deeppink" => Some(Color::from_rgba(255, 20, 147, 255)),
        "deepskyblue" => Some(Color::from_rgba(0, 191, 255, 255)),
        "dimgray" | "dimgrey" => Some(Color::from_rgba(105, 105, 105, 255)),
        "dodgerblue" => Some(Color::from_rgba(30, 144, 255, 255)),
        "firebrick" => Some(Color::from_rgba(178, 34, 34, 255)),
        "floralwhite" => Some(Color::from_rgba(255, 250, 240, 255)),
        "forestgreen" => Some(Color::from_rgba(34, 139, 34, 255)),
        "gainsboro" => Some(Color::from_rgba(220, 220, 220, 255)),
        "ghostwhite" => Some(Color::from_rgba(248, 248, 255, 255)),
        "gold" => Some(Color::from_rgba(255, 215, 0, 255)),
        "goldenrod" => Some(Color::from_rgba(218, 165, 32, 255)),
        "greenyellow" => Some(Color::from_rgba(173, 255, 47, 255)),
        "honeydew" => Some(Color::from_rgba(240, 255, 240, 255)),
        "hotpink" => Some(Color::from_rgba(255, 105, 180, 255)),
        "indianred" => Some(Color::from_rgba(205, 92, 92, 255)),
        "indigo" => Some(Color::from_rgba(75, 0, 130, 255)),
        "ivory" => Some(Color::from_rgba(255, 255, 240, 255)),
        "khaki" => Some(Color::from_rgba(240, 230, 140, 255)),
        "lavender" => Some(Color::from_rgba(230, 230, 250, 255)),
        "lavenderblush" => Some(Color::from_rgba(255, 240, 245, 255)),
        "lawngreen" => Some(Color::from_rgba(124, 252, 0, 255)),
        "lemonchiffon" => Some(Color::from_rgba(255, 250, 205, 255)),
        "lightblue" => Some(Color::from_rgba(173, 216, 230, 255)),
        "lightcoral" => Some(Color::from_rgba(240, 128, 128, 255)),
        "lightcyan" => Some(Color::from_rgba(224, 255, 255, 255)),
        "lightgoldenrodyellow" => Some(Color::from_rgba(250, 250, 210, 255)),
        "lightgray" | "lightgrey" => Some(Color::from_rgba(211, 211, 211, 255)),
        "lightgreen" => Some(Color::from_rgba(144, 238, 144, 255)),
        "lightpink" => Some(Color::from_rgba(255, 182, 193, 255)),
        "lightsalmon" => Some(Color::from_rgba(255, 160, 122, 255)),
        "lightseagreen" => Some(Color::from_rgba(32, 178, 170, 255)),
        "lightskyblue" => Some(Color::from_rgba(135, 206, 250, 255)),
        "lightslategray" | "lightslategrey" => Some(Color::from_rgba(119, 136, 153, 255)),
        "lightsteelblue" => Some(Color::from_rgba(176, 196, 222, 255)),
        "lightyellow" => Some(Color::from_rgba(255, 255, 224, 255)),
        "limegreen" => Some(Color::from_rgba(50, 205, 50, 255)),
        "linen" => Some(Color::from_rgba(250, 240, 230, 255)),
        "mediumaquamarine" => Some(Color::from_rgba(102, 205, 170, 255)),
        "mediumblue" => Some(Color::from_rgba(0, 0, 205, 255)),
        "mediumorchid" => Some(Color::from_rgba(186, 85, 211, 255)),
        "mediumpurple" => Some(Color::from_rgba(147, 112, 219, 255)),
        "mediumseagreen" => Some(Color::from_rgba(60, 179, 113, 255)),
        "mediumslateblue" => Some(Color::from_rgba(123, 104, 238, 255)),
        "mediumspringgreen" => Some(Color::from_rgba(0, 250, 154, 255)),
        "mediumturquoise" => Some(Color::from_rgba(72, 209, 204, 255)),
        "mediumvioletred" => Some(Color::from_rgba(199, 21, 133, 255)),
        "midnightblue" => Some(Color::from_rgba(25, 25, 112, 255)),
        "mintcream" => Some(Color::from_rgba(245, 255, 250, 255)),
        "mistyrose" => Some(Color::from_rgba(255, 228, 225, 255)),
        "moccasin" => Some(Color::from_rgba(255, 228, 181, 255)),
        "navajowhite" => Some(Color::from_rgba(255, 222, 173, 255)),
        "oldlace" => Some(Color::from_rgba(253, 245, 230, 255)),
        "olivedrab" => Some(Color::from_rgba(107, 142, 35, 255)),
        "orangered" => Some(Color::from_rgba(255, 69, 0, 255)),
        "orchid" => Some(Color::from_rgba(218, 112, 214, 255)),
        "palegoldenrod" => Some(Color::from_rgba(238, 232, 170, 255)),
        "palegreen" => Some(Color::from_rgba(152, 251, 152, 255)),
        "paleturquoise" => Some(Color::from_rgba(175, 238, 238, 255)),
        "palevioletred" => Some(Color::from_rgba(219, 112, 147, 255)),
        "papayawhip" => Some(Color::from_rgba(255, 239, 213, 255)),
        "peachpuff" => Some(Color::from_rgba(255, 218, 185, 255)),
        "peru" => Some(Color::from_rgba(205, 133, 63, 255)),
        "plum" => Some(Color::from_rgba(221, 160, 221, 255)),
        "powderblue" => Some(Color::from_rgba(176, 224, 230, 255)),
        "rebeccapurple" => Some(Color::from_rgba(102, 51, 153, 255)),
        "rosybrown" => Some(Color::from_rgba(188, 143, 143, 255)),
        "royalblue" => Some(Color::from_rgba(65, 105, 225, 255)),
        "saddlebrown" => Some(Color::from_rgba(139, 69, 19, 255)),
        "salmon" => Some(Color::from_rgba(250, 128, 114, 255)),
        "sandybrown" => Some(Color::from_rgba(244, 164, 96, 255)),
        "seagreen" => Some(Color::from_rgba(46, 139, 87, 255)),
        "seashell" => Some(Color::from_rgba(255, 245, 238, 255)),
        "sienna" => Some(Color::from_rgba(160, 82, 45, 255)),
        "skyblue" => Some(Color::from_rgba(135, 206, 235, 255)),
        "slateblue" => Some(Color::from_rgba(106, 90, 205, 255)),
        "slategray" | "slategrey" => Some(Color::from_rgba(112, 128, 144, 255)),
        "snow" => Some(Color::from_rgba(255, 250, 250, 255)),
        "springgreen" => Some(Color::from_rgba(0, 255, 127, 255)),
        "steelblue" => Some(Color::from_rgba(70, 130, 180, 255)),
        "tan" => Some(Color::from_rgba(210, 180, 140, 255)),
        "thistle" => Some(Color::from_rgba(216, 191, 216, 255)),
        "tomato" => Some(Color::from_rgba(255, 99, 71, 255)),
        "turquoise" => Some(Color::from_rgba(64, 224, 208, 255)),
        "violet" => Some(Color::from_rgba(238, 130, 238, 255)),
        "wheat" => Some(Color::from_rgba(245, 222, 179, 255)),
        "whitesmoke" => Some(Color::from_rgba(245, 245, 245, 255)),
        "yellowgreen" => Some(Color::from_rgba(154, 205, 50, 255)),
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

use crate::path::parse_path_internal;
use crate::path::PathCmd;

/// Get element bounding box from its attributes
pub fn get_element_bbox(node: &Node) -> Option<(f64, f64, f64, f64)> {
    let tag = node.tag_name().name();

    match tag {
        "path" => {
            if let Some(d) = node.attribute("d") {
                let cmds = parse_path_internal(d);
                let mut min_x = f64::INFINITY;
                let mut min_y = f64::INFINITY;
                let mut max_x = f64::NEG_INFINITY;
                let mut max_y = f64::NEG_INFINITY;
                let mut has_points = false;

                // Simple bbox estimation from points (not perfect for bezier but close enough for filters usually)
                // To be exact we should sample or calculate bezier bounds
                let mut update_bounds = |x: f64, y: f64| {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    has_points = true;
                };

                for cmd in cmds {
                    match cmd {
                        PathCmd::M(x, y) | PathCmd::L(x, y) => update_bounds(x, y),
                        PathCmd::C(x1, y1, x2, y2, x, y) => {
                            update_bounds(x1, y1);
                            update_bounds(x2, y2);
                            update_bounds(x, y);
                        }
                        PathCmd::Q(x1, y1, x, y) => {
                            update_bounds(x1, y1);
                            update_bounds(x, y);
                        }
                        PathCmd::A(_, _, _, _, _, x, y) => update_bounds(x, y),
                        _ => {}
                    }
                }

                if has_points {
                    Some((min_x, min_y, max_x - min_x, max_y - min_y))
                } else {
                    None
                }
            } else {
                None
            }
        }
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
