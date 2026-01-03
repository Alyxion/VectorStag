//! SVG filter collection and application.

use roxmltree::Node;
use ndarray::{Array3, ArrayView3};
use std::collections::HashMap;
use once_cell::sync::Lazy;
use super::types::*;
use super::parsing::parse_color;
use crate::filters;

// ============================================================================
// Static Lookup Tables for sRGB <-> Linear RGB conversion
// ============================================================================
// These LUTs are computed once and eliminate expensive powf() calls

/// sRGB u8 -> Linear RGB u8 lookup table
static SRGB_TO_LINEAR_LUT: Lazy<[u8; 256]> = Lazy::new(|| {
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let v = i as f32 / 255.0;
        let linear = if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        };
        lut[i] = (linear * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    lut
});

/// Linear RGB u8 -> sRGB u8 lookup table
static LINEAR_TO_SRGB_LUT: Lazy<[u8; 256]> = Lazy::new(|| {
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let v = i as f32 / 255.0;
        let srgb = if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        lut[i] = (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    lut
});

/// Collect all filter definitions from the document
pub fn collect_all_filters(ctx: &mut RenderContext, node: &Node) {
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();

    if tag == "filter" {
        collect_filter(ctx, node);
        return;
    }

    for child in node.children() {
        collect_all_filters(ctx, &child);
    }
}

/// Collect a single filter definition
fn collect_filter(ctx: &mut RenderContext, node: &Node) {
    let id = match node.attribute("id") {
        Some(id) => id,
        None => return,
    };

    fn parse_filter_length(s: Option<&str>, default_pct: f64) -> LengthVal {
        match s {
            Some(s) => {
                let s = s.trim();
                if s.ends_with('%') {
                    let val = s.trim_end_matches('%').parse::<f64>().unwrap_or(default_pct);
                    LengthVal { value: val, is_percent: true }
                } else {
                    let val = s.parse::<f64>().unwrap_or(default_pct / 100.0);
                    LengthVal { value: val, is_percent: false }
                }
            }
            None => LengthVal { value: default_pct, is_percent: true },
        }
    }

    let x = parse_filter_length(node.attribute("x"), -10.0);
    let y = parse_filter_length(node.attribute("y"), -10.0);
    let width = parse_filter_length(node.attribute("width"), 120.0);
    let height = parse_filter_length(node.attribute("height"), 120.0);

    let filter_units = node.attribute("filterUnits") != Some("userSpaceOnUse");
    // primitiveUnits default is userSpaceOnUse (true). objectBoundingBox is false.
    let primitive_units = node.attribute("primitiveUnits") != Some("objectBoundingBox");

    let mut primitives = Vec::new();

    for child in node.children() {
        if !child.is_element() {
            continue;
        }

        if let Some(prim) = parse_filter_primitive(&child) {
            primitives.push(prim);
        }
    }

    ctx.filters.insert(id.to_string(), FilterDef {
        id: id.to_string(),
        primitives,
        x,
        y,
        width,
        height,
        filter_units,
        primitive_units,
    });
}

/// Get color interpolation from node or ancestors
fn get_color_interpolation(node: &Node) -> ColorInterpolation {
    let mut current = Some(*node);
    while let Some(n) = current {
        if n.is_element() {
            if let Some(val) = n.attribute("color-interpolation-filters") {
                return match val {
                    "sRGB" => ColorInterpolation::SRGB,
                    "linearRGB" => ColorInterpolation::LinearRGB,
                    "auto" => ColorInterpolation::LinearRGB,
                    _ => ColorInterpolation::LinearRGB,
                };
            }
        }
        current = n.parent();
    }
    ColorInterpolation::LinearRGB // Default per SVG spec
}

/// Parse a filter primitive element
fn parse_filter_primitive(node: &Node) -> Option<FilterPrimitive> {
    let tag = node.tag_name().name();
    // Default 'in' to empty string to indicate "previous result"
    let input = node.attribute("in").unwrap_or("").to_string();
    let result = node.attribute("result").unwrap_or("").to_string();
    let color_interpolation = get_color_interpolation(node);

    match tag {
        "feGaussianBlur" => {
            let std_dev = node.attribute("stdDeviation").unwrap_or("0");
            let parts: Vec<f64> = std_dev
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter_map(|s| s.parse().ok())
                .collect();
            let std_dev_x = parts.first().copied().unwrap_or(0.0);
            let std_dev_y = parts.get(1).copied().unwrap_or(std_dev_x);
            Some(FilterPrimitive::GaussianBlur {
                std_dev_x,
                std_dev_y,
                input,
                result,
                color_interpolation,
            })
        }
        "feOffset" => {
            let dx = node.attribute("dx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let dy = node.attribute("dy").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            Some(FilterPrimitive::Offset { dx, dy, input, result, color_interpolation })
        }
        "feFlood" => {
            let color_str = node.attribute("flood-color").unwrap_or("black");
            let opacity: f64 = node.attribute("flood-opacity")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let mut color = parse_color(color_str).unwrap_or(Color::from_rgba(0, 0, 0, 255));
            color.a = (color.a as f64 * opacity) as u8;
            Some(FilterPrimitive::Flood { color, result, color_interpolation })
        }
        "feMerge" => {
            let mut nodes = Vec::new();
            for child in node.children() {
                if child.is_element() && child.tag_name().name() == "feMergeNode" {
                    // Default to empty (previous result) if 'in' is missing
                    let in_ref = child.attribute("in").unwrap_or("").to_string();
                    nodes.push(in_ref);
                }
            }
            Some(FilterPrimitive::Merge { nodes, result, color_interpolation })
        }
        "feColorMatrix" => {
            let matrix_type = match node.attribute("type").unwrap_or("matrix") {
                "matrix" => 0,
                "saturate" => 1,
                "hueRotate" => 2,
                "luminanceToAlpha" => 3,
                _ => 0,
            };
            let values: Vec<f32> = node.attribute("values")
                .unwrap_or("")
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter_map(|s| s.parse().ok())
                .collect();
            Some(FilterPrimitive::ColorMatrix { matrix_type, values, input, result, color_interpolation })
        }
        "feBlend" => {
            let mode = match node.attribute("mode").unwrap_or("normal") {
                "normal" => 0,
                "multiply" => 1,
                "screen" => 2,
                "darken" => 3,
                "lighten" => 4,
                "overlay" => 5,
                "color-dodge" => 6,
                "color-burn" => 7,
                "hard-light" => 8,
                "soft-light" => 9,
                "difference" => 10,
                "exclusion" => 11,
                "hue" => 12,
                "saturation" => 13,
                "color" => 14,
                "luminosity" => 15,
                _ => 0,
            };
            let in1 = node.attribute("in").unwrap_or("").to_string();
            let in2 = node.attribute("in2").unwrap_or("BackgroundImage").to_string();
            Some(FilterPrimitive::Blend { mode, in1, in2, result, color_interpolation })
        }
        "feComposite" => {
            let operator = match node.attribute("operator").unwrap_or("over") {
                "over" => 0,
                "in" => 1,
                "out" => 2,
                "atop" => 3,
                "xor" => 4,
                "arithmetic" => 5,
                _ => 0,
            };
            let k1: f32 = node.attribute("k1").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let k2: f32 = node.attribute("k2").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let k3: f32 = node.attribute("k3").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let k4: f32 = node.attribute("k4").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let in1 = node.attribute("in").unwrap_or("").to_string();
            let in2 = node.attribute("in2").unwrap_or("BackgroundImage").to_string();
            Some(FilterPrimitive::Composite { operator, k1, k2, k3, k4, in1, in2, result, color_interpolation })
        }
        "feMorphology" => {
            let operator = match node.attribute("operator").unwrap_or("erode") {
                "erode" => 0,
                "dilate" => 1,
                _ => 0,
            };
            let radius_str = node.attribute("radius").unwrap_or("0");
            let parts: Vec<f64> = radius_str
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter_map(|s| s.parse().ok())
                .collect();
            let radius_x = parts.first().copied().unwrap_or(0.0);
            let radius_y = parts.get(1).copied().unwrap_or(radius_x);
            Some(FilterPrimitive::Morphology { operator, radius_x, radius_y, input, result, color_interpolation })
        }
        "feTurbulence" => {
            let freq_str = node.attribute("baseFrequency").unwrap_or("0");
            let parts: Vec<f64> = freq_str
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter_map(|s| s.parse().ok())
                .collect();
            let base_freq_x = parts.first().copied().unwrap_or(0.0);
            let base_freq_y = parts.get(1).copied().unwrap_or(base_freq_x);
            let num_octaves: usize = node.attribute("numOctaves")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            let seed: i32 = node.attribute("seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let noise_type = match node.attribute("type").unwrap_or("turbulence") {
                "fractalNoise" => 0,
                "turbulence" => 1,
                _ => 1,
            };
            Some(FilterPrimitive::Turbulence { base_freq_x, base_freq_y, num_octaves, seed, noise_type, result, color_interpolation })
        }
        "feTile" => {
            Some(FilterPrimitive::Tile { input, result, color_interpolation })
        }
        "feDropShadow" => {
            let dx: f64 = node.attribute("dx").and_then(|s| s.parse().ok()).unwrap_or(2.0);
            let dy: f64 = node.attribute("dy").and_then(|s| s.parse().ok()).unwrap_or(2.0);
            let std_dev = node.attribute("stdDeviation").unwrap_or("2");
            let parts: Vec<f64> = std_dev
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter_map(|s| s.parse().ok())
                .collect();
            let std_dev_x = parts.first().copied().unwrap_or(2.0);
            let std_dev_y = parts.get(1).copied().unwrap_or(std_dev_x);
            let color_str = node.attribute("flood-color").unwrap_or("black");
            let opacity: f64 = node.attribute("flood-opacity")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let mut flood_color = parse_color(color_str).unwrap_or(Color::from_rgba(0, 0, 0, 255));
            flood_color.a = (flood_color.a as f64 * opacity) as u8;
            Some(FilterPrimitive::DropShadow { dx, dy, std_dev_x, std_dev_y, flood_color, input, result, color_interpolation })
        }
        "feComponentTransfer" => {
            let func_r = parse_component_transfer_func(node, "feFuncR");
            let func_g = parse_component_transfer_func(node, "feFuncG");
            let func_b = parse_component_transfer_func(node, "feFuncB");
            let func_a = parse_component_transfer_func(node, "feFuncA");
            Some(FilterPrimitive::ComponentTransfer { func_r, func_g, func_b, func_a, input, result, color_interpolation })
        }
        "feConvolveMatrix" => {
            let order = node.attribute("order").unwrap_or("3");
            let parts: Vec<usize> = order
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter_map(|s| s.parse().ok())
                .collect();
            let order_x = parts.first().copied().unwrap_or(3);
            let order_y = parts.get(1).copied().unwrap_or(order_x);
            let kernel: Vec<f32> = node.attribute("kernelMatrix")
                .unwrap_or("")
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter_map(|s| s.parse().ok())
                .collect();
            let divisor: f32 = node.attribute("divisor")
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| kernel.iter().sum::<f32>().max(1.0));
            let bias: f32 = node.attribute("bias").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let target_x: usize = node.attribute("targetX")
                .and_then(|s| s.parse().ok())
                .unwrap_or(order_x / 2);
            let target_y: usize = node.attribute("targetY")
                .and_then(|s| s.parse().ok())
                .unwrap_or(order_y / 2);
            let edge_mode = match node.attribute("edgeMode").unwrap_or("duplicate") {
                "duplicate" => 0,
                "wrap" => 1,
                "none" => 2,
                _ => 0,
            };
            let preserve_alpha = node.attribute("preserveAlpha") == Some("true");
            Some(FilterPrimitive::ConvolveMatrix {
                order_x, order_y, kernel, divisor, bias, target_x, target_y, edge_mode, preserve_alpha, input, result, color_interpolation
            })
        }
        "feDiffuseLighting" => {
            let surface_scale: f32 = node.attribute("surfaceScale").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let diffuse_constant: f32 = node.attribute("diffuseConstant").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let color_str = node.attribute("lighting-color").unwrap_or("white");
            let color = parse_color(color_str).unwrap_or(Color::from_rgba(255, 255, 255, 255));
            let (light_type, azimuth, elevation, light_x, light_y, light_z, points_at_x, points_at_y, points_at_z, specular_exponent, limiting_cone_angle) = parse_light_source(node);
            Some(FilterPrimitive::DiffuseLighting {
                surface_scale, diffuse_constant,
                light_color: (color.r, color.g, color.b),
                light_type, azimuth, elevation, light_x, light_y, light_z,
                points_at_x, points_at_y, points_at_z, specular_exponent, limiting_cone_angle,
                input, result, color_interpolation
            })
        }
        "feSpecularLighting" => {
            let surface_scale: f32 = node.attribute("surfaceScale").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let specular_constant: f32 = node.attribute("specularConstant").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let specular_exponent: f32 = node.attribute("specularExponent").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let color_str = node.attribute("lighting-color").unwrap_or("white");
            let color = parse_color(color_str).unwrap_or(Color::from_rgba(255, 255, 255, 255));
            let (light_type, azimuth, elevation, light_x, light_y, light_z, points_at_x, points_at_y, points_at_z, spot_exponent, limiting_cone_angle) = parse_light_source(node);
            Some(FilterPrimitive::SpecularLighting {
                surface_scale, specular_constant, specular_exponent,
                light_color: (color.r, color.g, color.b),
                light_type, azimuth, elevation, light_x, light_y, light_z,
                points_at_x, points_at_y, points_at_z, spot_exponent, limiting_cone_angle,
                input, result, color_interpolation
            })
        }
        "feDisplacementMap" => {
            let scale: f32 = node.attribute("scale").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let x_channel = match node.attribute("xChannelSelector").unwrap_or("A") {
                "R" => 0, "G" => 1, "B" => 2, "A" => 3, _ => 3,
            };
            let y_channel = match node.attribute("yChannelSelector").unwrap_or("A") {
                "R" => 0, "G" => 1, "B" => 2, "A" => 3, _ => 3,
            };
            let in1 = node.attribute("in").unwrap_or("").to_string();
            let in2 = node.attribute("in2").unwrap_or("").to_string();
            Some(FilterPrimitive::DisplacementMap { scale, x_channel, y_channel, in1, in2, result, color_interpolation })
        }
        "feImage" => {
            let href = node.attribute("href").or_else(|| node.attribute((crate::svg_renderer::parsing::XLINK_NS, "href"))).unwrap_or("").to_string();
            Some(FilterPrimitive::Image { result, href, color_interpolation })
        }
        _ => None,
    }
}

/// Parse component transfer function for feComponentTransfer
fn parse_component_transfer_func(parent: &Node, func_name: &str) -> (u8, Vec<f32>, f32, f32, f32, f32, f32) {
    for child in parent.children() {
        if child.is_element() && child.tag_name().name() == func_name {
            let func_type = match child.attribute("type").unwrap_or("identity") {
                "identity" => 0,
                "table" => 1,
                "discrete" => 2,
                "linear" => 3,
                "gamma" => 4,
                _ => 0,
            };
            let table: Vec<f32> = child.attribute("tableValues")
                .unwrap_or("")
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter_map(|s| s.parse().ok())
                .collect();
            let slope: f32 = child.attribute("slope").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let intercept: f32 = child.attribute("intercept").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let amplitude: f32 = child.attribute("amplitude").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let exponent: f32 = child.attribute("exponent").and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let offset: f32 = child.attribute("offset").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            return (func_type, table, slope, intercept, amplitude, exponent, offset);
        }
    }
    // Identity function (default)
    (0, Vec::new(), 1.0, 0.0, 1.0, 1.0, 0.0)
}

/// Parse light source from feDiffuseLighting or feSpecularLighting
fn parse_light_source(parent: &Node) -> (u8, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    for child in parent.children() {
        if !child.is_element() {
            continue;
        }
        match child.tag_name().name() {
            "feDistantLight" => {
                let azimuth: f32 = child.attribute("azimuth").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let elevation: f32 = child.attribute("elevation").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                return (0, azimuth, elevation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0);
            }
            "fePointLight" => {
                let x: f32 = child.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = child.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = child.attribute("z").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                return (1, 0.0, 0.0, x, y, z, 0.0, 0.0, 0.0, 1.0, 90.0);
            }
            "feSpotLight" => {
                let x: f32 = child.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = child.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = child.attribute("z").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let points_at_x: f32 = child.attribute("pointsAtX").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let points_at_y: f32 = child.attribute("pointsAtY").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let points_at_z: f32 = child.attribute("pointsAtZ").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let specular_exponent: f32 = child.attribute("specularExponent").and_then(|s| s.parse().ok()).unwrap_or(1.0);
                let limiting_cone_angle: f32 = child.attribute("limitingConeAngle").and_then(|s| s.parse().ok()).unwrap_or(90.0);
                return (2, 0.0, 0.0, x, y, z, points_at_x, points_at_y, points_at_z, specular_exponent, limiting_cone_angle);
            }
            _ => {}
        }
    }
    // Default: distant light from straight above
    (0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0)
}

/// Apply a filter to a source buffer
pub fn apply_filter(
    filter: &FilterDef,
    source: &[u8],
    background: &[u8],
    width: usize,
    height: usize,
    transform: &Transform,
    bbox: Option<(f64, f64, f64, f64)>,
) -> Vec<u8> {
    if filter.primitives.is_empty() {
        return source.to_vec();
    }

    // Calculate scale factors from transform
    let scale_x = (transform.a * transform.a + transform.b * transform.b).sqrt();
    let scale_y = (transform.c * transform.c + transform.d * transform.d).sqrt();
    let scale = (scale_x + scale_y) / 2.0;

    // Calculate filter region in screen pixels
    let (region_min_x, region_min_y, region_max_x, region_max_y) = calculate_filter_region(filter, bbox, transform, width, height);

    // Convert source to ndarray for filter operations (Premultiplied sRGB)
    // We store intermediate results in Premultiplied sRGB to maximize precision in u8
    let source_arr = Array3::from_shape_fn((height, width, 4), |(y, x, c)| {
        if c == 3 {
            source[(y * width + x) * 4 + 3]
        } else {
            let a = source[(y * width + x) * 4 + 3] as f32 / 255.0;
            let val = source[(y * width + x) * 4 + c] as f32;
            if a > 0.0 {
                (val * a + 0.5) as u8
            } else {
                0
            }
        }
    });

    // Convert background to ndarray (Premultiplied sRGB)
    let bg_arr = Array3::from_shape_fn((height, width, 4), |(y, x, c)| {
        if c == 3 {
            background[(y * width + x) * 4 + 3]
        } else {
            let a = background[(y * width + x) * 4 + 3] as f32 / 255.0;
            let val = background[(y * width + x) * 4 + c] as f32;
            if a > 0.0 {
                (val * a + 0.5) as u8
            } else {
                0
            }
        }
    });

    // Source alpha for SourceAlpha input
    let source_alpha = filters::get_source_alpha_impl(&source_arr.view());

    // Results map for named results
    let mut results: HashMap<String, Array3<u8>> = HashMap::new();
    results.insert("SourceGraphic".to_string(), source_arr.clone());
    results.insert("SourceAlpha".to_string(), source_alpha);
    results.insert("BackgroundImage".to_string(), bg_arr);

    // Last result for implicit chaining
    let mut last_result = source_arr;

    // Calculate scale factors for primitiveUnits
    let (prim_scale_x, prim_scale_y) = if filter.primitive_units {
        // userSpaceOnUse: scale by transform scale
        (scale_x, scale_y)
    } else {
        // objectBoundingBox: scale by bbox size * transform scale
        if let Some((_, _, w, h)) = bbox {
            (w * scale_x, h * scale_y)
        } else {
            (scale_x, scale_y)
        }
    };

    // Helper: Convert Premultiplied sRGB -> Premultiplied LinearRGB
    let to_linear_premul = |mut arr: Array3<u8>| -> Array3<u8> {
        unpremultiply(&mut arr);
        let mut lin = array_srgb_to_linear(&arr.view());
        premultiply(&mut lin);
        lin
    };

    // Helper: Convert Premultiplied LinearRGB -> Premultiplied sRGB
    let to_srgb_premul = |mut arr: Array3<u8>| -> Array3<u8> {
        unpremultiply(&mut arr);
        let mut srgb = array_linear_to_srgb(&arr);
        premultiply(&mut srgb);
        srgb
    };

    // Apply each primitive in sequence
    for prim in &filter.primitives {
        let interpolation = get_prim_color_interpolation(prim);
        
        let convert_input = |input_name: &str, ensure_premultiplied: bool| -> Array3<u8> {
            let view = get_input(input_name, &results, &last_result);
            let mut arr = view.to_owned(); // Always Premultiplied sRGB from storage
            
            if interpolation == ColorInterpolation::LinearRGB {
                arr = to_linear_premul(arr);
            }
            
            if !ensure_premultiplied {
                unpremultiply(&mut arr);
            }
            arr
        };
        
        // Track if the result is premultiplied
        let (mut result, is_premultiplied) = match prim {
            FilterPrimitive::GaussianBlur { std_dev_x, std_dev_y, input, .. } => {
                let src = convert_input(input, true);
                let scaled_x = (*std_dev_x as f64 * scale_x) as f32;
                let scaled_y = (*std_dev_y as f64 * scale_y) as f32;
                (filters::fe_gaussian_blur_impl(&src.view(), scaled_x, scaled_y), true)
            }
            FilterPrimitive::Offset { dx, dy, input, .. } => {
                let src = convert_input(input, true);
                let scaled_dx = (*dx * scale_x) as i32;
                let scaled_dy = (*dy * scale_y) as i32;
                (filters::fe_offset_impl(&src.view(), scaled_dx, scaled_dy), true)
            }
            FilterPrimitive::Flood { color, .. } => {
                // Flood color is typically sRGB. Convert if needed.
                let (r, g, b) = if interpolation == ColorInterpolation::LinearRGB {
                    (srgb_to_linear(color.r), srgb_to_linear(color.g), srgb_to_linear(color.b))
                } else {
                    (color.r, color.g, color.b)
                };
                // Premultiply
                let a = color.a as f32 / 255.0;
                let r = (r as f32 * a + 0.5) as u8;
                let g = (g as f32 * a + 0.5) as u8;
                let b = (b as f32 * a + 0.5) as u8;
                (filters::fe_flood_impl(width, height, r, g, b, color.a), true)
            }
            FilterPrimitive::Merge { nodes, .. } => {
                let layers: Vec<Array3<u8>> = nodes.iter().map(|n| convert_input(n, true)).collect();
                let views: Vec<ArrayView3<u8>> = layers.iter().map(|l| l.view()).collect();
                (filters::fe_merge_impl(&views), true)
            }
            FilterPrimitive::ColorMatrix { matrix_type, values, input, .. } => {
                // Spec: "If the input graphic is premultiplied, the matrix operation is applied to the premultiplied components."
                let src = convert_input(input, true);
                (filters::fe_color_matrix_impl(&src.view(), *matrix_type, values), true)
            }
            FilterPrimitive::Blend { mode, in1, in2, .. } => {
                let src1 = convert_input(in1, true);
                let src2 = convert_input(in2, true);
                (filters::fe_blend_impl(&src1.view(), &src2.view(), *mode), true)
            }
            FilterPrimitive::Composite { operator, k1, k2, k3, k4, in1, in2, .. } => {
                let src1 = convert_input(in1, true);
                let src2 = convert_input(in2, true);
                (filters::fe_composite_impl(&src1.view(), &src2.view(), *operator, *k1, *k2, *k3, *k4), true)
            }
            FilterPrimitive::Morphology { operator, radius_x, radius_y, input, .. } => {
                let src = convert_input(input, true);
                let scaled_rx = (*radius_x as f64 * scale_x) as f32;
                let scaled_ry = (*radius_y as f64 * scale_y) as f32;
                (filters::fe_morphology_impl(&src.view(), *operator, scaled_rx, scaled_ry), true)
            }
            FilterPrimitive::Turbulence { base_freq_x, base_freq_y, num_octaves, seed, noise_type, .. } => {
                let freq_x = if scale_x > 0.0 { *base_freq_x / scale_x } else { *base_freq_x };
                let freq_y = if scale_y > 0.0 { *base_freq_y / scale_y } else { *base_freq_y };
                (filters::fe_turbulence_impl(width, height, freq_x, freq_y, *num_octaves, *seed, *noise_type, false), false)
            }
            FilterPrimitive::Tile { input, .. } => {
                let src = convert_input(input, true);
                (filters::fe_tile_impl(&src.view(), width, height), true)
            }
            FilterPrimitive::ComponentTransfer { func_r, func_g, func_b, func_a, input, .. } => {
                // Spec 15.11: "The transfer functions are defined to operate on unpremultiplied color values."
                // So we must provide Unpremultiplied input.
                let src = convert_input(input, false);
                (filters::fe_component_transfer_impl(&src.view(), func_r, func_g, func_b, func_a), false)
            }
            FilterPrimitive::ConvolveMatrix { order_x, order_y, kernel, divisor, bias, target_x, target_y, edge_mode, preserve_alpha, input, .. } => {
                let src = convert_input(input, true);
                (filters::fe_convolve_matrix_impl(&src.view(), *order_x, *order_y, kernel, *divisor, *bias, *target_x, *target_y, *edge_mode, *preserve_alpha), true)
            }
            FilterPrimitive::DiffuseLighting { surface_scale, diffuse_constant, light_color, light_type, azimuth, elevation, light_x, light_y, light_z, points_at_x, points_at_y, points_at_z, specular_exponent, limiting_cone_angle, input, .. } => {
                // Lighting inputs: Source Alpha (opaque)
                // Lighting uses Unpremultiplied usually for alpha map? No, it uses alpha channel. Premul or not, Alpha is same.
                let src = convert_input(input, true);
                
                let (r, g, b) = if interpolation == ColorInterpolation::LinearRGB {
                    (srgb_to_linear(light_color.0), srgb_to_linear(light_color.1), srgb_to_linear(light_color.2))
                } else {
                    (light_color.0, light_color.1, light_color.2)
                };
                let lc = (r, g, b);
                (filters::fe_diffuse_lighting_impl(&src.view(), *surface_scale, *diffuse_constant, lc, *light_type, *azimuth, *elevation, *light_x, *light_y, *light_z, *points_at_x, *points_at_y, *points_at_z, *specular_exponent, *limiting_cone_angle), false)
            }
            FilterPrimitive::SpecularLighting { surface_scale, specular_constant, specular_exponent, light_color, light_type, azimuth, elevation, light_x, light_y, light_z, points_at_x, points_at_y, points_at_z, spot_exponent, limiting_cone_angle, input, .. } => {
                let src = convert_input(input, true);
                let (r, g, b) = if interpolation == ColorInterpolation::LinearRGB {
                    (srgb_to_linear(light_color.0), srgb_to_linear(light_color.1), srgb_to_linear(light_color.2))
                } else {
                    (light_color.0, light_color.1, light_color.2)
                };
                let lc = (r, g, b);
                (filters::fe_specular_lighting_impl(&src.view(), *surface_scale, *specular_constant, *specular_exponent, lc, *light_type, *azimuth, *elevation, *light_x, *light_y, *light_z, *points_at_x, *points_at_y, *points_at_z, *spot_exponent, *limiting_cone_angle), false)
            }
            FilterPrimitive::DisplacementMap { scale: disp_scale, x_channel, y_channel, in1, in2, .. } => {
                let src = convert_input(in1, true);
                // Map is often Unpremultiplied? But we standard on premul. Map channels (RGBA) are used directly.
                // Using Premultiplied map means color channels are scaled by alpha.
                // Spec doesn't strictly specify un/premul for map input, but browsers likely use premultiplied.
                let map = convert_input(in2, true);
                let avg_scale = (scale_x + scale_y) / 2.0;
                let scaled_disp = *disp_scale * avg_scale as f32;
                (filters::fe_displacement_map_impl(&src.view(), &map.view(), scaled_disp, *x_channel, *y_channel), true)
            }
            FilterPrimitive::DropShadow { dx, dy, std_dev_x, std_dev_y, flood_color, input, .. } => {
                let src = convert_input(input, true);
                let scaled_dx = (*dx * scale_x) as f32;
                let scaled_dy = (*dy * scale_y) as f32;
                let scaled_std_x = (*std_dev_x as f64 * scale_x) as f32;
                let scaled_std_y = (*std_dev_y as f64 * scale_y) as f32;
                let (r, g, b) = if interpolation == ColorInterpolation::LinearRGB {
                    (srgb_to_linear(flood_color.r), srgb_to_linear(flood_color.g), srgb_to_linear(flood_color.b))
                } else {
                    (flood_color.r, flood_color.g, flood_color.b)
                };
                (filters::fe_drop_shadow_impl(&src.view(), scaled_dx, scaled_dy, scaled_std_x, scaled_std_y, r, g, b, flood_color.a), true)
            }
            FilterPrimitive::Image { .. } => {
                (Array3::<u8>::zeros((height, width, 4)), false)
            }
        };

        // If result is NOT premultiplied, premultiply it for storage
        if !is_premultiplied {
            premultiply(&mut result);
        }
        
        // Convert back to Premultiplied sRGB if we were in Linear
        if interpolation == ColorInterpolation::LinearRGB {
            result = to_srgb_premul(result);
        }

        // Clip to filter region
        clip_buffer(&mut result, region_min_x, region_min_y, region_max_x, region_max_y);

        // Store result if named
        let result_name = get_result_name(prim);
        if !result_name.is_empty() {
            results.insert(result_name.to_string(), result.clone());
        }

        last_result = result;
    }

    // Convert final result back to Vec<u8> (Unpremultiplied sRGB)
    // last_result is Premultiplied sRGB
    let mut output = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let a = last_result[[y, x, 3]] as f32 / 255.0;
            if a > 0.0 {
                // Unpremultiply
                output[idx] = (last_result[[y, x, 0]] as f32 / a + 0.5).clamp(0.0, 255.0) as u8;
                output[idx + 1] = (last_result[[y, x, 1]] as f32 / a + 0.5).clamp(0.0, 255.0) as u8;
                output[idx + 2] = (last_result[[y, x, 2]] as f32 / a + 0.5).clamp(0.0, 255.0) as u8;
            } else {
                output[idx] = 0;
                output[idx + 1] = 0;
                output[idx + 2] = 0;
            }
            output[idx + 3] = last_result[[y, x, 3]];
        }
    }

    output
}

/// Calculate filter region in screen coordinates (min_x, min_y, max_x, max_y)
fn calculate_filter_region(
    filter: &FilterDef,
    bbox: Option<(f64, f64, f64, f64)>,
    transform: &Transform,
    img_width: usize,
    img_height: usize,
) -> (i32, i32, i32, i32) {
    let (x, y, w, h) = if filter.filter_units {
        // objectBoundingBox
        if let Some((bx, by, bw, bh)) = bbox {
            
            let fx = if filter.x.is_percent { bx + (filter.x.value / 100.0) * bw } else { bx + filter.x.value * bw };
            let fy = if filter.y.is_percent { by + (filter.y.value / 100.0) * bh } else { by + filter.y.value * bh };
            let fw = if filter.width.is_percent { (filter.width.value / 100.0) * bw } else { filter.width.value * bw };
            let fh = if filter.height.is_percent { (filter.height.value / 100.0) * bh } else { filter.height.value * bh };
            
            (fx, fy, fw, fh)
        } else {
            // Fallback if no bbox
            (0.0, 0.0, img_width as f64, img_height as f64)
        }
    } else {
        // userSpaceOnUse
        // Wait, if it IS percent in userSpaceOnUse, I need viewport.
        // For now let's treat as absolute if not percent, or fail gracefully.
        (filter.x.value, filter.y.value, filter.width.value, filter.height.value)
    };

    // Transform region to screen space
    let (x1, y1) = transform.apply(x, y);
    let (x2, y2) = transform.apply(x + w, y);
    let (x3, y3) = transform.apply(x + w, y + h);
    let (x4, y4) = transform.apply(x, y + h);

    let min_x = x1.min(x2).min(x3).min(x4).floor() as i32;
    let max_x = x1.max(x2).max(x3).max(x4).ceil() as i32;
    let min_y = y1.min(y2).min(y3).min(y4).floor() as i32;
    let max_y = y1.max(y2).max(y3).max(y4).ceil() as i32;

    // Pad slightly to avoid edge artifacts?
    (min_x, min_y, max_x, max_y)
}

/// Clip buffer to region
fn clip_buffer(buffer: &mut Array3<u8>, min_x: i32, min_y: i32, max_x: i32, max_y: i32) {
    let (h, w, _) = (buffer.shape()[0], buffer.shape()[1], buffer.shape()[2]);
    let h = h as i32;
    let w = w as i32;

    if min_x <= 0 && min_y <= 0 && max_x >= w && max_y >= h {
        return;
    }

    // Optimization: iterate over outside regions?
    // Or just iterate everything if region is small?
    // Let's just iterate everything for safety now.
    for y in 0..h {
        if y < min_y || y >= max_y {
            for x in 0..w {
                buffer[[y as usize, x as usize, 0]] = 0;
                buffer[[y as usize, x as usize, 1]] = 0;
                buffer[[y as usize, x as usize, 2]] = 0;
                buffer[[y as usize, x as usize, 3]] = 0;
            }
        } else {
            for x in 0..min_x {
                buffer[[y as usize, x as usize, 0]] = 0;
                buffer[[y as usize, x as usize, 1]] = 0;
                buffer[[y as usize, x as usize, 2]] = 0;
                buffer[[y as usize, x as usize, 3]] = 0;
            }
            for x in max_x..w {
                buffer[[y as usize, x as usize, 0]] = 0;
                buffer[[y as usize, x as usize, 1]] = 0;
                buffer[[y as usize, x as usize, 2]] = 0;
                buffer[[y as usize, x as usize, 3]] = 0;
            }
        }
    }
}

/// Convert sRGB f32 to Linear RGB f32
fn srgb_to_linear_f32(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert Linear RGB f32 to sRGB f32
fn linear_to_srgb_f32(l: f32) -> f32 {
    if l <= 0.0031308 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

fn premultiply_f32(arr: &mut Array3<f32>) {
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    for y in 0..h {
        for x in 0..w {
            let a = arr[[y, x, 3]];
            if a > 0.0 && a < 1.0 {
                arr[[y, x, 0]] *= a;
                arr[[y, x, 1]] *= a;
                arr[[y, x, 2]] *= a;
            } else if a <= 0.0 {
                arr[[y, x, 0]] = 0.0;
                arr[[y, x, 1]] = 0.0;
                arr[[y, x, 2]] = 0.0;
            }
        }
    }
}

fn unpremultiply_f32(arr: &mut Array3<f32>) {
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    for y in 0..h {
        for x in 0..w {
            let a = arr[[y, x, 3]];
            if a > 0.0 && a < 1.0 {
                arr[[y, x, 0]] = (arr[[y, x, 0]] / a).clamp(0.0, 1.0);
                arr[[y, x, 1]] = (arr[[y, x, 1]] / a).clamp(0.0, 1.0);
                arr[[y, x, 2]] = (arr[[y, x, 2]] / a).clamp(0.0, 1.0);
            }
        }
    }
}

/// Convert sRGB u8 to Linear RGB u8 (using LUT)
#[inline(always)]
fn srgb_to_linear(val: u8) -> u8 {
    SRGB_TO_LINEAR_LUT[val as usize]
}

/// Convert Linear RGB u8 to sRGB u8 (using LUT)
#[inline(always)]
fn linear_to_srgb(val: u8) -> u8 {
    LINEAR_TO_SRGB_LUT[val as usize]
}

/// Convert entire array from sRGB to Linear RGB (skipping alpha)
fn array_srgb_to_linear(arr: &ArrayView3<u8>) -> Array3<u8> {
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    Array3::from_shape_fn((h, w, 4), |(y, x, c)| {
        if c == 3 {
            arr[[y, x, 3]]  // Alpha unchanged
        } else {
            SRGB_TO_LINEAR_LUT[arr[[y, x, c]] as usize]
        }
    })
}

/// Convert entire array from Linear RGB to sRGB (skipping alpha)
fn array_linear_to_srgb(arr: &Array3<u8>) -> Array3<u8> {
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    Array3::from_shape_fn((h, w, 4), |(y, x, c)| {
        if c == 3 {
            arr[[y, x, 3]]  // Alpha unchanged
        } else {
            LINEAR_TO_SRGB_LUT[arr[[y, x, c]] as usize]
        }
    })
}

fn premultiply(arr: &mut Array3<u8>) {
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    for y in 0..h {
        for x in 0..w {
            let a = arr[[y, x, 3]] as f32 / 255.0;
            if a > 0.0 && a < 1.0 {
                arr[[y, x, 0]] = (arr[[y, x, 0]] as f32 * a + 0.5) as u8;
                arr[[y, x, 1]] = (arr[[y, x, 1]] as f32 * a + 0.5) as u8;
                arr[[y, x, 2]] = (arr[[y, x, 2]] as f32 * a + 0.5) as u8;
            } else if a <= 0.0 {
                arr[[y, x, 0]] = 0;
                arr[[y, x, 1]] = 0;
                arr[[y, x, 2]] = 0;
            }
        }
    }
}

fn unpremultiply(arr: &mut Array3<u8>) {
    let (h, w, _) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    for y in 0..h {
        for x in 0..w {
            let a = arr[[y, x, 3]] as f32 / 255.0;
            if a > 0.0 && a < 1.0 {
                arr[[y, x, 0]] = (arr[[y, x, 0]] as f32 / a + 0.5).clamp(0.0, 255.0) as u8;
                arr[[y, x, 1]] = (arr[[y, x, 1]] as f32 / a + 0.5).clamp(0.0, 255.0) as u8;
                arr[[y, x, 2]] = (arr[[y, x, 2]] as f32 / a + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Get the result name from a filter primitive
fn get_result_name(prim: &FilterPrimitive) -> &str {
    match prim {
        FilterPrimitive::GaussianBlur { result, .. } => result,
        FilterPrimitive::Offset { result, .. } => result,
        FilterPrimitive::Flood { result, .. } => result,
        FilterPrimitive::Merge { result, .. } => result,
        FilterPrimitive::ColorMatrix { result, .. } => result,
        FilterPrimitive::Blend { result, .. } => result,
        FilterPrimitive::Composite { result, .. } => result,
        FilterPrimitive::Morphology { result, .. } => result,
        FilterPrimitive::Turbulence { result, .. } => result,
        FilterPrimitive::Tile { result, .. } => result,
        FilterPrimitive::ComponentTransfer { result, .. } => result,
        FilterPrimitive::ConvolveMatrix { result, .. } => result,
        FilterPrimitive::DiffuseLighting { result, .. } => result,
        FilterPrimitive::SpecularLighting { result, .. } => result,
        FilterPrimitive::DisplacementMap { result, .. } => result,
        FilterPrimitive::DropShadow { result, .. } => result,
        FilterPrimitive::Image { result, .. } => result,
    }
}

/// Get color interpolation for primitive
fn get_prim_color_interpolation(prim: &FilterPrimitive) -> ColorInterpolation {
    match prim {
        FilterPrimitive::GaussianBlur { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Offset { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Flood { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Merge { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::ColorMatrix { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Blend { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Composite { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Morphology { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Turbulence { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Tile { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::ComponentTransfer { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::ConvolveMatrix { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::DiffuseLighting { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::SpecularLighting { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::DisplacementMap { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::DropShadow { color_interpolation, .. } => *color_interpolation,
        FilterPrimitive::Image { color_interpolation, .. } => *color_interpolation,
    }
}

/// Get input array from results or use last result
fn get_input<'a>(
    input: &str,
    results: &'a HashMap<String, Array3<u8>>,
    last_result: &'a Array3<u8>,
) -> ArrayView3<'a, u8> {
    if input.is_empty() {
        return last_result.view();
    }
    if input == "SourceGraphic" {
        if let Some(arr) = results.get("SourceGraphic") {
            return arr.view();
        }
    }
    if let Some(arr) = results.get(input) {
        return arr.view();
    }
    last_result.view()
}
