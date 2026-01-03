//! SVG filter collection and application.

use roxmltree::Node;
use ndarray::{Array3, ArrayView3};
use std::collections::HashMap;
use super::types::*;
use super::parsing::parse_color;
use crate::filters;

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

    let x = node.attribute("x")
        .and_then(|s| s.trim_end_matches('%').parse().ok())
        .unwrap_or(-10.0);
    let y = node.attribute("y")
        .and_then(|s| s.trim_end_matches('%').parse().ok())
        .unwrap_or(-10.0);
    let width = node.attribute("width")
        .and_then(|s| s.trim_end_matches('%').parse().ok())
        .unwrap_or(120.0);
    let height = node.attribute("height")
        .and_then(|s| s.trim_end_matches('%').parse().ok())
        .unwrap_or(120.0);

    let filter_units = node.attribute("filterUnits") != Some("userSpaceOnUse");
    let primitive_units = node.attribute("primitiveUnits") == Some("userSpaceOnUse");

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

/// Parse a filter primitive element
fn parse_filter_primitive(node: &Node) -> Option<FilterPrimitive> {
    let tag = node.tag_name().name();
    let input = node.attribute("in").unwrap_or("SourceGraphic").to_string();
    let result = node.attribute("result").unwrap_or("").to_string();

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
            })
        }
        "feOffset" => {
            let dx = node.attribute("dx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let dy = node.attribute("dy").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            Some(FilterPrimitive::Offset { dx, dy, input, result })
        }
        "feFlood" => {
            let color_str = node.attribute("flood-color").unwrap_or("black");
            let opacity: f64 = node.attribute("flood-opacity")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let mut color = parse_color(color_str).unwrap_or(Color::from_rgba(0, 0, 0, 255));
            color.a = (color.a as f64 * opacity) as u8;
            Some(FilterPrimitive::Flood { color, result })
        }
        "feMerge" => {
            let mut nodes = Vec::new();
            for child in node.children() {
                if child.is_element() && child.tag_name().name() == "feMergeNode" {
                    let in_ref = child.attribute("in").unwrap_or("").to_string();
                    nodes.push(in_ref);
                }
            }
            Some(FilterPrimitive::Merge { nodes, result })
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
            Some(FilterPrimitive::ColorMatrix { matrix_type, values, input, result })
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
            let in1 = node.attribute("in").unwrap_or("SourceGraphic").to_string();
            let in2 = node.attribute("in2").unwrap_or("BackgroundImage").to_string();
            Some(FilterPrimitive::Blend { mode, in1, in2, result })
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
            let in1 = node.attribute("in").unwrap_or("SourceGraphic").to_string();
            let in2 = node.attribute("in2").unwrap_or("BackgroundImage").to_string();
            Some(FilterPrimitive::Composite { operator, k1, k2, k3, k4, in1, in2, result })
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
            Some(FilterPrimitive::Morphology { operator, radius_x, radius_y, input, result })
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
            Some(FilterPrimitive::Turbulence { base_freq_x, base_freq_y, num_octaves, seed, noise_type, result })
        }
        "feTile" => {
            Some(FilterPrimitive::Tile { input, result })
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
            Some(FilterPrimitive::DropShadow { dx, dy, std_dev_x, std_dev_y, flood_color, input, result })
        }
        "feComponentTransfer" => {
            let func_r = parse_component_transfer_func(node, "feFuncR");
            let func_g = parse_component_transfer_func(node, "feFuncG");
            let func_b = parse_component_transfer_func(node, "feFuncB");
            let func_a = parse_component_transfer_func(node, "feFuncA");
            Some(FilterPrimitive::ComponentTransfer { func_r, func_g, func_b, func_a, input, result })
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
                order_x, order_y, kernel, divisor, bias, target_x, target_y, edge_mode, preserve_alpha, input, result
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
                input, result
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
                input, result
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
            let in1 = node.attribute("in").unwrap_or("SourceGraphic").to_string();
            let in2 = node.attribute("in2").unwrap_or("").to_string();
            Some(FilterPrimitive::DisplacementMap { scale, x_channel, y_channel, in1, in2, result })
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
    width: usize,
    height: usize,
    scale: f64,
) -> Vec<u8> {
    if filter.primitives.is_empty() {
        return source.to_vec();
    }

    // Convert source to ndarray for filter operations
    let source_arr = Array3::from_shape_fn((height, width, 4), |(y, x, c)| {
        source[(y * width + x) * 4 + c]
    });

    // Source alpha for SourceAlpha input
    let source_alpha = filters::get_source_alpha_impl(&source_arr.view());

    // Results map for named results
    let mut results: HashMap<String, Array3<u8>> = HashMap::new();
    results.insert("SourceGraphic".to_string(), source_arr.clone());
    results.insert("SourceAlpha".to_string(), source_alpha);

    // Last result for implicit chaining
    let mut last_result = source_arr;

    // Apply each primitive in sequence, scaling parameters by transform scale
    for prim in &filter.primitives {
        let output = apply_primitive(prim, &results, &last_result, width, height, scale);

        // Store result if named
        let result_name = get_result_name(prim);
        if !result_name.is_empty() {
            results.insert(result_name.to_string(), output.clone());
        }

        last_result = output;
    }

    // Convert final result back to Vec<u8>
    let mut output = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            output[idx] = last_result[[y, x, 0]];
            output[idx + 1] = last_result[[y, x, 1]];
            output[idx + 2] = last_result[[y, x, 2]];
            output[idx + 3] = last_result[[y, x, 3]];
        }
    }

    output
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

/// Get input array from results or use last result
fn get_input<'a>(
    input: &str,
    results: &'a HashMap<String, Array3<u8>>,
    last_result: &'a Array3<u8>,
) -> ArrayView3<'a, u8> {
    if input.is_empty() || input == "SourceGraphic" {
        if let Some(arr) = results.get("SourceGraphic") {
            return arr.view();
        }
    }
    if let Some(arr) = results.get(input) {
        return arr.view();
    }
    last_result.view()
}

/// Apply a single filter primitive
fn apply_primitive(
    prim: &FilterPrimitive,
    results: &HashMap<String, Array3<u8>>,
    last_result: &Array3<u8>,
    width: usize,
    height: usize,
    scale: f64,
) -> Array3<u8> {
    match prim {
        FilterPrimitive::GaussianBlur { std_dev_x, std_dev_y, input, .. } => {
            let src = get_input(input, results, last_result);
            // Scale stdDeviation by transform scale
            let scaled_x = (*std_dev_x * scale) as f32;
            let scaled_y = (*std_dev_y * scale) as f32;
            filters::fe_gaussian_blur_impl(&src, scaled_x, scaled_y)
        }
        FilterPrimitive::Offset { dx, dy, input, .. } => {
            let src = get_input(input, results, last_result);
            // Scale offset by transform scale
            let scaled_dx = (*dx * scale) as i32;
            let scaled_dy = (*dy * scale) as i32;
            filters::fe_offset_impl(&src, scaled_dx, scaled_dy)
        }
        FilterPrimitive::Flood { color, .. } => {
            filters::fe_flood_impl(width, height, color.r, color.g, color.b, color.a)
        }
        FilterPrimitive::Merge { nodes, .. } => {
            let layers: Vec<ArrayView3<u8>> = nodes.iter()
                .map(|n| get_input(n, results, last_result))
                .collect();
            filters::fe_merge_impl(&layers)
        }
        FilterPrimitive::ColorMatrix { matrix_type, values, input, .. } => {
            let src = get_input(input, results, last_result);
            filters::fe_color_matrix_impl(&src, *matrix_type, values)
        }
        FilterPrimitive::Blend { mode, in1, in2, .. } => {
            let src1 = get_input(in1, results, last_result);
            let src2 = get_input(in2, results, last_result);
            filters::fe_blend_impl(&src1, &src2, *mode)
        }
        FilterPrimitive::Composite { operator, k1, k2, k3, k4, in1, in2, .. } => {
            let src1 = get_input(in1, results, last_result);
            let src2 = get_input(in2, results, last_result);
            filters::fe_composite_impl(&src1, &src2, *operator, *k1, *k2, *k3, *k4)
        }
        FilterPrimitive::Morphology { operator, radius_x, radius_y, input, .. } => {
            let src = get_input(input, results, last_result);
            // Scale radius by transform scale
            let scaled_rx = (*radius_x * scale) as f32;
            let scaled_ry = (*radius_y * scale) as f32;
            filters::fe_morphology_impl(&src, *operator, scaled_rx, scaled_ry)
        }
        FilterPrimitive::Turbulence { base_freq_x, base_freq_y, num_octaves, seed, noise_type, .. } => {
            filters::fe_turbulence_impl(width, height, *base_freq_x, *base_freq_y, *num_octaves, *seed, *noise_type, false)
        }
        FilterPrimitive::Tile { input, .. } => {
            let src = get_input(input, results, last_result);
            filters::fe_tile_impl(&src, width, height)
        }
        FilterPrimitive::ComponentTransfer { func_r, func_g, func_b, func_a, input, .. } => {
            let src = get_input(input, results, last_result);
            filters::fe_component_transfer_impl(&src, func_r, func_g, func_b, func_a)
        }
        FilterPrimitive::ConvolveMatrix { order_x, order_y, kernel, divisor, bias, target_x, target_y, edge_mode, preserve_alpha, input, .. } => {
            let src = get_input(input, results, last_result);
            filters::fe_convolve_matrix_impl(&src, *order_x, *order_y, kernel, *divisor, *bias, *target_x, *target_y, *edge_mode, *preserve_alpha)
        }
        FilterPrimitive::DiffuseLighting { surface_scale, diffuse_constant, light_color, light_type, azimuth, elevation, light_x, light_y, light_z, points_at_x, points_at_y, points_at_z, specular_exponent, limiting_cone_angle, input, .. } => {
            let src = get_input(input, results, last_result);
            filters::fe_diffuse_lighting_impl(&src, *surface_scale, *diffuse_constant, *light_color, *light_type, *azimuth, *elevation, *light_x, *light_y, *light_z, *points_at_x, *points_at_y, *points_at_z, *specular_exponent, *limiting_cone_angle)
        }
        FilterPrimitive::SpecularLighting { surface_scale, specular_constant, specular_exponent, light_color, light_type, azimuth, elevation, light_x, light_y, light_z, points_at_x, points_at_y, points_at_z, spot_exponent, limiting_cone_angle, input, .. } => {
            let src = get_input(input, results, last_result);
            filters::fe_specular_lighting_impl(&src, *surface_scale, *specular_constant, *specular_exponent, *light_color, *light_type, *azimuth, *elevation, *light_x, *light_y, *light_z, *points_at_x, *points_at_y, *points_at_z, *spot_exponent, *limiting_cone_angle)
        }
        FilterPrimitive::DisplacementMap { scale: disp_scale, x_channel, y_channel, in1, in2, .. } => {
            let src = get_input(in1, results, last_result);
            let map = get_input(in2, results, last_result);
            // Scale displacement by transform scale
            let scaled_disp = *disp_scale * scale as f32;
            filters::fe_displacement_map_impl(&src, &map, scaled_disp, *x_channel, *y_channel)
        }
        FilterPrimitive::DropShadow { dx, dy, std_dev_x, std_dev_y, flood_color, input, .. } => {
            let src = get_input(input, results, last_result);
            // Scale drop shadow parameters by transform scale
            let scaled_dx = (*dx * scale) as f32;
            let scaled_dy = (*dy * scale) as f32;
            let scaled_std_x = (*std_dev_x * scale) as f32;
            let scaled_std_y = (*std_dev_y * scale) as f32;
            filters::fe_drop_shadow_impl(&src, scaled_dx, scaled_dy, scaled_std_x, scaled_std_y, flood_color.r, flood_color.g, flood_color.b, flood_color.a)
        }
        FilterPrimitive::Image { .. } => {
            // feImage not fully supported - return transparent
            Array3::<u8>::zeros((height, width, 4))
        }
    }
}
