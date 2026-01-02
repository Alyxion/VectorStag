//! Full SVG renderer using VectorStag's own implementation.
//!
//! This module provides complete SVG parsing and rendering in Rust,
//! eliminating Python→Rust boundary crossings for maximum performance.

use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};
use ndarray::Array3;
use roxmltree::{Document, Node};
use std::collections::HashMap;

/// RGBA color
#[derive(Clone, Copy, Debug, Default)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Color {
    fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    fn transparent() -> Self {
        Self { r: 0, g: 0, b: 0, a: 0 }
    }

    fn white() -> Self {
        Self { r: 255, g: 255, b: 255, a: 255 }
    }
}

/// 2D Transform matrix (a, b, c, d, e, f)
#[derive(Clone, Copy, Debug)]
struct Transform {
    a: f64, b: f64, c: f64, d: f64, e: f64, f: f64,
}

impl Default for Transform {
    fn default() -> Self {
        Self { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: 0.0, f: 0.0 }
    }
}

impl Transform {
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }

    fn multiply(&self, other: &Transform) -> Transform {
        Transform {
            a: self.a * other.a + self.c * other.b,
            b: self.b * other.a + self.d * other.b,
            c: self.a * other.c + self.c * other.d,
            d: self.b * other.c + self.d * other.d,
            e: self.a * other.e + self.c * other.f + self.e,
            f: self.b * other.e + self.d * other.f + self.f,
        }
    }

    fn scale(sx: f64, sy: f64) -> Transform {
        Transform { a: sx, b: 0.0, c: 0.0, d: sy, e: 0.0, f: 0.0 }
    }

    fn translate(tx: f64, ty: f64) -> Transform {
        Transform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: tx, f: ty }
    }

    fn rotate(angle: f64) -> Transform {
        let cos = angle.cos();
        let sin = angle.sin();
        Transform { a: cos, b: sin, c: -sin, d: cos, e: 0.0, f: 0.0 }
    }
}

/// Style properties for an element
#[derive(Clone, Debug, Default)]
struct Style {
    fill: Option<Paint>,
    stroke: Option<Paint>,
    stroke_width: f64,
    fill_opacity: f64,
    stroke_opacity: f64,
    opacity: f64,
    fill_rule: FillRule,
    stroke_linecap: LineCap,
    stroke_linejoin: LineJoin,
    stroke_miterlimit: f64,
    display: bool, // true = visible, false = none
}

impl Style {
    fn new() -> Self {
        Self {
            fill: Some(Paint::Color(Color::from_rgba(0, 0, 0, 255))),
            stroke: None,
            stroke_width: 1.0,
            fill_opacity: 1.0,
            stroke_opacity: 1.0,
            opacity: 1.0,
            fill_rule: FillRule::NonZero,
            stroke_linecap: LineCap::Butt,
            stroke_linejoin: LineJoin::Miter,
            stroke_miterlimit: 4.0,
            display: true,
        }
    }
}

#[derive(Clone, Debug)]
enum Paint {
    Color(Color),
    Gradient(String),
    None,
}

#[derive(Clone, Copy, Debug, Default)]
enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy, Debug, Default)]
enum LineCap {
    #[default]
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, Default)]
enum LineJoin {
    #[default]
    Miter,
    Round,
    Bevel,
}

/// Gradient definition
#[derive(Clone, Debug)]
struct GradientDef {
    id: String,
    is_radial: bool,
    // Linear gradient coords
    x1: f64, y1: f64, x2: f64, y2: f64,
    // Radial gradient coords
    cx: f64, cy: f64, r: f64, fx: f64, fy: f64,
    // Stops: (offset, r, g, b, a)
    stops: Vec<(f64, u8, u8, u8, u8)>,
    // Units
    user_space: bool,
    // Transform
    transform: Transform,
}

/// Render context
struct RenderContext {
    buffer: Vec<u8>,
    width: usize,
    height: usize,
    gradients: HashMap<String, GradientDef>,
    antialias: u32,
    shapes_rendered: usize,
}

impl RenderContext {
    fn new(width: usize, height: usize, background: Color, antialias: u32) -> Self {
        let render_width = width * antialias as usize;
        let render_height = height * antialias as usize;
        let mut buffer = vec![0u8; render_width * render_height * 4];

        // Fill with background
        for i in 0..(render_width * render_height) {
            buffer[i * 4] = background.r;
            buffer[i * 4 + 1] = background.g;
            buffer[i * 4 + 2] = background.b;
            buffer[i * 4 + 3] = background.a;
        }

        Self {
            buffer,
            width: render_width,
            height: render_height,
            gradients: HashMap::new(),
            antialias,
            shapes_rendered: 0,
        }
    }

    fn can_render_more(&self) -> bool {
        self.shapes_rendered < MAX_SHAPES
    }

    fn increment_shapes(&mut self) {
        self.shapes_rendered += 1;
    }

    fn blend_pixel(&mut self, x: usize, y: usize, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) * 4;
        let src_a = color.a as f32 / 255.0;

        if src_a >= 1.0 {
            // Fully opaque - direct copy
            self.buffer[idx] = color.r;
            self.buffer[idx + 1] = color.g;
            self.buffer[idx + 2] = color.b;
            self.buffer[idx + 3] = 255;
        } else if src_a > 0.0 {
            // Alpha blend
            let dst_a = self.buffer[idx + 3] as f32 / 255.0;
            let out_a = src_a + dst_a * (1.0 - src_a);

            if out_a > 0.0 {
                let blend = |src: u8, dst: u8| -> u8 {
                    let s = src as f32;
                    let d = dst as f32;
                    ((s * src_a + d * dst_a * (1.0 - src_a)) / out_a) as u8
                };

                self.buffer[idx] = blend(color.r, self.buffer[idx]);
                self.buffer[idx + 1] = blend(color.g, self.buffer[idx + 1]);
                self.buffer[idx + 2] = blend(color.b, self.buffer[idx + 2]);
                self.buffer[idx + 3] = (out_a * 255.0) as u8;
            }
        }
    }

    fn fill_polygon(&mut self, points: &[(f64, f64)], color: Color, fill_rule: FillRule) {
        if points.len() < 3 || color.a == 0 {
            return;
        }

        // Get bounding box
        let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        // Early exit if polygon is completely outside the canvas
        if max_x < 0.0 || min_x >= self.width as f64 ||
           max_y < 0.0 || min_y >= self.height as f64 {
            return;
        }

        let y_start = (min_y.floor() as i32).max(0) as usize;
        let y_end = (max_y.ceil() as i32).min(self.height as i32) as usize;
        let x_start = (min_x.floor() as i32).max(0) as usize;
        let x_end = (max_x.ceil() as i32).min(self.width as i32) as usize;

        // Early exit if no scanlines to process
        if y_start >= y_end || x_start >= x_end {
            return;
        }

        // Build edges (limit to prevent excessive processing)
        let n = points.len();
        if n > MAX_POLYGON_POINTS {
            return;
        }
        let mut edges: Vec<(f64, f64, f64, f64, i32)> = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            let (x1, y1) = points[i];
            let (x2, y2) = points[j];

            if (y1 - y2).abs() < 1e-10 {
                continue;
            }

            let (x1, y1, x2, y2, dir) = if y1 < y2 {
                (x1, y1, x2, y2, 1)
            } else {
                (x2, y2, x1, y1, -1)
            };

            edges.push((x1, y1, x2, y2, dir));
        }

        // Scanline fill
        for y in y_start..y_end {
            let scan_y = y as f64 + 0.5;
            let mut intersections: Vec<(f64, i32)> = Vec::new();

            for &(x1, y1, x2, y2, dir) in &edges {
                if y1 <= scan_y && scan_y < y2 {
                    let t = (scan_y - y1) / (y2 - y1);
                    let x = x1 + t * (x2 - x1);
                    intersections.push((x, dir));
                }
            }

            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            match fill_rule {
                FillRule::NonZero => {
                    let mut winding = 0;
                    let mut last_x: Option<f64> = None;

                    for (x, dir) in intersections {
                        if winding != 0 {
                            if let Some(lx) = last_x {
                                let x_s = (lx.floor() as usize).max(x_start);
                                let x_e = (x.ceil() as usize).min(x_end);
                                for px in x_s..x_e {
                                    self.blend_pixel(px, y, color);
                                }
                            }
                        }
                        winding += dir;
                        last_x = Some(x);
                    }
                }
                FillRule::EvenOdd => {
                    let mut inside = false;
                    let mut last_x: Option<f64> = None;

                    for (x, _) in intersections {
                        if inside {
                            if let Some(lx) = last_x {
                                let x_s = (lx.floor() as usize).max(x_start);
                                let x_e = (x.ceil() as usize).min(x_end);
                                for px in x_s..x_e {
                                    self.blend_pixel(px, y, color);
                                }
                            }
                        }
                        inside = !inside;
                        last_x = Some(x);
                    }
                }
            }
        }
    }

    /// Interpolate gradient color at position t (0.0 to 1.0)
    fn interpolate_gradient_color(stops: &[(f64, u8, u8, u8, u8)], t: f64) -> Color {
        if stops.is_empty() {
            return Color::from_rgba(0, 0, 0, 255);
        }
        if stops.len() == 1 {
            let s = &stops[0];
            return Color::from_rgba(s.1, s.2, s.3, s.4);
        }

        let t = t.clamp(0.0, 1.0);

        // Find the two stops to interpolate between
        let mut prev_stop = &stops[0];
        for stop in stops.iter() {
            if stop.0 >= t {
                // Interpolate between prev_stop and stop
                let range = stop.0 - prev_stop.0;
                if range < 0.001 {
                    return Color::from_rgba(stop.1, stop.2, stop.3, stop.4);
                }
                let local_t = (t - prev_stop.0) / range;
                let r = (prev_stop.1 as f64 + (stop.1 as f64 - prev_stop.1 as f64) * local_t) as u8;
                let g = (prev_stop.2 as f64 + (stop.2 as f64 - prev_stop.2 as f64) * local_t) as u8;
                let b = (prev_stop.3 as f64 + (stop.3 as f64 - prev_stop.3 as f64) * local_t) as u8;
                let a = (prev_stop.4 as f64 + (stop.4 as f64 - prev_stop.4 as f64) * local_t) as u8;
                return Color::from_rgba(r, g, b, a);
            }
            prev_stop = stop;
        }

        // Past the last stop
        let last = stops.last().unwrap();
        Color::from_rgba(last.1, last.2, last.3, last.4)
    }

    /// Fill polygon with a gradient
    fn fill_polygon_gradient(&mut self, points: &[(f64, f64)], gradient: &GradientDef,
                             transform: &Transform, fill_rule: FillRule, opacity: f64) {
        if points.len() < 3 {
            return;
        }

        // Get bounding box
        let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        // Early exit if polygon is completely outside the canvas
        if max_x < 0.0 || min_x >= self.width as f64 ||
           max_y < 0.0 || min_y >= self.height as f64 {
            return;
        }

        let y_start = (min_y.floor() as i32).max(0) as usize;
        let y_end = (max_y.ceil() as i32).min(self.height as i32) as usize;
        let x_start = (min_x.floor() as i32).max(0) as usize;
        let x_end = (max_x.ceil() as i32).min(self.width as i32) as usize;

        if y_start >= y_end || x_start >= x_end {
            return;
        }

        // Build edges
        let n = points.len();
        if n > MAX_POLYGON_POINTS {
            return;
        }
        let mut edges: Vec<(f64, f64, f64, f64, i32)> = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            let (x1, y1) = points[i];
            let (x2, y2) = points[j];

            if (y1 - y2).abs() < 1e-10 {
                continue;
            }

            let (x1, y1, x2, y2, dir) = if y1 < y2 {
                (x1, y1, x2, y2, 1)
            } else {
                (x2, y2, x1, y1, -1)
            };

            edges.push((x1, y1, x2, y2, dir));
        }

        // Compute gradient parameters based on bounding box (for objectBoundingBox units)
        let (gx1, gy1, gx2, gy2, gcx, gcy, gr) = if gradient.user_space {
            // userSpaceOnUse - apply transform to gradient coords
            let (gx1, gy1) = transform.apply(gradient.x1, gradient.y1);
            let (gx2, gy2) = transform.apply(gradient.x2, gradient.y2);
            let (gcx, gcy) = transform.apply(gradient.cx, gradient.cy);
            // Scale radius by average scale factor
            let scale = ((transform.a * transform.a + transform.b * transform.b).sqrt() +
                        (transform.c * transform.c + transform.d * transform.d).sqrt()) / 2.0;
            let gr = gradient.r * scale;
            (gx1, gy1, gx2, gy2, gcx, gcy, gr)
        } else {
            // objectBoundingBox - coords are 0-1 relative to bounding box (or 0-100 for percentage)
            let bbox_w = max_x - min_x;
            let bbox_h = max_y - min_y;
            let normalize = |v: f64| if v > 1.0 { v / 100.0 } else { v };
            let gx1 = min_x + normalize(gradient.x1) * bbox_w;
            let gy1 = min_y + normalize(gradient.y1) * bbox_h;
            let gx2 = min_x + normalize(gradient.x2) * bbox_w;
            let gy2 = min_y + normalize(gradient.y2) * bbox_h;
            let gcx = min_x + normalize(gradient.cx) * bbox_w;
            let gcy = min_y + normalize(gradient.cy) * bbox_h;
            let gr = normalize(gradient.r) * bbox_w.max(bbox_h);
            (gx1, gy1, gx2, gy2, gcx, gcy, gr)
        };

        // Precompute linear gradient direction
        let grad_dx = gx2 - gx1;
        let grad_dy = gy2 - gy1;
        let grad_len_sq = grad_dx * grad_dx + grad_dy * grad_dy;

        // Scanline fill with gradient
        for y in y_start..y_end {
            let scan_y = y as f64 + 0.5;
            let mut intersections: Vec<(f64, i32)> = Vec::new();

            for &(x1, y1, x2, y2, dir) in &edges {
                if y1 <= scan_y && scan_y < y2 {
                    let t = (scan_y - y1) / (y2 - y1);
                    let x = x1 + t * (x2 - x1);
                    intersections.push((x, dir));
                }
            }

            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            match fill_rule {
                FillRule::NonZero => {
                    let mut winding = 0;
                    let mut last_x: Option<f64> = None;

                    for (x, dir) in intersections {
                        if winding != 0 {
                            if let Some(lx) = last_x {
                                let x_s = (lx.floor() as usize).max(x_start);
                                let x_e = (x.ceil() as usize).min(x_end);
                                for px in x_s..x_e {
                                    let t = if gradient.is_radial {
                                        // Radial gradient: distance from center
                                        let dx = px as f64 + 0.5 - gcx;
                                        let dy = scan_y - gcy;
                                        let dist = (dx * dx + dy * dy).sqrt();
                                        if gr > 0.0 { dist / gr } else { 0.0 }
                                    } else {
                                        // Linear gradient: project onto gradient line
                                        if grad_len_sq > 0.001 {
                                            let dx = px as f64 + 0.5 - gx1;
                                            let dy = scan_y - gy1;
                                            (dx * grad_dx + dy * grad_dy) / grad_len_sq
                                        } else {
                                            0.0
                                        }
                                    };
                                    let mut color = Self::interpolate_gradient_color(&gradient.stops, t);
                                    color.a = (color.a as f64 * opacity) as u8;
                                    self.blend_pixel(px, y, color);
                                }
                            }
                        }
                        winding += dir;
                        last_x = Some(x);
                    }
                }
                FillRule::EvenOdd => {
                    let mut inside = false;
                    let mut last_x: Option<f64> = None;

                    for (x, _) in intersections {
                        if inside {
                            if let Some(lx) = last_x {
                                let x_s = (lx.floor() as usize).max(x_start);
                                let x_e = (x.ceil() as usize).min(x_end);
                                for px in x_s..x_e {
                                    let t = if gradient.is_radial {
                                        let dx = px as f64 + 0.5 - gcx;
                                        let dy = scan_y - gcy;
                                        let dist = (dx * dx + dy * dy).sqrt();
                                        if gr > 0.0 { dist / gr } else { 0.0 }
                                    } else {
                                        if grad_len_sq > 0.001 {
                                            let dx = px as f64 + 0.5 - gx1;
                                            let dy = scan_y - gy1;
                                            (dx * grad_dx + dy * grad_dy) / grad_len_sq
                                        } else {
                                            0.0
                                        }
                                    };
                                    let mut color = Self::interpolate_gradient_color(&gradient.stops, t);
                                    color.a = (color.a as f64 * opacity) as u8;
                                    self.blend_pixel(px, y, color);
                                }
                            }
                        }
                        inside = !inside;
                        last_x = Some(x);
                    }
                }
            }
        }
    }

    fn downsample(&self, out_width: usize, out_height: usize) -> Vec<u8> {
        if self.antialias == 1 {
            return self.buffer.clone();
        }

        let aa = self.antialias as usize;
        let mut result = vec![0u8; out_width * out_height * 4];
        let area = (aa * aa) as u32;

        for dy in 0..out_height {
            for dx in 0..out_width {
                let mut r: u32 = 0;
                let mut g: u32 = 0;
                let mut b: u32 = 0;
                let mut a: u32 = 0;

                for sy in 0..aa {
                    for sx in 0..aa {
                        let src_x = dx * aa + sx;
                        let src_y = dy * aa + sy;
                        let idx = (src_y * self.width + src_x) * 4;
                        r += self.buffer[idx] as u32;
                        g += self.buffer[idx + 1] as u32;
                        b += self.buffer[idx + 2] as u32;
                        a += self.buffer[idx + 3] as u32;
                    }
                }

                let dst_idx = (dy * out_width + dx) * 4;
                result[dst_idx] = (r / area) as u8;
                result[dst_idx + 1] = (g / area) as u8;
                result[dst_idx + 2] = (b / area) as u8;
                result[dst_idx + 3] = (a / area) as u8;
            }
        }

        result
    }
}

/// Parse color from string
fn parse_color(s: &str) -> Option<Color> {
    let s = s.trim();

    if s == "none" || s == "transparent" {
        return None;
    }

    // currentColor defaults to black (the inherited text color)
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
fn parse_paint(s: &str) -> Paint {
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

/// Parse transform attribute
fn parse_transform(s: &str) -> Transform {
    let mut result = Transform::default();
    let s = s.trim();

    // Simple regex-free parsing
    let mut remaining = s;
    while !remaining.is_empty() {
        remaining = remaining.trim_start();

        if remaining.starts_with("translate(") {
            let end = remaining.find(')').unwrap_or(remaining.len());
            let args = &remaining[10..end];
            let nums: Vec<f64> = args.split(|c| c == ',' || c == ' ')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if nums.len() >= 1 {
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
            if nums.len() >= 1 {
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
            if nums.len() >= 1 {
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
            // Skip unknown content
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
fn parse_style(node: &Node, parent_style: &Style) -> Style {
    let mut style = parent_style.clone();

    // Parse style attribute
    if let Some(style_attr) = node.attribute("style") {
        for part in style_attr.split(';') {
            let part = part.trim();
            if let Some(colon) = part.find(':') {
                let prop = part[..colon].trim();
                let val = part[colon + 1..].trim();
                apply_style_property(&mut style, prop, val);
            }
        }
    }

    // Parse individual attributes (override style)
    for attr in node.attributes() {
        apply_style_property(&mut style, attr.name(), attr.value());
    }

    style
}

fn apply_style_property(style: &mut Style, prop: &str, val: &str) {
    match prop {
        "fill" => style.fill = Some(parse_paint(val)),
        "stroke" => style.stroke = Some(parse_paint(val)),
        "stroke-width" => {
            if let Ok(w) = val.trim_end_matches("px").parse() {
                style.stroke_width = w;
            }
        }
        "fill-opacity" => {
            if let Ok(o) = val.parse() {
                style.fill_opacity = o;
            }
        }
        "stroke-opacity" => {
            if let Ok(o) = val.parse() {
                style.stroke_opacity = o;
            }
        }
        "opacity" => {
            if let Ok(o) = val.parse() {
                style.opacity = o;
            }
        }
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
        "display" => {
            style.display = val != "none";
        }
        "visibility" => {
            if val == "hidden" || val == "collapse" {
                style.display = false;
            }
        }
        _ => {}
    }
}

/// Convert path data to list of polygons
fn path_to_polygons(d: &str, transform: &Transform) -> Vec<Vec<(f64, f64)>> {
    let commands = crate::path::parse_path_internal(d);
    let mut polygons: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut current_poly: Vec<(f64, f64)> = Vec::new();
    let mut current_x = 0.0;
    let mut current_y = 0.0;

    for cmd in commands {
        match cmd {
            crate::path::PathCmd::M(x, y) => {
                if current_poly.len() >= 3 {
                    polygons.push(current_poly);
                }
                current_poly = Vec::new();
                let (tx, ty) = transform.apply(x, y);
                current_poly.push((tx, ty));
                current_x = x;
                current_y = y;
            }
            crate::path::PathCmd::L(x, y) => {
                let (tx, ty) = transform.apply(x, y);
                current_poly.push((tx, ty));
                current_x = x;
                current_y = y;
            }
            crate::path::PathCmd::C(x1, y1, x2, y2, x, y) => {
                // Sample cubic bezier
                let n_samples = 16;
                for i in 1..=n_samples {
                    let t = i as f64 / n_samples as f64;
                    let mt = 1.0 - t;
                    let px = mt.powi(3) * current_x
                        + 3.0 * mt.powi(2) * t * x1
                        + 3.0 * mt * t.powi(2) * x2
                        + t.powi(3) * x;
                    let py = mt.powi(3) * current_y
                        + 3.0 * mt.powi(2) * t * y1
                        + 3.0 * mt * t.powi(2) * y2
                        + t.powi(3) * y;
                    let (tx, ty) = transform.apply(px, py);
                    current_poly.push((tx, ty));
                }
                current_x = x;
                current_y = y;
            }
            crate::path::PathCmd::Q(x1, y1, x, y) => {
                // Sample quadratic bezier
                let n_samples = 8;
                for i in 1..=n_samples {
                    let t = i as f64 / n_samples as f64;
                    let mt = 1.0 - t;
                    let px = mt.powi(2) * current_x
                        + 2.0 * mt * t * x1
                        + t.powi(2) * x;
                    let py = mt.powi(2) * current_y
                        + 2.0 * mt * t * y1
                        + t.powi(2) * y;
                    let (tx, ty) = transform.apply(px, py);
                    current_poly.push((tx, ty));
                }
                current_x = x;
                current_y = y;
            }
            crate::path::PathCmd::Z => {
                if current_poly.len() >= 2 {
                    polygons.push(current_poly);
                }
                current_poly = Vec::new();
            }
            _ => {}
        }
    }

    if current_poly.len() >= 2 {
        polygons.push(current_poly);
    }

    polygons
}

/// Find an element by ID in the document tree
fn find_element_by_id<'a>(root: &Node<'a, '_>, id: &str) -> Option<Node<'a, 'a>> {
    // First check if root matches
    if root.attribute("id") == Some(id) {
        return Some(*root);
    }

    // Search descendants
    for desc in root.descendants() {
        if desc.attribute("id") == Some(id) {
            return Some(desc);
        }
    }
    None
}

/// Maximum recursion depth to prevent infinite loops
const MAX_DEPTH: usize = 100;

/// Maximum number of shapes to render per SVG to prevent hangs
const MAX_SHAPES: usize = 500;

/// Maximum points per polygon
const MAX_POLYGON_POINTS: usize = 10000;

/// Render a node and its children
fn render_node(ctx: &mut RenderContext, node: &Node, parent_transform: &Transform, parent_style: &Style, depth: usize, root: &Node) {
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
    // These are either metadata or effect definitions that need special handling
    match tag {
        // Metadata elements
        "metadata" | "title" | "desc" | "style" | "script" => return,
        // Effect definitions (not rendered directly, applied elsewhere)
        "filter" | "clipPath" | "mask" | "pattern" | "symbol" | "marker" => return,
        // Text elements (not implemented yet)
        "text" | "tspan" | "textPath" | "font" | "font-face" | "glyph" | "missing-glyph" => return,
        // Other elements to skip
        "switch" | "foreignObject" | "image" => return,
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

    // Render based on element type
    match tag {
        "g" | "svg" => {
            // Group - render children
            for child in node.children() {
                render_node(ctx, &child, &transform, &style, depth + 1, root);
            }
        }
        "path" => {
            if let Some(d) = node.attribute("d") {
                render_path(ctx, d, &transform, &style);
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
            render_line(ctx, node, &transform, &style);
        }
        "polyline" => {
            render_polyline(ctx, node, &transform, &style);
        }
        "polygon" => {
            render_polygon_elem(ctx, node, &transform, &style);
        }
        "use" => {
            // Resolve href attribute (try both href and xlink:href)
            let href = node.attribute("href")
                .or_else(|| node.attribute(("http://www.w3.org/1999/xlink", "href")));

            if let Some(href) = href {
                // Strip # prefix if present
                let target_id = href.trim_start_matches('#');

                // Find the referenced element
                if let Some(target) = find_element_by_id(root, target_id) {
                    // Get x/y offset for positioning
                    let x = node.attribute("x")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let y = node.attribute("y")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);

                    // Create transform with x/y translation
                    let use_transform = if x != 0.0 || y != 0.0 {
                        let translate = Transform {
                            a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: x, f: y,
                        };
                        transform.multiply(&translate)
                    } else {
                        transform.clone()
                    };

                    // For symbol elements, render their children with symbol's viewBox
                    // For other elements, render directly
                    let target_tag = target.tag_name().name();
                    if target_tag == "symbol" {
                        // Get use element width/height (defaults to symbol or parent viewBox)
                        let use_width: f64 = node.attribute("width")
                            .and_then(|s| s.trim_end_matches("px").parse().ok())
                            .unwrap_or(0.0);
                        let use_height: f64 = node.attribute("height")
                            .and_then(|s| s.trim_end_matches("px").parse().ok())
                            .unwrap_or(0.0);

                        // Parse symbol viewBox if present
                        if let Some(viewbox_str) = target.attribute("viewBox") {
                            let parts: Vec<f64> = viewbox_str
                                .split(|c: char| c == ',' || c.is_whitespace())
                                .filter_map(|s| s.trim().parse().ok())
                                .collect();
                            if parts.len() == 4 {
                                let (vb_x, vb_y, vb_w, vb_h) = (parts[0], parts[1], parts[2], parts[3]);

                                // Get parent viewport size from root SVG viewBox
                                // (use defaults to 100% of viewport if no width/height specified)
                                let (viewport_w, viewport_h) = root.attribute("viewBox")
                                    .and_then(|vb| {
                                        let p: Vec<f64> = vb.split(|c: char| c == ',' || c.is_whitespace())
                                            .filter_map(|s| s.trim().parse().ok())
                                            .collect();
                                        if p.len() == 4 { Some((p[2], p[3])) } else { None }
                                    })
                                    .unwrap_or((vb_w, vb_h));

                                // Determine target size (from use element or default to viewport)
                                let target_w = if use_width > 0.0 { use_width } else { viewport_w };
                                let target_h = if use_height > 0.0 { use_height } else { viewport_h };

                                // Calculate scale to map viewBox to target size
                                let scale_x = target_w / vb_w;
                                let scale_y = target_h / vb_h;
                                let scale = scale_x.min(scale_y);

                                // Offset to center the scaled content (for preserveAspectRatio=xMidYMid)
                                let scaled_w = vb_w * scale;
                                let scaled_h = vb_h * scale;
                                let center_offset_x = (target_w - scaled_w) / 2.0;
                                let center_offset_y = (target_h - scaled_h) / 2.0;

                                // The viewBox transform: translate to account for viewBox origin, then scale
                                // Points at (vb_x, vb_y) should map to (0, 0) after scaling
                                let symbol_transform = use_transform
                                    .multiply(&Transform::translate(center_offset_x, center_offset_y))
                                    .multiply(&Transform::scale(scale, scale))
                                    .multiply(&Transform::translate(-vb_x, -vb_y));

                                for child in target.children() {
                                    render_node(ctx, &child, &symbol_transform, &style, depth + 1, root);
                                }
                            } else {
                                for child in target.children() {
                                    render_node(ctx, &child, &use_transform, &style, depth + 1, root);
                                }
                            }
                        } else {
                            for child in target.children() {
                                render_node(ctx, &child, &use_transform, &style, depth + 1, root);
                            }
                        }
                    } else {
                        render_node(ctx, &target, &use_transform, &style, depth + 1, root);
                    }
                }
            }
        }
        "a" => {
            // Links - render children
            for child in node.children() {
                render_node(ctx, &child, &transform, &style, depth + 1, root);
            }
        }
        _ => {
            // Unknown element - skip (don't recurse into unknown elements as they may be effect containers)
        }
    }
}

fn collect_defs(ctx: &mut RenderContext, node: &Node) {
    for child in node.children() {
        if !child.is_element() {
            continue;
        }

        let tag = child.tag_name().name();
        if tag == "linearGradient" || tag == "radialGradient" {
            if let Some(id) = child.attribute("id") {
                let is_radial = tag == "radialGradient";

                let mut grad = GradientDef {
                    id: id.to_string(),
                    is_radial,
                    x1: child.attribute("x1").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(0.0),
                    y1: child.attribute("y1").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(0.0),
                    x2: child.attribute("x2").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(100.0),
                    y2: child.attribute("y2").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(0.0),
                    cx: child.attribute("cx").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
                    cy: child.attribute("cy").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
                    r: child.attribute("r").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
                    fx: child.attribute("fx").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
                    fy: child.attribute("fy").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(50.0),
                    stops: Vec::new(),
                    user_space: child.attribute("gradientUnits") == Some("userSpaceOnUse"),
                    transform: child.attribute("gradientTransform")
                        .map(parse_transform)
                        .unwrap_or_default(),
                };

                // Collect stops
                for stop in child.children() {
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
        }
    }
}

/// Recursively collect all gradients from the entire document tree
fn collect_all_gradients(ctx: &mut RenderContext, node: &Node) {
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();

    // Collect gradients regardless of where they appear
    if tag == "linearGradient" || tag == "radialGradient" {
        if let Some(id) = node.attribute("id") {
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
        return; // Don't recurse into gradient children (stops already handled)
    }

    // Recurse into children
    for child in node.children() {
        collect_all_gradients(ctx, &child);
    }
}

fn render_path(ctx: &mut RenderContext, d: &str, transform: &Transform, style: &Style) {
    // Check shape limit
    if !ctx.can_render_more() {
        return;
    }

    let polygons = path_to_polygons(d, transform);

    for poly in &polygons {
        // Skip very large polygons
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
                    // Look up gradient and fill
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

    // Stroke (simplified)
    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                for poly in &polygons {
                    if poly.len() > MAX_POLYGON_POINTS {
                        continue;
                    }
                    // Paths are typically open (not closed) unless explicitly looped
                    render_stroke(ctx, poly, c, style.stroke_width * transform.a.abs(),
                        style.stroke_linecap, style.stroke_linejoin, false);
                }
            }
        }
    }
}

fn render_stroke(ctx: &mut RenderContext, points: &[(f64, f64)], color: Color, width: f64, linecap: LineCap, linejoin: LineJoin, closed: bool) {
    if points.len() < 2 || color.a == 0 {
        return;
    }

    let half_width = width / 2.0;

    // Draw stroke as thick line segments
    let n = if closed { points.len() } else { points.len() - 1 };
    for i in 0..n {
        let j = (i + 1) % points.len();
        let (x1, y1) = points[i];
        let (x2, y2) = points[j];

        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 0.001 {
            continue;
        }

        let perp_x = -dy / len * half_width;
        let perp_y = dx / len * half_width;

        let quad = vec![
            (x1 + perp_x, y1 + perp_y),
            (x2 + perp_x, y2 + perp_y),
            (x2 - perp_x, y2 - perp_y),
            (x1 - perp_x, y1 - perp_y),
        ];

        ctx.fill_polygon(&quad, color, FillRule::NonZero);
    }

    // Draw round linejoins at internal vertices
    if matches!(linejoin, LineJoin::Round) && points.len() > 2 {
        let start = if closed { 0 } else { 1 };
        let end = if closed { points.len() } else { points.len() - 1 };
        for i in start..end {
            let (cx, cy) = points[i];
            draw_circle(ctx, cx, cy, half_width, color);
        }
    }

    // Draw linecaps on open paths
    if !closed {
        match linecap {
            LineCap::Round => {
                // Draw circles at start and end
                let (x1, y1) = points[0];
                let (x2, y2) = points[points.len() - 1];
                draw_circle(ctx, x1, y1, half_width, color);
                draw_circle(ctx, x2, y2, half_width, color);
            }
            LineCap::Square => {
                // Extend the stroke by half_width at each end
                if points.len() >= 2 {
                    // Start cap
                    let (x1, y1) = points[0];
                    let (x2, y2) = points[1];
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len > 0.001 {
                        let ext_x = -dx / len * half_width;
                        let ext_y = -dy / len * half_width;
                        let perp_x = -dy / len * half_width;
                        let perp_y = dx / len * half_width;
                        let cap = vec![
                            (x1 + perp_x, y1 + perp_y),
                            (x1 - perp_x, y1 - perp_y),
                            (x1 + ext_x - perp_x, y1 + ext_y - perp_y),
                            (x1 + ext_x + perp_x, y1 + ext_y + perp_y),
                        ];
                        ctx.fill_polygon(&cap, color, FillRule::NonZero);
                    }
                    // End cap
                    let (x1, y1) = points[points.len() - 2];
                    let (x2, y2) = points[points.len() - 1];
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len > 0.001 {
                        let ext_x = dx / len * half_width;
                        let ext_y = dy / len * half_width;
                        let perp_x = -dy / len * half_width;
                        let perp_y = dx / len * half_width;
                        let cap = vec![
                            (x2 + perp_x, y2 + perp_y),
                            (x2 - perp_x, y2 - perp_y),
                            (x2 + ext_x - perp_x, y2 + ext_y - perp_y),
                            (x2 + ext_x + perp_x, y2 + ext_y + perp_y),
                        ];
                        ctx.fill_polygon(&cap, color, FillRule::NonZero);
                    }
                }
            }
            LineCap::Butt => {}
        }
    }
}

fn draw_circle(ctx: &mut RenderContext, cx: f64, cy: f64, radius: f64, color: Color) {
    // Create circle polygon
    let segments = 16;
    let mut circle: Vec<(f64, f64)> = Vec::with_capacity(segments);
    for i in 0..segments {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
        circle.push((cx + radius * angle.cos(), cy + radius * angle.sin()));
    }
    ctx.fill_polygon(&circle, color, FillRule::NonZero);
}

fn render_rect(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();
    let x: f64 = node.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y: f64 = node.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let w: f64 = node.attribute("width").and_then(|s| s.trim_end_matches("px").parse().ok()).unwrap_or(0.0);
    let h: f64 = node.attribute("height").and_then(|s| s.trim_end_matches("px").parse().ok()).unwrap_or(0.0);

    if w <= 0.0 || h <= 0.0 {
        return;
    }

    // Parse rx/ry for rounded corners
    let mut rx: f64 = node.attribute("rx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let mut ry: f64 = node.attribute("ry").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    // Per SVG spec: if only rx or ry is specified, the other defaults to it
    if rx > 0.0 && ry == 0.0 { ry = rx; }
    if ry > 0.0 && rx == 0.0 { rx = ry; }

    // Clamp to half width/height
    rx = rx.min(w / 2.0);
    ry = ry.min(h / 2.0);

    let corners = if rx > 0.0 && ry > 0.0 {
        // Rounded rectangle
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let segments = 8; // Segments per corner

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
            Paint::Gradient(id) => {
                if let Some(gradient) = ctx.gradients.get(id).cloned() {
                    let opacity = style.fill_opacity * style.opacity;
                    ctx.fill_polygon_gradient(&corners, &gradient, transform, style.fill_rule, opacity);
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
                    style.stroke_linecap, style.stroke_linejoin, true);
            }
        }
    }
}

fn render_circle(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();
    let cx: f64 = node.attribute("cx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let cy: f64 = node.attribute("cy").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let r: f64 = node.attribute("r").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    if r <= 0.0 {
        return;
    }

    // Approximate circle with polygon
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
            Paint::Gradient(id) => {
                if let Some(gradient) = ctx.gradients.get(id).cloned() {
                    let opacity = style.fill_opacity * style.opacity;
                    ctx.fill_polygon_gradient(&points, &gradient, transform, style.fill_rule, opacity);
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
                    style.stroke_linecap, style.stroke_linejoin, true);
            }
        }
    }
}

fn render_ellipse(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();
    let cx: f64 = node.attribute("cx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let cy: f64 = node.attribute("cy").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let rx: f64 = node.attribute("rx").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let ry: f64 = node.attribute("ry").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    if rx <= 0.0 || ry <= 0.0 {
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
            Paint::Gradient(id) => {
                if let Some(gradient) = ctx.gradients.get(id).cloned() {
                    let opacity = style.fill_opacity * style.opacity;
                    ctx.fill_polygon_gradient(&points, &gradient, transform, style.fill_rule, opacity);
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
                    style.stroke_linecap, style.stroke_linejoin, true);
            }
        }
    }
}

fn render_line(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();
    let x1: f64 = node.attribute("x1").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y1: f64 = node.attribute("y1").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let x2: f64 = node.attribute("x2").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y2: f64 = node.attribute("y2").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                let p1 = transform.apply(x1, y1);
                let p2 = transform.apply(x2, y2);
                render_stroke(ctx, &[p1, p2], c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, false);
            }
        }
    }
}

fn render_polyline(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
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
                // Open polyline - render as single stroke
                render_stroke(ctx, &points, c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, false);
            }
        }
    }
}

fn render_polygon_elem(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
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
            Paint::Gradient(id) => {
                if let Some(gradient) = ctx.gradients.get(id).cloned() {
                    let opacity = style.fill_opacity * style.opacity;
                    ctx.fill_polygon_gradient(&points, &gradient, transform, style.fill_rule, opacity);
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
                    style.stroke_linecap, style.stroke_linejoin, true);
            }
        }
    }
}

fn parse_points(s: &str, transform: &Transform) -> Vec<(f64, f64)> {
    let nums: Vec<f64> = s.split(|c: char| c == ',' || c.is_whitespace())
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    nums.chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| transform.apply(c[0], c[1]))
        .collect()
}

/// Parse viewBox attribute
fn parse_viewbox(s: &str) -> Option<(f64, f64, f64, f64)> {
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
fn parse_length(s: &str, default: f64) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return default;
    }

    // Remove units
    let num_str = s.trim_end_matches(|c: char| c.is_alphabetic() || c == '%');
    num_str.parse().unwrap_or(default)
}

/// Full SVG renderer using VectorStag's own implementation
#[pyclass]
pub struct VectorStagRenderer {
    antialias: u32,
}

#[pymethods]
impl VectorStagRenderer {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self { antialias: 4 })
    }

    /// Render SVG content to a numpy array
    #[pyo3(signature = (svg_content, width=None, height=None, scale=None, background=None, antialias=None))]
    fn render<'py>(
        &self,
        py: Python<'py>,
        svg_content: &str,
        width: Option<u32>,
        height: Option<u32>,
        scale: Option<f32>,
        background: Option<(u8, u8, u8, u8)>,
        antialias: Option<u8>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let scale = scale.unwrap_or(1.0) as f64;
        let antialias = antialias.unwrap_or(4) as u32;
        let bg = background.unwrap_or((255, 255, 255, 255));
        let background = Color::from_rgba(bg.0, bg.1, bg.2, bg.3);

        // Parse SVG
        let doc = Document::parse(svg_content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to parse SVG: {}", e)
            ))?;

        let root = doc.root_element();

        // Get document dimensions
        let viewbox = root.attribute("viewBox").and_then(parse_viewbox);
        let svg_width = root.attribute("width")
            .map(|s| parse_length(s, 100.0))
            .or_else(|| viewbox.map(|v| v.2))
            .unwrap_or(100.0);
        let svg_height = root.attribute("height")
            .map(|s| parse_length(s, 100.0))
            .or_else(|| viewbox.map(|v| v.3))
            .unwrap_or(100.0);

        // Calculate output dimensions
        let (out_width, out_height) = match (width, height) {
            (Some(w), Some(h)) => (w as usize, h as usize),
            (Some(w), None) => {
                let aspect = svg_height / svg_width;
                (w as usize, (w as f64 * aspect) as usize)
            }
            (None, Some(h)) => {
                let aspect = svg_width / svg_height;
                ((h as f64 * aspect) as usize, h as usize)
            }
            (None, None) => (
                (svg_width * scale) as usize,
                (svg_height * scale) as usize,
            ),
        };

        let out_width = out_width.max(1);
        let out_height = out_height.max(1);

        // Create render context
        let mut ctx = RenderContext::new(out_width, out_height, background, antialias);

        // Calculate transform from viewBox to output
        let (vb_x, vb_y, vb_w, vb_h) = viewbox.unwrap_or((0.0, 0.0, svg_width, svg_height));
        let render_width = out_width as f64 * antialias as f64;
        let render_height = out_height as f64 * antialias as f64;

        let scale_x = render_width / vb_w;
        let scale_y = render_height / vb_h;
        let scale_factor = scale_x.min(scale_y);

        let offset_x = (render_width - vb_w * scale_factor) / 2.0 - vb_x * scale_factor;
        let offset_y = (render_height - vb_h * scale_factor) / 2.0 - vb_y * scale_factor;

        let base_transform = Transform::translate(offset_x, offset_y)
            .multiply(&Transform::scale(scale_factor, scale_factor));

        // First pass: collect all gradients from the entire document
        for child in root.children() {
            collect_all_gradients(&mut ctx, &child);
        }

        // Second pass: render tree
        // Parse style from root SVG element (many SVGs like Lucide define stroke/fill on root)
        let base_style = Style::new();
        let root_style = parse_style(&root, &base_style);
        for child in root.children() {
            render_node(&mut ctx, &child, &base_transform, &root_style, 0, &root);
        }

        // Downsample
        let final_buffer = ctx.downsample(out_width, out_height);

        // Convert to numpy array
        let arr = Array3::from_shape_vec(
            (out_height, out_width, 4),
            final_buffer,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create array: {}", e)
        ))?;

        Ok(arr.into_pyarray(py))
    }

    /// Render SVG file to numpy array
    #[pyo3(signature = (file_path, width=None, height=None, scale=None, background=None, antialias=None))]
    fn render_file<'py>(
        &self,
        py: Python<'py>,
        file_path: &str,
        width: Option<u32>,
        height: Option<u32>,
        scale: Option<f32>,
        background: Option<(u8, u8, u8, u8)>,
        antialias: Option<u8>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let svg_content = std::fs::read_to_string(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to read file: {}", e)
            ))?;

        self.render(py, &svg_content, width, height, scale, background, antialias)
    }
}

/// Register the svg_renderer module
pub fn register(m: &Bound<'_, pyo3::prelude::PyModule>) -> PyResult<()> {
    m.add_class::<VectorStagRenderer>()?;
    Ok(())
}
