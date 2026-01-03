//! Full SVG renderer using VectorStag's own implementation.
//!
//! This module provides complete SVG parsing and rendering in Rust,
//! eliminating Python→Rust boundary crossings for maximum performance.

use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};
use ndarray::Array3;
use roxmltree::{Document, Node};
use std::collections::HashMap;
use std::sync::Arc;
use crate::text::FontManager;
use crate::path::PathCmd;

mod preserve_aspect_ratio;
use preserve_aspect_ratio::*;

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

    #[allow(dead_code)]
    fn transparent() -> Self {
        Self { r: 0, g: 0, b: 0, a: 0 }
    }

    #[allow(dead_code)]
    fn white() -> Self {
        Self { r: 255, g: 255, b: 255, a: 255 }
    }
}

/// 2D Transform matrix (a, b, c, d, e, f)
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub a: f64, pub b: f64, pub c: f64, pub d: f64, pub e: f64, pub f: f64,
}

impl Default for Transform {
    fn default() -> Self {
        Self { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: 0.0, f: 0.0 }
    }
}

impl Transform {
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }

    pub fn multiply(&self, other: &Transform) -> Transform {
        Transform {
            a: self.a * other.a + self.c * other.b,
            b: self.b * other.a + self.d * other.b,
            c: self.a * other.c + self.c * other.d,
            d: self.b * other.c + self.d * other.d,
            e: self.a * other.e + self.c * other.f + self.e,
            f: self.b * other.e + self.d * other.f + self.f,
        }
    }

    pub fn scale(sx: f64, sy: f64) -> Transform {
        Transform { a: sx, b: 0.0, c: 0.0, d: sy, e: 0.0, f: 0.0 }
    }

    pub fn translate(tx: f64, ty: f64) -> Transform {
        Transform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: tx, f: ty }
    }

    pub fn rotate(angle: f64) -> Transform {
        let cos = angle.cos();
        let sin = angle.sin();
        Transform { a: cos, b: sin, c: -sin, d: cos, e: 0.0, f: 0.0 }
    }

    #[allow(dead_code)]
    pub fn invert(&self) -> Option<Transform> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-10 {
            return None;
        }
        Some(Transform {
            a: self.d / det,
            b: -self.b / det,
            c: -self.c / det,
            d: self.a / det,
            e: (self.c * self.f - self.d * self.e) / det,
            f: (self.b * self.e - self.a * self.f) / det,
        })
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
    visibility: bool, // true = visible, false = hidden/collapse
    // Font properties
    font_family: String,
    font_size: f64,
    font_weight: u16,
    font_style: String, // "normal", "italic", "oblique"
    text_anchor: String, // "start", "middle", "end"
    // Marker references
    marker_start: Option<String>,
    marker_mid: Option<String>,
    marker_end: Option<String>,
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
            visibility: true,
            font_family: "sans-serif".to_string(),
            font_size: 12.0,
            font_weight: 400,
            font_style: "normal".to_string(),
            text_anchor: "start".to_string(),
            marker_start: None,
            marker_mid: None,
            marker_end: None,
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
#[allow(dead_code)]
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

/// ClipPath definition - stores path data for clipping
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ClipPathDef {
    id: String,
    // Polygons that define the clip region
    polygons: Vec<Vec<(f64, f64)>>,
    // Whether coordinates are in userSpaceOnUse
    user_space: bool,
}

/// Mask definition
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct MaskDef {
    id: String,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// Marker orientation
#[derive(Clone, Debug)]
enum MarkerOrient {
    Auto,
    AutoStartReverse,
    Angle(f64), // in radians
}

/// Marker definition
#[derive(Clone, Debug)]
struct MarkerDef {
    id: String,
    ref_x: f64,
    ref_y: f64,
    marker_width: f64,
    marker_height: f64,
    orient: MarkerOrient,
    // viewBox if specified
    viewbox: Option<(f64, f64, f64, f64)>,
    // markerUnits: true = strokeWidth, false = userSpaceOnUse
    stroke_width_units: bool,
    // The marker element node content (we'll store the XML string to re-render)
    // Actually, we'll store pre-parsed path data for children
    children_xml: String,
}

/// Render context
struct RenderContext {
    buffer: Vec<u8>,
    width: usize,
    height: usize,
    gradients: HashMap<String, GradientDef>,
    clip_paths: HashMap<String, ClipPathDef>,
    masks: HashMap<String, MaskDef>,
    markers: HashMap<String, MarkerDef>,
    antialias: u32,
    shapes_rendered: usize,
    // Active clip path for current element (polygons in render coordinates)
    active_clip: Option<Vec<Vec<(f64, f64)>>>,
    // Bounding box of the active clip path (min_x, min_y, max_x, max_y)
    active_clip_bbox: Option<(f64, f64, f64, f64)>,
    // Font manager for text rendering
    font_manager: Arc<FontManager>,
}

impl RenderContext {
    fn new(width: usize, height: usize, background: Color, antialias: u32, font_manager: Arc<FontManager>) -> Self {
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
            clip_paths: HashMap::new(),
            masks: HashMap::new(),
            markers: HashMap::new(),
            antialias,
            shapes_rendered: 0,
            active_clip: None,
            active_clip_bbox: None,
            font_manager,
        }
    }

    fn can_render_more(&self) -> bool {
        self.shapes_rendered < MAX_SHAPES
    }

    fn increment_shapes(&mut self) {
        self.shapes_rendered += 1;
    }

    /// Check if a point is inside the active clip path
    fn is_inside_clip(&self, x: f64, y: f64) -> bool {
        match &self.active_clip {
            None => true, // No clip = always inside
            Some(clip_polygons) => {
                // Check bounding box first
                if let Some((min_x, min_y, max_x, max_y)) = self.active_clip_bbox {
                    if x < min_x || x > max_x || y < min_y || y > max_y {
                        return false;
                    }
                }

                for polygon in clip_polygons {
                    if polygon.len() < 3 {
                        continue;
                    }
                    // Ray casting algorithm
                    let mut inside = false;
                    let n = polygon.len();
                    let mut j = n - 1;
                    for i in 0..n {
                        let (xi, yi) = polygon[i];
                        let (xj, yj) = polygon[j];
                        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                            inside = !inside;
                        }
                        j = i;
                    }
                    if inside {
                        return true;
                    }
                }
                false
            }
        }
    }

    fn blend_pixel(&mut self, x: usize, y: usize, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }

        // Check clip
        if !self.is_inside_clip(x as f64 + 0.5, y as f64 + 0.5) {
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
            // effective transform = element_transform * gradient_transform
            let combined_transform = transform.multiply(&gradient.transform);
            
            let (gx1, gy1) = combined_transform.apply(gradient.x1, gradient.y1);
            let (gx2, gy2) = combined_transform.apply(gradient.x2, gradient.y2);
            let (gcx, gcy) = combined_transform.apply(gradient.cx, gradient.cy);
            // Scale radius by average scale factor
            let scale = ((combined_transform.a * combined_transform.a + combined_transform.b * combined_transform.b).sqrt() +
                        (combined_transform.c * combined_transform.c + combined_transform.d * combined_transform.d).sqrt()) / 2.0;
            let gr = gradient.r * scale;
            (gx1, gy1, gx2, gy2, gcx, gcy, gr)
        } else {
            // objectBoundingBox - coords are 0-1 relative to bounding box (or 0-100 for percentage)
            let bbox_w = max_x - min_x;
            let bbox_h = max_y - min_y;
            let normalize = |v: f64| if v > 1.0 { v / 100.0 } else { v };
            
            // Apply gradientTransform to the normalized coordinates
            let (tx1, ty1) = gradient.transform.apply(normalize(gradient.x1), normalize(gradient.y1));
            let (tx2, ty2) = gradient.transform.apply(normalize(gradient.x2), normalize(gradient.y2));
            let (tcx, tcy) = gradient.transform.apply(normalize(gradient.cx), normalize(gradient.cy));
            
            // Transform radius (scale only)
            let scale = ((gradient.transform.a * gradient.transform.a + gradient.transform.b * gradient.transform.b).sqrt() +
                        (gradient.transform.c * gradient.transform.c + gradient.transform.d * gradient.transform.d).sqrt()) / 2.0;
            let tr = normalize(gradient.r) * scale;

            // Map to screen space using bounding box
            let gx1 = min_x + tx1 * bbox_w;
            let gy1 = min_y + ty1 * bbox_h;
            let gx2 = min_x + tx2 * bbox_w;
            let gy2 = min_y + ty2 * bbox_h;
            let gcx = min_x + tcx * bbox_w;
            let gcy = min_y + tcy * bbox_h;
            let gr = tr * bbox_w.max(bbox_h);
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
                // #RGB - expand each digit
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                Some(Color::from_rgba(r, g, b, 255))
            }
            4 => {
                // #RGBA (CSS4) - expand each digit
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

/// Parse marker URL reference (e.g., "url(#marker1)")
fn parse_marker_url(s: &str) -> Option<String> {
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

    // Helper to apply a single property
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
            // Font properties
            "font-family" => style.font_family = val.trim_matches('\'').trim_matches('"').to_string(),
            "font-size" => style.font_size = parse_length(val, 12.0),
            "font-weight" => style.font_weight = match val {
                "bold" => 700,
                "normal" => 400,
                _ => val.parse().unwrap_or(400),
            },
            "font-style" => style.font_style = val.to_string(),
            "text-anchor" => style.text_anchor = val.to_string(),
            // Marker properties
            "marker-start" => style.marker_start = parse_marker_url(val),
            "marker-mid" => style.marker_mid = parse_marker_url(val),
            "marker-end" => style.marker_end = parse_marker_url(val),
            "marker" => {
                // Shorthand sets all three
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

/// Convert path commands to list of polygons
fn commands_to_polygons(commands: &[PathCmd], transform: &Transform) -> Vec<Vec<(f64, f64)>> {
    let mut polygons: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut current_poly: Vec<(f64, f64)> = Vec::new();
    let mut last_point = (0.0, 0.0);
    let mut start_point = (0.0, 0.0);

    for cmd in commands {
        match cmd {
            PathCmd::M(x, y) => {
                if !current_poly.is_empty() {
                    polygons.push(current_poly);
                    current_poly = Vec::new();
                }
                let p = transform.apply(*x, *y);
                current_poly.push(p);
                last_point = p;
                start_point = p;
            }
            PathCmd::L(x, y) => {
                let p = transform.apply(*x, *y);
                if current_poly.is_empty() {
                    current_poly.push(last_point);
                }
                current_poly.push(p);
                last_point = p;
            }
            PathCmd::C(x1, y1, x2, y2, x, y) => {
                let p0 = last_point;
                let p1 = transform.apply(*x1, *y1);
                let p2 = transform.apply(*x2, *y2);
                let p3 = transform.apply(*x, *y);

                // Estimate number of segments based on distance
                let dist = (p1.0 - p0.0).hypot(p1.1 - p0.1) + 
                           (p2.0 - p1.0).hypot(p2.1 - p1.1) + 
                           (p3.0 - p2.0).hypot(p3.1 - p2.1);
                let segments = (dist / 2.0).max(4.0).min(100.0) as usize;

                let points = crate::path::sample_cubic_bezier(
                    p0.0, p0.1, p1.0, p1.1, p2.0, p2.1, p3.0, p3.1, segments
                );

                if current_poly.is_empty() {
                    current_poly.push(p0);
                }
                current_poly.extend(points);
                last_point = p3;
            }
            PathCmd::Q(x1, y1, x, y) => {
                let p0 = last_point;
                let p1 = transform.apply(*x1, *y1);
                let p2 = transform.apply(*x, *y);

                let dist = (p1.0 - p0.0).hypot(p1.1 - p0.1) + (p2.0 - p1.0).hypot(p2.1 - p1.1);
                let segments = (dist / 2.0).max(4.0).min(100.0) as usize;

                let points = crate::path::sample_quadratic_bezier(
                    p0.0, p0.1, p1.0, p1.1, p2.0, p2.1, segments
                );

                if current_poly.is_empty() {
                    current_poly.push(p0);
                }
                current_poly.extend(points);
                last_point = p2;
            }
            PathCmd::A(_rx, _ry, _rot, _large_arc, _sweep, x, y) => {
                // Arcs are usually converted to C by parser, but handle just in case
                let p = transform.apply(*x, *y);
                if current_poly.is_empty() {
                    current_poly.push(last_point);
                }
                current_poly.push(p);
                last_point = p;
            }
            PathCmd::Z => {
                if !current_poly.is_empty() {
                    // Close polygon
                    if (last_point.0 - start_point.0).hypot(last_point.1 - start_point.1) > 1e-6 {
                        current_poly.push(start_point);
                    }
                    polygons.push(current_poly);
                    current_poly = Vec::new();
                    last_point = start_point;
                }
            }
        }
    }

    if !current_poly.is_empty() {
        polygons.push(current_poly);
    }

    polygons
}

/// Convert path data to list of polygons
fn path_to_polygons(d: &str, transform: &Transform) -> Vec<Vec<(f64, f64)>> {
    let commands = crate::path::parse_path_internal(d);
    commands_to_polygons(&commands, transform)
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
        // Font definition elements (text is handled below)
        "font" | "font-face" | "glyph" | "missing-glyph" => return,
        // Other elements to skip
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

    // Skip elements with visibility: hidden (they still take up space, but aren't drawn)
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
                // Save previous clip
                prev_clip = ctx.active_clip.clone();
                prev_clip_bbox = ctx.active_clip_bbox;

                // Prepare new clip polygons
                let new_polygons = if clip_def.user_space {
                     clip_def.polygons.clone()
                } else {
                    // TODO: Implement objectBoundingBox support
                    // Needs bbox of the element we are about to render.
                    clip_def.polygons.clone()
                };

                // If there was an active clip, we strictly should INTERSECT.
                // But doing polygon intersection is expensive/complex.
                // Alternative: render to mask buffer. 
                // Since we are doing per-pixel check:
                // We could change active_clip to be `Vec<ClipPathDef>` (stack of clips)
                // But `is_inside_clip` takes `&self`.
                
                // Let's stick to replacing for now, or just append if we want "inside A OR inside B" (Union).
                // But clipping is usually intersection.
                // As a quick fix/feature enable: just use the new clip.
                
                // Calculate bbox for the new clip
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
            // Group - render children
            for child in node.children() {
                render_node(ctx, &child, &transform, &style, depth + 1, root);
            }
        }
        "svg" => {
            // Nested SVG element - handle x, y positioning
            // Note: Full viewport/clipping support would require render-to-buffer approach
            let x = node.attribute("x")
                .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
                .unwrap_or(0.0);
            let y = node.attribute("y")
                .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
                .unwrap_or(0.0);

            // Apply x/y offset
            let nested_transform = if x != 0.0 || y != 0.0 {
                transform.multiply(&Transform::translate(x, y))
            } else {
                transform.clone()
            };

            for child in node.children() {
                render_node(ctx, &child, &nested_transform, &style, depth + 1, root);
            }
        }
        "switch" => {
            // Render the first child that passes its conditional tests
            for child in node.children() {
                if !child.is_element() {
                    continue;
                }

                // Check conditional attributes
                // requiredExtensions - we don't support any extensions, so skip if present
                if child.attribute("requiredExtensions").is_some() {
                    continue;
                }

                // requiredFeatures - skip for now (would need full feature detection)
                if child.attribute("requiredFeatures").is_some() {
                    continue;
                }

                // systemLanguage - for simplicity, accept "en" languages, skip others
                if let Some(lang) = child.attribute("systemLanguage") {
                    // Accept en, en-US, en-GB, etc. Skip others for simplicity.
                    if !lang.starts_with("en") {
                        continue;
                    }
                }

                // This child passes all tests - render it and stop
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
            // tspan and textPath are handled within render_text_element
            // If we encounter them at top level, ignore them
        }
        "image" => {
            render_image_element(ctx, node, &transform, &style);
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

                                // Parse preserveAspectRatio
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

    // Restore previous clip path
    ctx.active_clip = prev_clip;
    ctx.active_clip_bbox = prev_clip_bbox;
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
        } else if tag == "marker" {
            if let Some(id) = child.attribute("id") {
                let ref_x = child.attribute("refX")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let ref_y = child.attribute("refY")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let marker_width = child.attribute("markerWidth")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3.0);
                let marker_height = child.attribute("markerHeight")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3.0);

                let orient = match child.attribute("orient") {
                    Some("auto") => MarkerOrient::Auto,
                    Some("auto-start-reverse") => MarkerOrient::AutoStartReverse,
                    Some(s) => {
                        // Parse angle (may be in degrees or with "deg" suffix)
                        let angle_str = s.trim_end_matches("deg");
                        angle_str.parse::<f64>()
                            .map(|deg| MarkerOrient::Angle(deg.to_radians()))
                            .unwrap_or(MarkerOrient::Angle(0.0))
                    }
                    None => MarkerOrient::Angle(0.0),
                };

                let viewbox = child.attribute("viewBox").and_then(parse_viewbox);
                let stroke_width_units = child.attribute("markerUnits") != Some("userSpaceOnUse");

                // Store a representation of the marker content
                // We'll reconstruct from the node during rendering
                let marker = MarkerDef {
                    id: id.to_string(),
                    ref_x,
                    ref_y,
                    marker_width,
                    marker_height,
                    orient,
                    viewbox,
                    stroke_width_units,
                    children_xml: String::new(), // Not used - we render from node
                };

                ctx.markers.insert(id.to_string(), marker);
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

/// Recursively collect all markers from the entire document tree
fn collect_all_markers(ctx: &mut RenderContext, node: &Node) {
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();

    if tag == "marker" {
        if let Some(id) = node.attribute("id") {
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
        return; // Don't recurse into marker children
    }

    // Recurse into children
    for child in node.children() {
        collect_all_markers(ctx, &child);
    }
}

/// Collect all clipPath and mask definitions from the document
fn collect_clip_paths_and_masks(ctx: &mut RenderContext, node: &Node, transform: &Transform) {
    if !node.is_element() {
        return;
    }

    let tag = node.tag_name().name();

    if tag == "clipPath" {
        if let Some(id) = node.attribute("id") {
            let user_space = node.attribute("clipPathUnits") == Some("userSpaceOnUse");
            let mut polygons: Vec<Vec<(f64, f64)>> = Vec::new();

            // Collect all paths/shapes inside the clipPath
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
                            let polys = path_to_polygons(d, &combined);
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
                    _ => {}
                }
            }

            // Always insert the clipPath - even if empty
            // An empty clipPath should clip everything (show nothing)
            ctx.clip_paths.insert(id.to_string(), ClipPathDef {
                id: id.to_string(),
                polygons,
                user_space,
            });
        }
        return;
    }

    if tag == "mask" {
        if let Some(id) = node.attribute("id") {
            let x: f64 = node.attribute("x").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(-10.0);
            let y: f64 = node.attribute("y").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(-10.0);
            let width: f64 = node.attribute("width").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(120.0);
            let height: f64 = node.attribute("height").and_then(|s| s.trim_end_matches('%').parse().ok()).unwrap_or(120.0);
            ctx.masks.insert(id.to_string(), MaskDef { id: id.to_string(), x, y, width, height });
        }
        return;
    }

    // Recurse into children
    for child in node.children() {
        collect_clip_paths_and_masks(ctx, &child, transform);
    }
}

/// Check if a point is inside a polygon using ray casting algorithm
#[allow(dead_code)]
fn point_in_polygon(x: f64, y: f64, polygon: &[(f64, f64)]) -> bool {
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
fn point_in_clip_path(x: f64, y: f64, clip_path: &ClipPathDef) -> bool {
    for polygon in &clip_path.polygons {
        if point_in_polygon(x, y, polygon) {
            return true;
        }
    }
    false
}

/// Render markers on a path's vertices
fn render_markers(
    ctx: &mut RenderContext,
    points: &[(f64, f64)],
    style: &Style,
    transform: &Transform,
    root: &Node,
) {
    if points.len() < 2 {
        return;
    }

    let stroke_width = style.stroke_width * transform.a.abs();

    // Calculate angles for each point
    let mut angles: Vec<f64> = Vec::with_capacity(points.len());

    for i in 0..points.len() {
        let angle = if i == 0 {
            // Start point: angle from first to second point
            let (x1, y1) = points[0];
            let (x2, y2) = points[1];
            (y2 - y1).atan2(x2 - x1)
        } else if i == points.len() - 1 {
            // End point: angle from second-to-last to last point
            let (x1, y1) = points[points.len() - 2];
            let (x2, y2) = points[points.len() - 1];
            (y2 - y1).atan2(x2 - x1)
        } else {
            // Mid point: average of incoming and outgoing angles
            let (x0, y0) = points[i - 1];
            let (x1, y1) = points[i];
            let (x2, y2) = points[i + 1];
            let a1 = (y1 - y0).atan2(x1 - x0);
            let a2 = (y2 - y1).atan2(x2 - x1);
            // Average the angles (this is simplified - proper bisector is more complex)
            (a1 + a2) / 2.0
        };
        angles.push(angle);
    }

    // Render start marker
    if let Some(ref marker_id) = style.marker_start {
        if let Some(marker_def) = ctx.markers.get(marker_id).cloned() {
            let (x, y) = points[0];
            let angle = match &marker_def.orient {
                MarkerOrient::Auto => angles[0],
                MarkerOrient::AutoStartReverse => angles[0] + std::f64::consts::PI,
                MarkerOrient::Angle(a) => *a,
            };
            render_single_marker(ctx, &marker_def, x, y, angle, stroke_width, root, marker_id);
        }
    }

    // Render mid markers
    if let Some(ref marker_id) = style.marker_mid {
        if points.len() > 2 {
            if let Some(marker_def) = ctx.markers.get(marker_id).cloned() {
                for i in 1..(points.len() - 1) {
                    let (x, y) = points[i];
                    let angle = match &marker_def.orient {
                        MarkerOrient::Auto | MarkerOrient::AutoStartReverse => angles[i],
                        MarkerOrient::Angle(a) => *a,
                    };
                    render_single_marker(ctx, &marker_def, x, y, angle, stroke_width, root, marker_id);
                }
            }
        }
    }

    // Render end marker
    if let Some(ref marker_id) = style.marker_end {
        if let Some(marker_def) = ctx.markers.get(marker_id).cloned() {
            let (x, y) = points[points.len() - 1];
            let angle = match &marker_def.orient {
                MarkerOrient::Auto | MarkerOrient::AutoStartReverse => angles[points.len() - 1],
                MarkerOrient::Angle(a) => *a,
            };
            render_single_marker(ctx, &marker_def, x, y, angle, stroke_width, root, marker_id);
        }
    }
}

/// Render a single marker at a specific position
fn render_single_marker(
    ctx: &mut RenderContext,
    marker_def: &MarkerDef,
    x: f64,
    y: f64,
    angle: f64,
    stroke_width: f64,
    root: &Node,
    marker_id: &str,
) {
    // Find the marker element in the document
    if let Some(marker_elem) = find_element_by_id(root, marker_id) {
        // Calculate scale based on markerUnits
        let scale = if marker_def.stroke_width_units {
            stroke_width
        } else {
            1.0
        };

        // Calculate marker transform:
        // 1. Translate to marker position
        // 2. Rotate to path direction
        // 3. Scale by stroke width (if strokeWidth units)
        // 4. Apply viewBox transform if present
        // 5. Translate by -refX, -refY

        let marker_transform = if let Some((vb_x, vb_y, vb_w, vb_h)) = marker_def.viewbox {
            // With viewBox: scale from viewBox to markerWidth/Height
            let sx = marker_def.marker_width / vb_w;
            let sy = marker_def.marker_height / vb_h;
            let s = sx.min(sy) * scale;

            Transform::translate(x, y)
                .multiply(&Transform::rotate(angle))
                .multiply(&Transform::scale(s, s))
                .multiply(&Transform::translate(-marker_def.ref_x, -marker_def.ref_y))
        } else {
            // Without viewBox: use markerWidth/Height directly as the coordinate space
            Transform::translate(x, y)
                .multiply(&Transform::rotate(angle))
                .multiply(&Transform::scale(scale, scale))
                .multiply(&Transform::translate(-marker_def.ref_x, -marker_def.ref_y))
        };

        // Render marker children with the calculated transform
        let base_style = Style::new();
        for child in marker_elem.children() {
            render_node(ctx, &child, &marker_transform, &base_style, 0, root);
        }
    }
}

fn render_path_with_markers(ctx: &mut RenderContext, d: &str, transform: &Transform, style: &Style, root: &Node) {
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

/// Check if all points in a path are effectively at the same location (zero-length path)
fn is_zero_length_path(points: &[(f64, f64)]) -> bool {
    if points.is_empty() {
        return true;
    }
    let (x0, y0) = points[0];
    const EPSILON: f64 = 0.001;
    points.iter().all(|(x, y)| (x - x0).abs() < EPSILON && (y - y0).abs() < EPSILON)
}

fn render_stroke(ctx: &mut RenderContext, points: &[(f64, f64)], color: Color, width: f64, linecap: LineCap, linejoin: LineJoin, closed: bool) {
    if color.a == 0 {
        return;
    }

    let half_width = width / 2.0;

    // Handle zero-length paths (single point or all points identical)
    if points.len() == 1 || (points.len() >= 2 && is_zero_length_path(points)) {
        // For zero-length paths, draw based on linecap
        let (cx, cy) = points[0];
        match linecap {
            LineCap::Round => {
                // Draw a circle
                draw_circle(ctx, cx, cy, half_width, color);
            }
            LineCap::Square => {
                // Draw a square centered at the point
                let square = vec![
                    (cx - half_width, cy - half_width),
                    (cx + half_width, cy - half_width),
                    (cx + half_width, cy + half_width),
                    (cx - half_width, cy + half_width),
                ];
                ctx.fill_polygon(&square, color, FillRule::NonZero);
            }
            LineCap::Butt => {
                // Butt caps on zero-length paths render nothing
            }
        }
        return;
    }

    if points.len() < 2 {
        return;
    }

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

fn render_line(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style, root: &Node) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();
    let x1: f64 = node.attribute("x1").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y1: f64 = node.attribute("y1").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let x2: f64 = node.attribute("x2").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y2: f64 = node.attribute("y2").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let p1 = transform.apply(x1, y1);
    let p2 = transform.apply(x2, y2);
    let points = vec![p1, p2];

    if let Some(ref stroke) = style.stroke {
        if style.stroke_width > 0.0 {
            if let Paint::Color(color) = stroke {
                let mut c = *color;
                c.a = (c.a as f64 * style.stroke_opacity * style.opacity) as u8;
                render_stroke(ctx, &points, c, style.stroke_width * transform.a.abs(),
                    style.stroke_linecap, style.stroke_linejoin, false);
            }
        }
    }

    // Render markers
    let has_markers = style.marker_start.is_some() || style.marker_mid.is_some() || style.marker_end.is_some();
    if has_markers {
        render_markers(ctx, &points, style, transform, root);
    }
}

fn render_polyline(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style, root: &Node) {
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

    // Render markers
    let has_markers = style.marker_start.is_some() || style.marker_mid.is_some() || style.marker_end.is_some();
    if has_markers && points.len() >= 2 {
        render_markers(ctx, &points, style, transform, root);
    }
}

fn render_polygon_elem(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style, root: &Node) {
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

    // Render markers (polygons are closed, so vertices form a loop)
    let has_markers = style.marker_start.is_some() || style.marker_mid.is_some() || style.marker_end.is_some();
    if has_markers && points.len() >= 2 {
        render_markers(ctx, &points, style, transform, root);
    }
}

/// Render a text element
fn render_text_element(ctx: &mut RenderContext, node: &Node, transform: &Transform, style: &Style) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();

    // Get text position
    let x: f64 = node.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y: f64 = node.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    // Collect text content (direct text and tspan children)
    let text_content = collect_text_content(node);
    if text_content.is_empty() {
        return;
    }

    // Get font properties from style
    let font_family = &style.font_family;
    let font_size = style.font_size;
    let font_weight = style.font_weight;
    let italic = style.font_style == "italic" || style.font_style == "oblique";
    let text_anchor = &style.text_anchor;

    // Layout text to get glyph paths
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

    // Render each glyph
    for glyph_commands in glyph_paths {
        // Convert to polygons and render
        let polygons = commands_to_polygons(&glyph_commands, transform);

        for poly in &polygons {
            if poly.len() < 3 {
                continue;
            }

            // Fill the glyph
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
fn collect_text_content(node: &Node) -> String {
    let mut content = String::new();

    for child in node.children() {
        if child.is_text() {
            if let Some(text) = child.text() {
                content.push_str(text);
            }
        } else if child.is_element() && child.tag_name().name() == "tspan" {
            // Recursively collect tspan content
            content.push_str(&collect_text_content(&child));
        }
    }

    content
}

/// Render an image element (embedded or external)
fn render_image_element(ctx: &mut RenderContext, node: &Node, transform: &Transform, _style: &Style) {
    if !ctx.can_render_more() { return; }
    ctx.increment_shapes();

    // Get image dimensions and position
    let x: f64 = node.attribute("x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let y: f64 = node.attribute("y").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let width: f64 = node.attribute("width").and_then(|s| parse_length(s, 0.0).into()).unwrap_or(0.0);
    let height: f64 = node.attribute("height").and_then(|s| parse_length(s, 0.0).into()).unwrap_or(0.0);

    if width <= 0.0 || height <= 0.0 {
        return;
    }

    // Get href (try both forms)
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

    // Parse data URL
    let img_data = match decode_data_url(href) {
        Some(data) => data,
        None => return,
    };

    // Decode image
    let img = match image::load_from_memory(&img_data) {
        Ok(img) => img.to_rgba8(),
        Err(_) => return,
    };

    // Calculate destination rectangle in screen coordinates
    let (dst_x1, dst_y1) = transform.apply(x, y);
    let (dst_x2, dst_y2) = transform.apply(x + width, y + height);

    let dst_x = dst_x1.min(dst_x2) as i32;
    let dst_y = dst_y1.min(dst_y2) as i32;
    let dst_w = (dst_x2 - dst_x1).abs() as u32;
    let dst_h = (dst_y2 - dst_y1).abs() as u32;

    if dst_w == 0 || dst_h == 0 {
        return;
    }

    // Resize image to destination size
    let resized = image::imageops::resize(&img, dst_w, dst_h, image::imageops::FilterType::Lanczos3);

    // Composite onto canvas
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

            // Alpha compositing
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

/// Decode a data: URL to raw bytes
fn decode_data_url(url: &str) -> Option<Vec<u8>> {
    // Format: data:[<mediatype>][;base64],<data>
    let url = url.strip_prefix("data:")?;

    // Find the comma that separates metadata from data
    let comma_pos = url.find(',')?;
    let (metadata, data) = url.split_at(comma_pos);
    let data = &data[1..]; // Skip the comma

    // Check if base64 encoded
    if metadata.contains(";base64") {
        // Decode base64, ignoring whitespace
        let clean_data: String = data.chars().filter(|c| !c.is_whitespace()).collect();
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.decode(&clean_data).ok()
    } else {
        // URL-encoded data (not commonly used for images)
        None
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
    font_manager: Arc<FontManager>,
}

#[pymethods]
impl VectorStagRenderer {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self { 
            antialias: 4,
            font_manager: Arc::new(FontManager::new()),
        })
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
        let antialias = antialias.map(|a| a as u32).unwrap_or(self.antialias);
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
        let mut ctx = RenderContext::new(out_width, out_height, background, antialias, self.font_manager.clone());

        // Calculate transform from viewBox to output
        let (vb_x, vb_y, vb_w, vb_h) = viewbox.unwrap_or((0.0, 0.0, svg_width, svg_height));
        let render_width = out_width as f64 * antialias as f64;
        let render_height = out_height as f64 * antialias as f64;

        let par = root.attribute("preserveAspectRatio")
            .map(parse_preserve_aspect_ratio)
            .unwrap_or_default();

        let base_transform = compute_viewbox_transform(
            vb_x, vb_y, vb_w, vb_h,
            render_width, render_height,
            par
        );

        // First pass: collect all gradients from the entire document
        for child in root.children() {
            collect_all_gradients(&mut ctx, &child);
        }

        // Also collect all markers from the entire document
        for child in root.children() {
            collect_all_markers(&mut ctx, &child);
        }

        // Second pass: collect clipPaths and masks
        for child in root.children() {
            collect_clip_paths_and_masks(&mut ctx, &child, &base_transform);
        }

        // Third pass: render tree
        // Parse style from root SVG element (many SVGs like Lucide define stroke/fill on root)
        let base_style = Style::new();
        let root_style = parse_style(&root, &base_style);

        // Check if root SVG has display="none"
        if root_style.display {
            for child in root.children() {
                render_node(&mut ctx, &child, &base_transform, &root_style, 0, &root);
            }
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
