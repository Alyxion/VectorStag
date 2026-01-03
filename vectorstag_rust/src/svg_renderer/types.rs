//! Core types for SVG rendering.

use std::collections::HashMap;
use std::sync::Arc;
use crate::text::FontManager;

/// RGBA color
#[derive(Clone, Copy, Debug, Default)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    #[allow(dead_code)]
    pub fn transparent() -> Self {
        Self { r: 0, g: 0, b: 0, a: 0 }
    }

    #[allow(dead_code)]
    pub fn white() -> Self {
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

/// Paint type for fill/stroke
#[derive(Clone, Debug)]
pub enum Paint {
    Color(Color),
    Gradient(String),
    None,
}

/// Fill rule for polygon filling
#[derive(Clone, Copy, Debug, Default)]
pub enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}

/// Line cap style
#[derive(Clone, Copy, Debug, Default)]
pub enum LineCap {
    #[default]
    Butt,
    Round,
    Square,
}

/// Line join style
#[derive(Clone, Copy, Debug, Default)]
pub enum LineJoin {
    #[default]
    Miter,
    Round,
    Bevel,
}

/// Style properties for an element
#[derive(Clone, Debug, Default)]
pub struct Style {
    pub fill: Option<Paint>,
    pub stroke: Option<Paint>,
    pub stroke_width: f64,
    pub fill_opacity: f64,
    pub stroke_opacity: f64,
    pub opacity: f64,
    pub fill_rule: FillRule,
    pub stroke_linecap: LineCap,
    pub stroke_linejoin: LineJoin,
    pub stroke_miterlimit: f64,
    pub display: bool,
    pub visibility: bool,
    // Font properties
    pub font_family: String,
    pub font_size: f64,
    pub font_weight: u16,
    pub font_style: String,
    pub text_anchor: String,
    // Marker references
    pub marker_start: Option<String>,
    pub marker_mid: Option<String>,
    pub marker_end: Option<String>,
}

impl Style {
    pub fn new() -> Self {
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

/// Gradient definition
#[derive(Clone, Debug)]
pub struct GradientDef {
    pub id: String,
    pub is_radial: bool,
    pub x1: f64, pub y1: f64, pub x2: f64, pub y2: f64,
    pub cx: f64, pub cy: f64, pub r: f64, pub fx: f64, pub fy: f64,
    pub stops: Vec<(f64, u8, u8, u8, u8)>,
    pub user_space: bool,
    pub transform: Transform,
}

/// ClipPath definition
#[derive(Clone, Debug)]
pub struct ClipPathDef {
    pub id: String,
    pub polygons: Vec<Vec<(f64, f64)>>,
    pub user_space: bool,
}

/// Mask definition
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct MaskDef {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Marker orientation
#[derive(Clone, Debug)]
pub enum MarkerOrient {
    Auto,
    AutoStartReverse,
    Angle(f64),
}

/// Marker definition
#[derive(Clone, Debug)]
pub struct MarkerDef {
    pub id: String,
    pub ref_x: f64,
    pub ref_y: f64,
    pub marker_width: f64,
    pub marker_height: f64,
    pub orient: MarkerOrient,
    pub viewbox: Option<(f64, f64, f64, f64)>,
    pub stroke_width_units: bool,
    pub children_xml: String,
}

/// Render context holding buffer and state
pub struct RenderContext {
    pub buffer: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub gradients: HashMap<String, GradientDef>,
    pub clip_paths: HashMap<String, ClipPathDef>,
    pub masks: HashMap<String, MaskDef>,
    pub markers: HashMap<String, MarkerDef>,
    pub antialias: u32,
    pub shapes_rendered: usize,
    pub active_clip: Option<Vec<Vec<(f64, f64)>>>,
    pub active_clip_bbox: Option<(f64, f64, f64, f64)>,
    pub font_manager: Arc<FontManager>,
}

/// Maximum shapes to render (prevents infinite loops)
pub const MAX_SHAPES: usize = 100_000;

/// Maximum polygon points
pub const MAX_POLYGON_POINTS: usize = 100_000;

/// Maximum recursion depth
pub const MAX_DEPTH: usize = 100;
