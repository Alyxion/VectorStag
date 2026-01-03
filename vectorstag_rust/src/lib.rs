//! VectorStag Rust extension for fast SVG rendering operations
//!
//! This library provides optimized implementations of:
//! - Polygon fill operations (nonzero/evenodd rules, anti-aliasing)
//! - Stroke rendering for closed polygons
//! - Gradient interpolation (linear and radial)
//! - SVG path parsing and curve sampling
//! - Image operations (compositing, resizing, color space conversion)
//! - SVG filter primitives (blur, morphology, lighting, etc.)
//! - CSS selector parsing and matching
//! - Full SVG rendering pipeline (VectorStagRenderer class)

use pyo3::prelude::*;

mod polygon;
mod stroke;
mod gradient;
pub mod path;
mod image;
mod filters;
mod text;
mod css;
mod canvas;
mod owned_canvas;
mod svg_renderer;

/// A Python module implemented in Rust for fast SVG rendering operations.
#[pymodule]
fn vectorstag_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all module functions
    polygon::register(m)?;
    stroke::register(m)?;
    gradient::register(m)?;
    path::register(m)?;
    image::register(m)?;
    filters::register(m)?;
    css::register(m)?;
    canvas::register(m)?;
    owned_canvas::register(m)?;
    svg_renderer::register(m)?;
    Ok(())
}
