//! Full SVG renderer using VectorStag's own implementation.
//!
//! This module provides complete SVG parsing and rendering in Rust,
//! eliminating Python→Rust boundary crossings for maximum performance.

use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};
use ndarray::Array3;
use roxmltree::Document;
use std::sync::Arc;
use crate::text::FontManager;

mod types;
mod context;
mod parsing;
mod defs;
mod path_utils;
mod markers;
mod stroke;
mod shapes;
mod elements;
mod render;
mod preserve_aspect_ratio;
mod filter;

use types::*;
use parsing::{parse_style, parse_viewbox, parse_length};
use defs::{collect_all_gradients, collect_all_patterns, collect_all_markers, collect_clip_paths_and_masks};
use filter::collect_all_filters;
use render::render_node;
use preserve_aspect_ratio::{parse_preserve_aspect_ratio, compute_viewbox_transform};

// Re-export Transform for use in other modules
pub use types::Transform;

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

        // Strip DOCTYPE declaration (not needed for rendering, causes roxmltree to reject)
        let svg_clean = if svg_content.contains("<!DOCTYPE") {
            let re = regex::Regex::new(r"<!DOCTYPE[^>]*>").unwrap();
            re.replace(svg_content, "").to_string()
        } else {
            svg_content.to_string()
        };

        // Parse SVG
        let doc = Document::parse(&svg_clean)
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

        // Set initial viewport dimensions to viewBox for percent calculations
        ctx.viewport_width = vb_w;
        ctx.viewport_height = vb_h;

        let par = root.attribute("preserveAspectRatio")
            .map(parse_preserve_aspect_ratio)
            .unwrap_or_default();

        let base_transform = compute_viewbox_transform(
            vb_x, vb_y, vb_w, vb_h,
            render_width, render_height,
            par
        );
        
        // Set viewbox scale for pattern sampling (userSpaceOnUse)
        // This is the scale from viewbox coordinates to render coordinates
        ctx.viewbox_scale_x = render_width / vb_w;
        ctx.viewbox_scale_y = render_height / vb_h;

        // Apply root-level clipping to viewBox (default overflow: hidden for <svg>)
        let overflow = root.attribute("overflow").unwrap_or("hidden");
        if overflow == "hidden" || overflow == "scroll" {
            let (x1, y1) = base_transform.apply(vb_x, vb_y);
            let (x2, y2) = base_transform.apply(vb_x + vb_w, vb_y);
            let (x3, y3) = base_transform.apply(vb_x + vb_w, vb_y + vb_h);
            let (x4, y4) = base_transform.apply(vb_x, vb_y + vb_h);

            let clip_polygon = vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4)];
            
            // Calculate bbox for the clip
            let min_x = x1.min(x2).min(x3).min(x4);
            let max_x = x1.max(x2).max(x3).max(x4);
            let min_y = y1.min(y2).min(y3).min(y4);
            let max_y = y1.max(y2).max(y3).max(y4);

            ctx.active_clip = Some(vec![clip_polygon]);
            ctx.active_clip_bbox = Some((min_x, min_y, max_x, max_y));
        }

        // First pass: collect all gradients from the entire document
        for child in root.children() {
            collect_all_gradients(&mut ctx, &child);
        }

        // Also collect all patterns from the entire document
        for child in root.children() {
            collect_all_patterns(&mut ctx, &child);
        }

        // Also collect all markers from the entire document
        for child in root.children() {
            collect_all_markers(&mut ctx, &child);
        }

        // Collect all filters from the entire document
        for child in root.children() {
            collect_all_filters(&mut ctx, &child);
        }

        // Second pass: collect clipPaths and masks
        for child in root.children() {
            collect_clip_paths_and_masks(&mut ctx, &child, &base_transform, &root);
        }

        // Third pass: render tree
        let base_style = Style::new();
        let root_style = parse_style(&root, &base_style);

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
