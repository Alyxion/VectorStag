//! Main SVG renderer module exposing Python bindings
//!
//! This module provides the SvgRenderer class that uses resvg for
//! high-performance SVG rendering entirely in Rust.

use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};
use std::sync::Arc;
use usvg::{TreeParsing, TreeTextToPath};

/// High-performance SVG renderer using resvg
#[pyclass]
pub struct SvgRenderer {
    fontdb: Arc<fontdb::Database>,
}

#[pymethods]
impl SvgRenderer {
    /// Create a new SVG renderer with system fonts loaded
    #[new]
    fn new() -> PyResult<Self> {
        let mut fontdb = fontdb::Database::new();
        fontdb.load_system_fonts();

        Ok(Self {
            fontdb: Arc::new(fontdb),
        })
    }

    /// Render SVG content to a numpy array
    ///
    /// Args:
    ///     svg_content: SVG string content
    ///     width: Output width (optional, uses SVG default if None)
    ///     height: Output height (optional, uses SVG default if None)
    ///     scale: Scale factor (default 1.0)
    ///     background: Background color as (r, g, b, a) tuple
    ///     antialias: Antialiasing factor (default 4, used for supersampling)
    ///
    /// Returns:
    ///     RGBA numpy array of shape (height, width, 4)
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
        let scale = scale.unwrap_or(1.0);
        let antialias = antialias.unwrap_or(4) as u32;
        let background = background.unwrap_or((255, 255, 255, 255));

        // Parse SVG with usvg
        let opt = usvg::Options::default();

        let mut tree = usvg::Tree::from_str(svg_content, &opt)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to parse SVG: {}", e)
            ))?;

        // Convert text to paths using fontdb
        tree.convert_text(&self.fontdb);

        // Calculate output dimensions
        let svg_size = tree.size;
        let (out_width, out_height) = match (width, height) {
            (Some(w), Some(h)) => (w, h),
            (Some(w), None) => {
                let aspect = svg_size.height() / svg_size.width();
                (w, (w as f32 * aspect) as u32)
            }
            (None, Some(h)) => {
                let aspect = svg_size.width() / svg_size.height();
                ((h as f32 * aspect) as u32, h)
            }
            (None, None) => (
                (svg_size.width() * scale) as u32,
                (svg_size.height() * scale) as u32,
            ),
        };

        // Ensure minimum size
        let out_width = out_width.max(1);
        let out_height = out_height.max(1);

        // Render at higher resolution for antialiasing
        let render_width = out_width * antialias;
        let render_height = out_height * antialias;

        // Create pixmap for rendering
        let mut pixmap = tiny_skia::Pixmap::new(render_width, render_height)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Failed to create pixmap"
            ))?;

        // Fill with background color (premultiplied alpha)
        let bg_a = background.3 as f32 / 255.0;
        pixmap.fill(tiny_skia::Color::from_rgba8(
            (background.0 as f32 * bg_a) as u8,
            (background.1 as f32 * bg_a) as u8,
            (background.2 as f32 * bg_a) as u8,
            background.3,
        ));

        // Calculate transform to fit SVG into render area
        let scale_x = render_width as f32 / svg_size.width();
        let scale_y = render_height as f32 / svg_size.height();
        let transform = tiny_skia::Transform::from_scale(scale_x, scale_y);

        // Convert to resvg tree and render
        let rtree = resvg::Tree::from_usvg(&tree);
        rtree.render(transform, &mut pixmap.as_mut());

        // Get pixel data (premultiplied RGBA)
        let mut buffer = pixmap.take();

        // Convert from premultiplied to straight alpha
        unpremultiply_alpha(&mut buffer);

        // Downsample if antialiasing was used
        let final_buffer = if antialias > 1 {
            downsample_box(&buffer, render_width as usize, render_height as usize,
                          out_width as usize, out_height as usize)
        } else {
            buffer
        };

        // Convert to numpy array
        let arr = ndarray::Array3::from_shape_vec(
            (out_height as usize, out_width as usize, 4),
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

/// Convert premultiplied alpha to straight alpha
fn unpremultiply_alpha(buffer: &mut [u8]) {
    for chunk in buffer.chunks_exact_mut(4) {
        let a = chunk[3];
        if a > 0 && a < 255 {
            let a_f = a as f32 / 255.0;
            chunk[0] = (chunk[0] as f32 / a_f).min(255.0) as u8;
            chunk[1] = (chunk[1] as f32 / a_f).min(255.0) as u8;
            chunk[2] = (chunk[2] as f32 / a_f).min(255.0) as u8;
        }
    }
}

/// Box filter downsampling for antialiasing
fn downsample_box(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_width * dst_height * 4];
    let scale_x = src_width / dst_width;
    let scale_y = src_height / dst_height;
    let area = (scale_x * scale_y) as u32;

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let mut r: u32 = 0;
            let mut g: u32 = 0;
            let mut b: u32 = 0;
            let mut a: u32 = 0;

            for sy in 0..scale_y {
                for sx in 0..scale_x {
                    let src_x = dx * scale_x + sx;
                    let src_y = dy * scale_y + sy;
                    let src_idx = (src_y * src_width + src_x) * 4;

                    r += src[src_idx] as u32;
                    g += src[src_idx + 1] as u32;
                    b += src[src_idx + 2] as u32;
                    a += src[src_idx + 3] as u32;
                }
            }

            let dst_idx = (dy * dst_width + dx) * 4;
            dst[dst_idx] = (r / area) as u8;
            dst[dst_idx + 1] = (g / area) as u8;
            dst[dst_idx + 2] = (b / area) as u8;
            dst[dst_idx + 3] = (a / area) as u8;
        }
    }

    dst
}

/// Register the renderer module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SvgRenderer>()?;
    Ok(())
}
