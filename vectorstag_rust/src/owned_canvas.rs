//! Rust-owned Canvas that eliminates Python↔Rust boundary crossings.
//!
//! The Canvas owns its pixel buffer and exposes all drawing operations as methods.
//! This avoids passing numpy arrays back and forth for each operation.

use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};
use ndarray::Array3;

/// A Rust-owned RGBA canvas for high-performance rendering.
/// All operations happen in Rust memory without GIL overhead.
#[pyclass]
pub struct OwnedCanvas {
    width: usize,
    height: usize,
    /// RGBA pixel data, row-major: data[y * width * 4 + x * 4 + channel]
    data: Vec<u8>,
}

#[pymethods]
impl OwnedCanvas {
    /// Create a new canvas with transparent background
    #[new]
    #[pyo3(signature = (width, height, background=None))]
    fn new(width: usize, height: usize, background: Option<(u8, u8, u8, u8)>) -> Self {
        let size = width * height * 4;
        let data = if let Some((r, g, b, a)) = background {
            let mut v = vec![0u8; size];
            for i in 0..(width * height) {
                v[i * 4] = r;
                v[i * 4 + 1] = g;
                v[i * 4 + 2] = b;
                v[i * 4 + 3] = a;
            }
            v
        } else {
            vec![0u8; size]
        };

        OwnedCanvas { width, height, data }
    }

    /// Get canvas dimensions
    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Clear the canvas to transparent
    fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Fill a rectangle with a solid color
    fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, r: u8, g: u8, b: u8, a: u8) {
        let x_start = x.max(0) as usize;
        let y_start = y.max(0) as usize;
        let x_end = ((x + w) as usize).min(self.width);
        let y_end = ((y + h) as usize).min(self.height);

        for py in y_start..y_end {
            for px in x_start..x_end {
                let idx = (py * self.width + px) * 4;
                self.blend_pixel(idx, r, g, b, a);
            }
        }
    }

    /// Fill a polygon with analytical antialiasing
    fn fill_polygon_aa(&mut self, points: Vec<(f32, f32)>, r: u8, g: u8, b: u8, a: u8) {
        if points.len() < 3 {
            return;
        }

        // Use the analytical AA algorithm from canvas.rs
        let edges = Self::build_edges(&points);
        if edges.is_empty() {
            return;
        }

        // Find bounding box
        let (min_y, max_y) = edges.iter().fold((f32::MAX, f32::MIN), |(min, max), e| {
            (min.min(e.y_top), max.max(e.y_bottom))
        });

        let start_y = (min_y.floor() as i32).max(0);
        let end_y = (max_y.ceil() as i32).min(self.height as i32);

        let mut coverage = vec![0.0f32; self.width];

        for y in start_y..end_y {
            coverage.fill(0.0);
            Self::process_scanline(&edges, y, &mut coverage, self.width);

            // Apply coverage to pixels
            let mut accum = 0.0f32;
            for x in 0..self.width {
                accum += coverage[x];
                let cov = accum.abs().min(1.0);
                if cov > 0.001 {
                    let idx = (y as usize * self.width + x) * 4;
                    let alpha = (a as f32 * cov) as u8;
                    self.blend_pixel(idx, r, g, b, alpha);
                }
            }
        }
    }

    /// Alpha composite another canvas onto this one at the given offset
    fn composite(&mut self, other: &OwnedCanvas, offset_x: i32, offset_y: i32) {
        let start_x = offset_x.max(0) as usize;
        let start_y = offset_y.max(0) as usize;
        let end_x = ((offset_x + other.width as i32) as usize).min(self.width);
        let end_y = ((offset_y + other.height as i32) as usize).min(self.height);

        let src_start_x = (-offset_x).max(0) as usize;
        let src_start_y = (-offset_y).max(0) as usize;

        for dy in start_y..end_y {
            let sy = src_start_y + (dy - start_y);
            if sy >= other.height { break; }

            for dx in start_x..end_x {
                let sx = src_start_x + (dx - start_x);
                if sx >= other.width { break; }

                let src_idx = (sy * other.width + sx) * 4;
                let dst_idx = (dy * self.width + dx) * 4;

                let src_a = other.data[src_idx + 3];
                if src_a == 0 { continue; }

                self.blend_pixel(
                    dst_idx,
                    other.data[src_idx],
                    other.data[src_idx + 1],
                    other.data[src_idx + 2],
                    src_a,
                );
            }
        }
    }

    /// Apply a grayscale mask to this canvas's alpha channel (in-place)
    fn apply_mask(&mut self, mask: numpy::PyReadonlyArray2<'_, u8>) {
        let mask_arr = mask.as_array();
        let (mask_h, mask_w) = (mask_arr.shape()[0], mask_arr.shape()[1]);

        if mask_h != self.height || mask_w != self.width {
            return;
        }

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) * 4;
                let img_alpha = self.data[idx + 3] as u32;
                let mask_val = mask_arr[[y, x]] as u32;
                self.data[idx + 3] = ((img_alpha * mask_val) / 255) as u8;
            }
        }
    }

    /// Apply mask and composite onto another canvas in one pass
    fn apply_mask_and_composite_to(
        &self,
        dst: &mut OwnedCanvas,
        mask: numpy::PyReadonlyArray2<'_, u8>,
        offset_x: i32,
        offset_y: i32,
    ) {
        let mask_arr = mask.as_array();

        let start_x = offset_x.max(0) as usize;
        let start_y = offset_y.max(0) as usize;
        let end_x = ((offset_x + self.width as i32) as usize).min(dst.width);
        let end_y = ((offset_y + self.height as i32) as usize).min(dst.height);

        let src_start_x = (-offset_x).max(0) as usize;
        let src_start_y = (-offset_y).max(0) as usize;

        for dy in start_y..end_y {
            let sy = src_start_y + (dy - start_y);
            if sy >= self.height { break; }

            for dx in start_x..end_x {
                let sx = src_start_x + (dx - start_x);
                if sx >= self.width { break; }

                let src_idx = (sy * self.width + sx) * 4;
                let dst_idx = (dy * dst.width + dx) * 4;

                // Apply mask to source alpha
                let src_a_orig = self.data[src_idx + 3] as u32;
                let mask_val = mask_arr[[sy, sx]] as u32;
                let src_a = ((src_a_orig * mask_val) / 255) as u8;

                if src_a == 0 { continue; }

                dst.blend_pixel(
                    dst_idx,
                    self.data[src_idx],
                    self.data[src_idx + 1],
                    self.data[src_idx + 2],
                    src_a,
                );
            }
        }
    }

    /// Convert to numpy array (only call at the end!)
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        let mut arr = Array3::<u8>::zeros((self.height, self.width, 4));
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) * 4;
                arr[[y, x, 0]] = self.data[idx];
                arr[[y, x, 1]] = self.data[idx + 1];
                arr[[y, x, 2]] = self.data[idx + 2];
                arr[[y, x, 3]] = self.data[idx + 3];
            }
        }
        arr.into_pyarray(py)
    }

    /// Get a pixel value
    fn get_pixel(&self, x: usize, y: usize) -> (u8, u8, u8, u8) {
        if x >= self.width || y >= self.height {
            return (0, 0, 0, 0);
        }
        let idx = (y * self.width + x) * 4;
        (self.data[idx], self.data[idx + 1], self.data[idx + 2], self.data[idx + 3])
    }

    /// Set a pixel value directly (no blending)
    fn set_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8, a: u8) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) * 4;
        self.data[idx] = r;
        self.data[idx + 1] = g;
        self.data[idx + 2] = b;
        self.data[idx + 3] = a;
    }
}

/// Edge structure for AA polygon filling
struct CanvasEdge {
    x_top: f32,
    y_top: f32,
    y_bottom: f32,
    dx_per_dy: f32,
    direction: i8,
}

// Private implementation methods
impl OwnedCanvas {
    /// Alpha blend a pixel at the given index
    #[inline]
    fn blend_pixel(&mut self, idx: usize, r: u8, g: u8, b: u8, a: u8) {
        if a == 0 { return; }

        if a == 255 {
            self.data[idx] = r;
            self.data[idx + 1] = g;
            self.data[idx + 2] = b;
            self.data[idx + 3] = 255;
        } else {
            let src_a = a as u32;
            let dst_a = self.data[idx + 3] as u32;
            let inv_src_a = 255 - src_a;
            let out_a = src_a + (dst_a * inv_src_a / 255);

            if out_a > 0 {
                let blend = |s: u8, d: u8| -> u8 {
                    ((s as u32 * src_a + d as u32 * dst_a * inv_src_a / 255) / out_a).min(255) as u8
                };
                self.data[idx] = blend(r, self.data[idx]);
                self.data[idx + 1] = blend(g, self.data[idx + 1]);
                self.data[idx + 2] = blend(b, self.data[idx + 2]);
                self.data[idx + 3] = out_a.min(255) as u8;
            }
        }
    }

    fn build_edges(points: &[(f32, f32)]) -> Vec<CanvasEdge> {
        let mut edges = Vec::new();
        let n = points.len();

        for i in 0..n {
            let (x1, y1) = points[i];
            let (x2, y2) = points[(i + 1) % n];

            let dy = y2 - y1;
            if dy.abs() < 1e-6 { continue; }

            let dx = x2 - x1;
            let dx_per_dy = dx / dy;

            if y1 < y2 {
                edges.push(CanvasEdge {
                    x_top: x1, y_top: y1, y_bottom: y2,
                    dx_per_dy, direction: 1,
                });
            } else {
                edges.push(CanvasEdge {
                    x_top: x2, y_top: y2, y_bottom: y1,
                    dx_per_dy, direction: -1,
                });
            }
        }
        edges
    }

    fn process_scanline(edges: &[CanvasEdge], y: i32, coverage: &mut [f32], width: usize) {
        let yf = y as f32;
        let y_next = yf + 1.0;

        for edge in edges {
            if edge.y_bottom <= yf || edge.y_top >= y_next {
                continue;
            }

            let y_enter = edge.y_top.max(yf);
            let y_exit = edge.y_bottom.min(y_next);
            let height = y_exit - y_enter;

            let x_at_enter = edge.x_top + (y_enter - edge.y_top) * edge.dx_per_dy;
            let x_at_exit = edge.x_top + (y_exit - edge.y_top) * edge.dx_per_dy;

            let x_left = x_at_enter.min(x_at_exit);
            let x_right = x_at_enter.max(x_at_exit);

            let px_left = (x_left.floor() as i32).max(0) as usize;
            let px_right = ((x_right.ceil() as i32) as usize).min(width);

            let dir = edge.direction as f32;

            for px in px_left..px_right {
                let px_left_edge = px as f32;
                let px_right_edge = px_left_edge + 1.0;

                let cov = if x_right <= px_left_edge {
                    0.0
                } else if x_left >= px_right_edge {
                    height * dir
                } else {
                    let left_in = (x_left - px_left_edge).max(0.0).min(1.0);
                    let right_in = (x_right - px_left_edge).max(0.0).min(1.0);
                    let area = (right_in - left_in + (1.0 - right_in)) * height;
                    area * dir
                };

                if px < width {
                    coverage[px] += cov;
                }
            }
        }
    }
}

/// Register the OwnedCanvas class
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OwnedCanvas>()?;
    Ok(())
}
