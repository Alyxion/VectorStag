//! Render context implementation.

use std::collections::HashMap;
use std::sync::Arc;
use crate::text::FontManager;
use super::types::*;

impl RenderContext {
    pub fn new(width: usize, height: usize, background: Color, antialias: u32, font_manager: Arc<FontManager>) -> Self {
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
            patterns: HashMap::new(),
            clip_paths: HashMap::new(),
            masks: HashMap::new(),
            markers: HashMap::new(),
            filters: HashMap::new(),
            antialias,
            shapes_rendered: 0,
            active_clip: None,
            active_clip_bbox: None,
            font_manager,
            viewport_width: render_width as f64,
            viewport_height: render_height as f64,
            viewbox_scale_x: 1.0,  // Will be set after viewbox transform is computed
            viewbox_scale_y: 1.0,
        }
    }

    pub fn can_render_more(&self) -> bool {
        self.shapes_rendered < MAX_SHAPES
    }

    pub fn increment_shapes(&mut self) {
        self.shapes_rendered += 1;
    }

    /// Check if a point is inside the active clip path
    pub fn is_inside_clip(&self, x: f64, y: f64) -> bool {
        match &self.active_clip {
            None => true,
            Some(clip_polygons) => {
                if let Some((min_x, min_y, max_x, max_y)) = self.active_clip_bbox {
                    if x < min_x || x > max_x || y < min_y || y > max_y {
                        return false;
                    }
                }

                for polygon in clip_polygons {
                    if polygon.len() < 3 {
                        continue;
                    }
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

    pub fn blend_pixel(&mut self, x: usize, y: usize, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }

        if !self.is_inside_clip(x as f64 + 0.5, y as f64 + 0.5) {
            return;
        }

        let idx = (y * self.width + x) * 4;
        let src_a = color.a as f32 / 255.0;

        if src_a >= 1.0 {
            self.buffer[idx] = color.r;
            self.buffer[idx + 1] = color.g;
            self.buffer[idx + 2] = color.b;
            self.buffer[idx + 3] = 255;
        } else if src_a > 0.0 {
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

    pub fn fill_polygon(&mut self, points: &[(f64, f64)], color: Color, fill_rule: FillRule) {
        if points.len() < 3 || color.a == 0 {
            return;
        }

        let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

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

    /// Fill multiple polygons as a compound shape (for paths with holes)
    /// All subpaths are processed together so fill-rule works correctly
    pub fn fill_compound_polygon(&mut self, polygons: &[Vec<(f64, f64)>], color: Color, fill_rule: FillRule) {
        if color.a == 0 {
            return;
        }

        // Compute bounding box across all polygons
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut total_points = 0;

        for poly in polygons {
            if poly.len() < 3 {
                continue;
            }
            total_points += poly.len();
            for &(x, y) in poly {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }

        if total_points < 3 || total_points > MAX_POLYGON_POINTS * polygons.len() {
            return;
        }

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

        // Collect edges from ALL polygons
        let mut edges: Vec<(f64, f64, f64, f64, i32)> = Vec::new();
        for poly in polygons {
            let n = poly.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let j = (i + 1) % n;
                let (x1, y1) = poly[i];
                let (x2, y2) = poly[j];

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
        }

        // Process all edges together in scanline fill
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
    pub fn interpolate_gradient_color(stops: &[(f64, u8, u8, u8, u8)], t: f64) -> Color {
        if stops.is_empty() {
            return Color::from_rgba(0, 0, 0, 255);
        }
        if stops.len() == 1 {
            let s = &stops[0];
            return Color::from_rgba(s.1, s.2, s.3, s.4);
        }

        let t = t.clamp(0.0, 1.0);

        let mut prev_stop = &stops[0];
        for stop in stops.iter() {
            if stop.0 >= t {
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

        let last = stops.last().unwrap();
        Color::from_rgba(last.1, last.2, last.3, last.4)
    }

    /// Fill polygon with a gradient
    pub fn fill_polygon_gradient(&mut self, points: &[(f64, f64)], gradient: &GradientDef,
                             transform: &Transform, fill_rule: FillRule, opacity: f64) {
        if points.len() < 3 {
            return;
        }

        let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

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

        let (gx1, gy1, gx2, gy2, gcx, gcy, gr) = if gradient.user_space {
            let combined_transform = transform.multiply(&gradient.transform);
            let (gx1, gy1) = combined_transform.apply(gradient.x1, gradient.y1);
            let (gx2, gy2) = combined_transform.apply(gradient.x2, gradient.y2);
            let (gcx, gcy) = combined_transform.apply(gradient.cx, gradient.cy);
            let scale = ((combined_transform.a * combined_transform.a + combined_transform.b * combined_transform.b).sqrt() +
                        (combined_transform.c * combined_transform.c + combined_transform.d * combined_transform.d).sqrt()) / 2.0;
            let gr = gradient.r * scale;
            (gx1, gy1, gx2, gy2, gcx, gcy, gr)
        } else {
            let bbox_w = max_x - min_x;
            let bbox_h = max_y - min_y;
            let normalize = |v: f64| if v > 1.0 { v / 100.0 } else { v };

            let (tx1, ty1) = gradient.transform.apply(normalize(gradient.x1), normalize(gradient.y1));
            let (tx2, ty2) = gradient.transform.apply(normalize(gradient.x2), normalize(gradient.y2));
            let (tcx, tcy) = gradient.transform.apply(normalize(gradient.cx), normalize(gradient.cy));

            let scale = ((gradient.transform.a * gradient.transform.a + gradient.transform.b * gradient.transform.b).sqrt() +
                        (gradient.transform.c * gradient.transform.c + gradient.transform.d * gradient.transform.d).sqrt()) / 2.0;
            let tr = normalize(gradient.r) * scale;

            let gx1 = min_x + tx1 * bbox_w;
            let gy1 = min_y + ty1 * bbox_h;
            let gx2 = min_x + tx2 * bbox_w;
            let gy2 = min_y + ty2 * bbox_h;
            let gcx = min_x + tcx * bbox_w;
            let gcy = min_y + tcy * bbox_h;
            let gr = tr * bbox_w.max(bbox_h);
            (gx1, gy1, gx2, gy2, gcx, gcy, gr)
        };

        let grad_dx = gx2 - gx1;
        let grad_dy = gy2 - gy1;
        let grad_len_sq = grad_dx * grad_dx + grad_dy * grad_dy;

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

    pub fn downsample(&self, out_width: usize, out_height: usize) -> Vec<u8> {
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

    pub fn fill_polygon_pattern(&mut self, points: &[(f64, f64)], pattern: &PatternDef, fill_rule: FillRule, opacity: f64) {
        if points.len() < 3 || pattern.width <= 0.0 || pattern.height <= 0.0 {
            return;
        }

        let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        if max_x < 0.0 || min_x >= self.width as f64 || max_y < 0.0 || min_y >= self.height as f64 {
            return;
        }

        let y_start = (min_y.floor() as i32).max(0) as usize;
        let y_end = (max_y.ceil() as i32).min(self.height as i32) as usize;
        let x_start = (min_x.floor() as i32).max(0) as usize;
        let x_end = (max_x.ceil() as i32).min(self.width as i32) as usize;
        if y_start >= y_end || x_start >= x_end {
            return;
        }

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
            let (x1, y1, x2, y2, dir) = if y1 < y2 { (x1, y1, x2, y2, 1) } else { (x2, y2, x1, y1, -1) };
            edges.push((x1, y1, x2, y2, dir));
        }

        // Scale factors for userSpaceOnUse: convert pixel coords to viewbox coords
        let scale_x = self.viewbox_scale_x;
        let scale_y = self.viewbox_scale_y;

        let sample_color = |px: f64, py: f64| -> Color {
            // userSpaceOnUse: transform pixel coordinates back to viewbox coordinates
            let vx = px / scale_x;
            let vy = py / scale_y;

            let mut x = vx - pattern.x;
            let mut y = vy - pattern.y;
            x = x.rem_euclid(pattern.width);
            y = y.rem_euclid(pattern.height);

            let mut out = Color::from_rgba(0, 0, 0, 0);
            for r in &pattern.rects {
                if x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height {
                    out = r.color;
                }
            }
            out
        };

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
                                    let mut c = sample_color(px as f64 + 0.5, scan_y);
                                    c.a = (c.a as f64 * opacity) as u8;
                                    self.blend_pixel(px, y, c);
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
                                    let mut c = sample_color(px as f64 + 0.5, scan_y);
                                    c.a = (c.a as f64 * opacity) as u8;
                                    self.blend_pixel(px, y, c);
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
}
