//! Marker rendering for SVG paths.

use roxmltree::Node;
use super::types::*;
use super::path_utils::find_element_by_id;

/// Render markers on a path's vertices
pub fn render_markers(
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
            let (x1, y1) = points[0];
            let (x2, y2) = points[1];
            (y2 - y1).atan2(x2 - x1)
        } else if i == points.len() - 1 {
            let (x1, y1) = points[points.len() - 2];
            let (x2, y2) = points[points.len() - 1];
            (y2 - y1).atan2(x2 - x1)
        } else {
            let (x0, y0) = points[i - 1];
            let (x1, y1) = points[i];
            let (x2, y2) = points[i + 1];
            let a1 = (y1 - y0).atan2(x1 - x0);
            let a2 = (y2 - y1).atan2(x2 - x1);
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
pub fn render_single_marker(
    ctx: &mut RenderContext,
    marker_def: &MarkerDef,
    x: f64,
    y: f64,
    angle: f64,
    stroke_width: f64,
    root: &Node,
    marker_id: &str,
) {
    if let Some(marker_elem) = find_element_by_id(root, marker_id) {
        let scale = if marker_def.stroke_width_units {
            stroke_width
        } else {
            1.0
        };

        let marker_transform = if let Some((vb_x, vb_y, vb_w, vb_h)) = marker_def.viewbox {
            let sx = marker_def.marker_width / vb_w;
            let sy = marker_def.marker_height / vb_h;
            let s = sx.min(sy) * scale;

            Transform::translate(x, y)
                .multiply(&Transform::rotate(angle))
                .multiply(&Transform::scale(s, s))
                .multiply(&Transform::translate(-marker_def.ref_x, -marker_def.ref_y))
        } else {
            Transform::translate(x, y)
                .multiply(&Transform::rotate(angle))
                .multiply(&Transform::scale(scale, scale))
                .multiply(&Transform::translate(-marker_def.ref_x, -marker_def.ref_y))
        };

        let base_style = Style::new();
        for child in marker_elem.children() {
            super::render::render_node(ctx, &child, &marker_transform, &base_style, 0, root);
        }
    }
}
