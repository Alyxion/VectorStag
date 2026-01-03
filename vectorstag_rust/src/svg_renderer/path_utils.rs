//! Path conversion utilities.

use roxmltree::Node;
use crate::path::PathCmd;
use super::types::Transform;

/// Convert path commands to list of polygons
pub fn commands_to_polygons(commands: &[PathCmd], transform: &Transform) -> Vec<Vec<(f64, f64)>> {
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
                let p = transform.apply(*x, *y);
                if current_poly.is_empty() {
                    current_poly.push(last_point);
                }
                current_poly.push(p);
                last_point = p;
            }
            PathCmd::Z => {
                if !current_poly.is_empty() {
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
pub fn path_to_polygons(d: &str, transform: &Transform) -> Vec<Vec<(f64, f64)>> {
    let commands = crate::path::parse_path_internal(d);
    commands_to_polygons(&commands, transform)
}

/// Find an element by ID in the document tree
pub fn find_element_by_id<'a>(root: &Node<'a, '_>, id: &str) -> Option<Node<'a, 'a>> {
    if root.attribute("id") == Some(id) {
        return Some(*root);
    }

    for desc in root.descendants() {
        if desc.attribute("id") == Some(id) {
            return Some(desc);
        }
    }
    None
}
