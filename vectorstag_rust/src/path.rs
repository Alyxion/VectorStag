//! SVG path parsing and sampling operations

use pyo3::prelude::*;
use pyo3::types::PyList;

/// SVG Path Command types for return values
#[derive(Debug, Clone)]
pub enum PathCmd {
    M(f64, f64),
    L(f64, f64),
    C(f64, f64, f64, f64, f64, f64),
    Q(f64, f64, f64, f64),
    A(f64, f64, f64, bool, bool, f64, f64),  // rx, ry, rotation, large-arc, sweep, x, y
    Z,
}

/// Sample points along a cubic bezier curve
/// Returns list of (x, y) tuples from t=1/n_samples to t=1 (excludes t=0)
#[pyfunction]
pub fn sample_cubic_bezier(
    x0: f64, y0: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
    x3: f64, y3: f64,
    n_samples: usize,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(n_samples);
    let n = n_samples as f64;

    for i in 1..=n_samples {
        let t = i as f64 / n;
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        let x = mt3 * x0 + 3.0 * mt2 * t * x1 + 3.0 * mt * t2 * x2 + t3 * x3;
        let y = mt3 * y0 + 3.0 * mt2 * t * y1 + 3.0 * mt * t2 * y2 + t3 * y3;
        points.push((x, y));
    }

    points
}

/// Sample points along a quadratic bezier curve
/// Returns list of (x, y) tuples from t=1/n_samples to t=1 (excludes t=0)
#[pyfunction]
pub fn sample_quadratic_bezier(
    x0: f64, y0: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
    n_samples: usize,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(n_samples);
    let n = n_samples as f64;

    for i in 1..=n_samples {
        let t = i as f64 / n;
        let mt = 1.0 - t;

        let x = mt * mt * x0 + 2.0 * mt * t * x1 + t * t * x2;
        let y = mt * mt * y0 + 2.0 * mt * t * y1 + t * t * y2;
        points.push((x, y));
    }

    points
}

/// Parse SVG path data into absolute commands
/// Returns a list of tuples representing commands
#[pyfunction]
pub fn parse_path<'py>(py: Python<'py>, d: &str) -> Bound<'py, PyList> {
    let commands = parse_path_internal(d);
    let result = PyList::empty(py);

    for cmd in commands {
        let tuple = match cmd {
            PathCmd::M(x, y) => ("M", x, y, 0.0, 0.0, 0.0, 0.0),
            PathCmd::L(x, y) => ("L", x, y, 0.0, 0.0, 0.0, 0.0),
            PathCmd::C(x1, y1, x2, y2, x, y) => ("C", x1, y1, x2, y2, x, y),
            PathCmd::Q(x1, y1, x, y) => ("Q", x1, y1, x, y, 0.0, 0.0),
            PathCmd::A(rx, ry, rotation, large_arc, sweep, x, _y) => {
                // Return arc as "A" with parameters encoded
                ("A", rx, ry, rotation, if large_arc { 1.0 } else { 0.0 }, if sweep { 1.0 } else { 0.0 }, x)
            }
            PathCmd::Z => ("Z", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };
        result.append(tuple).unwrap();
    }

    result
}

/// Internal path parsing function (public for use by svg_renderer)
pub fn parse_path_internal(d: &str) -> Vec<PathCmd> {
    let mut commands = Vec::new();
    let mut current_x: f64 = 0.0;
    let mut current_y: f64 = 0.0;
    let mut start_x: f64 = 0.0;
    let mut start_y: f64 = 0.0;
    let mut last_control: Option<(f64, f64)> = None;
    let mut last_cmd: Option<char> = None;

    let tokens = tokenize_path(d);
    let mut i = 0;

    while i < tokens.len() {
        let token = &tokens[i];

        let cmd = if token.len() == 1 && "MmZzLlHhVvCcSsQqTtAa".contains(&token[..]) {
            i += 1;
            token.chars().next().unwrap()
        } else {
            match last_cmd {
                Some('M') => 'L',
                Some('m') => 'l',
                Some(c) => c,
                None => { i += 1; continue; }
            }
        };

        let is_relative = cmd.is_lowercase();
        let cmd_upper = cmd.to_ascii_uppercase();

        match cmd_upper {
            'M' => {
                let nums = get_nums(&tokens, &mut i, 2);
                if nums.len() < 2 { continue; }
                let (mut x, mut y) = (nums[0], nums[1]);
                if is_relative {
                    x += current_x;
                    y += current_y;
                }
                commands.push(PathCmd::M(x, y));
                current_x = x;
                current_y = y;
                start_x = x;
                start_y = y;
                last_control = None;
            }
            'Z' => {
                commands.push(PathCmd::Z);
                current_x = start_x;
                current_y = start_y;
                last_control = None;
            }
            'L' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 2);
                    if nums.len() < 2 { break; }
                    let (mut x, mut y) = (nums[0], nums[1]);
                    if is_relative {
                        x += current_x;
                        y += current_y;
                    }
                    commands.push(PathCmd::L(x, y));
                    current_x = x;
                    current_y = y;
                }
                last_control = None;
            }
            'H' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 1);
                    if nums.is_empty() { break; }
                    let mut x = nums[0];
                    if is_relative {
                        x += current_x;
                    }
                    commands.push(PathCmd::L(x, current_y));
                    current_x = x;
                }
                last_control = None;
            }
            'V' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 1);
                    if nums.is_empty() { break; }
                    let mut y = nums[0];
                    if is_relative {
                        y += current_y;
                    }
                    commands.push(PathCmd::L(current_x, y));
                    current_y = y;
                }
                last_control = None;
            }
            'C' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 6);
                    if nums.len() < 6 { break; }
                    let (mut x1, mut y1, mut x2, mut y2, mut x, mut y) =
                        (nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]);
                    if is_relative {
                        x1 += current_x;
                        y1 += current_y;
                        x2 += current_x;
                        y2 += current_y;
                        x += current_x;
                        y += current_y;
                    }
                    commands.push(PathCmd::C(x1, y1, x2, y2, x, y));
                    last_control = Some((x2, y2));
                    current_x = x;
                    current_y = y;
                }
            }
            'S' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 4);
                    if nums.len() < 4 { break; }
                    let (mut x2, mut y2, mut x, mut y) = (nums[0], nums[1], nums[2], nums[3]);
                    if is_relative {
                        x2 += current_x;
                        y2 += current_y;
                        x += current_x;
                        y += current_y;
                    }

                    let (x1, y1) = if let Some((lx, ly)) = last_control {
                        if matches!(last_cmd, Some('C') | Some('c') | Some('S') | Some('s')) {
                            (2.0 * current_x - lx, 2.0 * current_y - ly)
                        } else {
                            (current_x, current_y)
                        }
                    } else {
                        (current_x, current_y)
                    };

                    commands.push(PathCmd::C(x1, y1, x2, y2, x, y));
                    last_control = Some((x2, y2));
                    current_x = x;
                    current_y = y;
                }
            }
            'Q' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 4);
                    if nums.len() < 4 { break; }
                    let (mut x1, mut y1, mut x, mut y) = (nums[0], nums[1], nums[2], nums[3]);
                    if is_relative {
                        x1 += current_x;
                        y1 += current_y;
                        x += current_x;
                        y += current_y;
                    }
                    commands.push(PathCmd::Q(x1, y1, x, y));
                    last_control = Some((x1, y1));
                    current_x = x;
                    current_y = y;
                }
            }
            'T' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 2);
                    if nums.len() < 2 { break; }
                    let (mut x, mut y) = (nums[0], nums[1]);
                    if is_relative {
                        x += current_x;
                        y += current_y;
                    }

                    let (x1, y1) = if let Some((lx, ly)) = last_control {
                        if matches!(last_cmd, Some('Q') | Some('q') | Some('T') | Some('t')) {
                            (2.0 * current_x - lx, 2.0 * current_y - ly)
                        } else {
                            (current_x, current_y)
                        }
                    } else {
                        (current_x, current_y)
                    };

                    commands.push(PathCmd::Q(x1, y1, x, y));
                    last_control = Some((x1, y1));
                    current_x = x;
                    current_y = y;
                }
            }
            'A' => {
                loop {
                    let nums = get_nums(&tokens, &mut i, 7);
                    if nums.len() < 7 { break; }
                    let (rx, ry, x_rot, large_arc, sweep, mut x, mut y) =
                        (nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], nums[6]);
                    if is_relative {
                        x += current_x;
                        y += current_y;
                    }

                    let arc_cmds = arc_to_bezier(
                        current_x, current_y,
                        rx, ry, x_rot,
                        large_arc as i32, sweep as i32,
                        x, y
                    );
                    commands.extend(arc_cmds);
                    current_x = x;
                    current_y = y;
                }
                last_control = None;
            }
            _ => {}
        }

        last_cmd = Some(cmd);
    }

    commands
}

/// Tokenize path data into commands and numbers
fn tokenize_path(d: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = d.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i] as char;

        if c.is_whitespace() || c == ',' {
            i += 1;
            continue;
        }

        if "MmZzLlHhVvCcSsQqTtAa".contains(c) {
            tokens.push(c.to_string());
            i += 1;
            continue;
        }

        if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() {
            let start = i;

            if bytes[i] as char == '-' || bytes[i] as char == '+' {
                i += 1;
            }

            while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                i += 1;
            }

            if i < bytes.len() && bytes[i] as char == '.' {
                i += 1;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
            }

            if i < bytes.len() && (bytes[i] as char == 'e' || bytes[i] as char == 'E') {
                i += 1;
                if i < bytes.len() && (bytes[i] as char == '-' || bytes[i] as char == '+') {
                    i += 1;
                }
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
            }

            if i > start {
                tokens.push(d[start..i].to_string());
            }
            continue;
        }

        i += 1;
    }

    tokens
}

/// Get numbers from tokens
fn get_nums(tokens: &[String], i: &mut usize, count: usize) -> Vec<f64> {
    let mut nums = Vec::with_capacity(count);

    while nums.len() < count && *i < tokens.len() {
        if let Ok(n) = tokens[*i].parse::<f64>() {
            nums.push(n);
            *i += 1;
        } else {
            break;
        }
    }

    nums
}

/// Convert SVG arc to cubic bezier curves
fn arc_to_bezier(x1: f64, y1: f64, rx: f64, ry: f64,
                 phi: f64, large_arc: i32, sweep: i32,
                 x2: f64, y2: f64) -> Vec<PathCmd> {
    let mut commands = Vec::new();

    if (x1 - x2).abs() < 1e-10 && (y1 - y2).abs() < 1e-10 {
        return commands;
    }

    if rx == 0.0 || ry == 0.0 {
        return vec![PathCmd::L(x2, y2)];
    }

    let mut rx = rx.abs();
    let mut ry = ry.abs();

    let phi_rad = phi.to_radians();
    let cos_phi = phi_rad.cos();
    let sin_phi = phi_rad.sin();

    let dx = (x1 - x2) / 2.0;
    let dy = (y1 - y2) / 2.0;
    let x1p = cos_phi * dx + sin_phi * dy;
    let y1p = -sin_phi * dx + cos_phi * dy;

    let lambda_sq = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if lambda_sq > 1.0 {
        let lambda_val = lambda_sq.sqrt();
        rx *= lambda_val;
        ry *= lambda_val;
    }

    let num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
    let den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
    let sq = if den > 0.0 { (num / den).max(0.0).sqrt() } else { 0.0 };

    let sq = if large_arc == sweep { -sq } else { sq };

    let cxp = sq * rx * y1p / ry;
    let cyp = -sq * ry * x1p / rx;

    let cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0;
    let cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0;

    fn angle(ux: f64, uy: f64, vx: f64, vy: f64) -> f64 {
        let n = (ux * ux + uy * uy).sqrt() * (vx * vx + vy * vy).sqrt();
        if n == 0.0 { return 0.0; }
        let c = (ux * vx + uy * vy) / n;
        let c = c.max(-1.0).min(1.0);
        let sign = if ux * vy - uy * vx >= 0.0 { 1.0 } else { -1.0 };
        sign * c.acos()
    }

    let theta1 = angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry);
    let mut dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                          (-x1p - cxp) / rx, (-y1p - cyp) / ry);

    if sweep == 0 && dtheta > 0.0 {
        dtheta -= 2.0 * std::f64::consts::PI;
    } else if sweep == 1 && dtheta < 0.0 {
        dtheta += 2.0 * std::f64::consts::PI;
    }

    let n_segs = ((dtheta.abs() / (std::f64::consts::PI / 2.0)).ceil() as usize).max(1);
    let d_theta = dtheta / n_segs as f64;

    let mut t = theta1;
    for _ in 0..n_segs {
        let t2 = t + d_theta;

        let half_d = d_theta / 2.0;
        let tan_half = half_d.tan();
        let alpha = d_theta.sin() * ((4.0 + 3.0 * tan_half * tan_half).sqrt() - 1.0) / 3.0;

        let cos_t = t.cos();
        let sin_t = t.sin();
        let x_start = cx + rx * cos_phi * cos_t - ry * sin_phi * sin_t;
        let y_start = cy + rx * sin_phi * cos_t + ry * cos_phi * sin_t;

        let cos_t2 = t2.cos();
        let sin_t2 = t2.sin();
        let x_end = cx + rx * cos_phi * cos_t2 - ry * sin_phi * sin_t2;
        let y_end = cy + rx * sin_phi * cos_t2 + ry * cos_phi * sin_t2;

        let dx_start = -rx * cos_phi * sin_t - ry * sin_phi * cos_t;
        let dy_start = -rx * sin_phi * sin_t + ry * cos_phi * cos_t;
        let dx_end = -rx * cos_phi * sin_t2 - ry * sin_phi * cos_t2;
        let dy_end = -rx * sin_phi * sin_t2 + ry * cos_phi * cos_t2;

        let cp1x = x_start + alpha * dx_start;
        let cp1y = y_start + alpha * dy_start;
        let cp2x = x_end - alpha * dx_end;
        let cp2y = y_end - alpha * dy_end;

        commands.push(PathCmd::C(cp1x, cp1y, cp2x, cp2y, x_end, y_end));

        t = t2;
    }

    commands
}

/// Sample elliptical arc points
/// Returns list of (x, y) tuples
#[pyfunction]
pub fn sample_arc(
    cx: f64, cy: f64,      // Center
    rx: f64, ry: f64,      // Radii
    start_angle: f64,      // Start angle in radians
    end_angle: f64,        // End angle in radians
    rotation: f64,         // X-axis rotation in radians
    n_samples: usize,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(n_samples);
    let cos_rot = rotation.cos();
    let sin_rot = rotation.sin();

    for i in 1..=n_samples {
        let t = i as f64 / n_samples as f64;
        let angle = start_angle + t * (end_angle - start_angle);

        let px = rx * angle.cos();
        let py = ry * angle.sin();

        let x = cx + px * cos_rot - py * sin_rot;
        let y = cy + px * sin_rot + py * cos_rot;
        points.push((x, y));
    }

    points
}

/// Register path module functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_cubic_bezier, m)?)?;
    m.add_function(wrap_pyfunction!(sample_quadratic_bezier, m)?)?;
    m.add_function(wrap_pyfunction!(sample_arc, m)?)?;
    m.add_function(wrap_pyfunction!(parse_path, m)?)?;
    Ok(())
}
