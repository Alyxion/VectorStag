//! SVG Filter Primitives implementation

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;

// ============================================================================
// HSL Color Space Helpers
// ============================================================================

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };

    let h = if (max - r).abs() < 1e-6 {
        let mut h = (g - b) / d;
        if g < b { h += 6.0; }
        h
    } else if (max - g).abs() < 1e-6 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };

    (h / 6.0, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-6 {
        return (l, l, l);
    }

    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;

    fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
        if t < 0.0 { t += 1.0; }
        if t > 1.0 { t -= 1.0; }
        if t < 1.0/6.0 { return p + (q - p) * 6.0 * t; }
        if t < 1.0/2.0 { return q; }
        if t < 2.0/3.0 { return p + (q - p) * (2.0/3.0 - t) * 6.0; }
        p
    }

    (hue_to_rgb(p, q, h + 1.0/3.0),
     hue_to_rgb(p, q, h),
     hue_to_rgb(p, q, h - 1.0/3.0))
}

fn luminosity(r: f32, g: f32, b: f32) -> f32 {
    0.3 * r + 0.59 * g + 0.11 * b
}

fn set_lum(r: f32, g: f32, b: f32, l: f32) -> (f32, f32, f32) {
    let d = l - luminosity(r, g, b);
    clip_color(r + d, g + d, b + d)
}

fn clip_color(mut r: f32, mut g: f32, mut b: f32) -> (f32, f32, f32) {
    let l = luminosity(r, g, b);
    let n = r.min(g).min(b);
    let x = r.max(g).max(b);

    if n < 0.0 {
        let d = l - n;
        if d.abs() > 1e-6 {
            r = l + (r - l) * l / d;
            g = l + (g - l) * l / d;
            b = l + (b - l) * l / d;
        }
    }
    if x > 1.0 {
        let d = x - l;
        if d.abs() > 1e-6 {
            r = l + (r - l) * (1.0 - l) / d;
            g = l + (g - l) * (1.0 - l) / d;
            b = l + (b - l) * (1.0 - l) / d;
        }
    }
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

#[allow(dead_code)]
fn saturation(r: f32, g: f32, b: f32) -> f32 {
    r.max(g).max(b) - r.min(g).min(b)
}

#[allow(dead_code)]
fn set_sat(r: f32, g: f32, b: f32, s: f32) -> (f32, f32, f32) {
    let mut vals = [(r, 0), (g, 1), (b, 2)];
    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let (min_v, _) = vals[0];
    let (mid_v, _) = vals[1];
    let (max_v, _) = vals[2];

    let mut result = [0.0f32; 3];

    if (max_v - min_v).abs() > 1e-6 {
        result[vals[1].1] = (mid_v - min_v) * s / (max_v - min_v);
        result[vals[2].1] = s;
    }
    result[vals[0].1] = 0.0;

    (result[0], result[1], result[2])
}

// ============================================================================
// Filter Primitives (Core Implementation)
// ============================================================================

pub fn fe_flood_impl_f32(
    width: usize,
    height: usize,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) -> Array3<f32> {
    let mut dst = Array3::<f32>::zeros((height, width, 4));
    for y in 0..height {
        for x in 0..width {
            dst[[y, x, 0]] = r;
            dst[[y, x, 1]] = g;
            dst[[y, x, 2]] = b;
            dst[[y, x, 3]] = a;
        }
    }
    dst
}

pub fn fe_flood_impl(width: usize, height: usize, r: u8, g: u8, b: u8, a: u8) -> Array3<u8> {
    let mut pixels = Array3::<u8>::zeros((height, width, 4));
    for y in 0..height {
        for x in 0..width {
            pixels[[y, x, 0]] = r;
            pixels[[y, x, 1]] = g;
            pixels[[y, x, 2]] = b;
            pixels[[y, x, 3]] = a;
        }
    }
    pixels
}

/// feFlood - fill entire region with solid color
#[pyfunction]
pub fn fe_flood<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    r: u8, g: u8, b: u8, a: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    fe_flood_impl(width, height, r, g, b, a).into_pyarray(py)
}

#[pyfunction]
pub fn fe_flood_f32<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    r: f32, g: f32, b: f32, a: f32,
) -> Bound<'py, numpy::PyArray3<f32>> {
    fe_flood_impl_f32(width, height, r, g, b, a).into_pyarray(py)
}

pub fn fe_offset_impl_f32(src: &ndarray::ArrayView3<f32>, dx: i32, dy: i32) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        let src_y = y as i32 - dy;
        if src_y < 0 || src_y >= h as i32 {
            continue;
        }
        for x in 0..w {
            let src_x = x as i32 - dx;
            if src_x < 0 || src_x >= w as i32 {
                continue;
            }
            for c in 0..4 {
                dst[[y, x, c]] = src[[src_y as usize, src_x as usize, c]];
            }
        }
    }
    dst
}

pub fn fe_offset_impl(src: &ndarray::ArrayView3<u8>, dx: i32, dy: i32) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        let src_y = y as i32 - dy;
        if src_y < 0 || src_y >= h as i32 {
            continue;
        }
        for x in 0..w {
            let src_x = x as i32 - dx;
            if src_x < 0 || src_x >= w as i32 {
                continue;
            }
            for c in 0..4 {
                dst[[y, x, c]] = src[[src_y as usize, src_x as usize, c]];
            }
        }
    }
    dst
}

/// feOffset - offset image by dx, dy
#[pyfunction]
pub fn fe_offset<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    dx: i32,
    dy: i32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let src_arr = src.as_array();
    fe_offset_impl(&src_arr, dx, dy).into_pyarray(py)
}

pub fn fe_blend_impl_f32(arr1: &ndarray::ArrayView3<f32>, arr2: &ndarray::ArrayView3<f32>, mode: u8) -> Array3<f32> {
    let (h, w, _) = (arr1.shape()[0], arr1.shape()[1], arr1.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let pca1_r = arr1[[y, x, 0]];
            let pca1_g = arr1[[y, x, 1]];
            let pca1_b = arr1[[y, x, 2]];
            let a1 = arr1[[y, x, 3]];

            let pca2_r = arr2[[y, x, 0]];
            let pca2_g = arr2[[y, x, 1]];
            let pca2_b = arr2[[y, x, 2]];
            let a2 = arr2[[y, x, 3]];

            let (r1, g1, b1) = if a1 > 0.0 {
                (pca1_r / a1, pca1_g / a1, pca1_b / a1)
            } else {
                (0.0, 0.0, 0.0)
            };

            let (r2, g2, b2) = if a2 > 0.0 {
                (pca2_r / a2, pca2_g / a2, pca2_b / a2)
            } else {
                (0.0, 0.0, 0.0)
            };

            let (br, bg, bb) = match mode {
                12 => {
                    let (h1, _, _) = rgb_to_hsl(r1, g1, b1);
                    let (_, s2, l2) = rgb_to_hsl(r2, g2, b2);
                    hsl_to_rgb(h1, s2, l2)
                }
                13 => {
                    let (_, s1, _) = rgb_to_hsl(r1, g1, b1);
                    let (h2, _, l2) = rgb_to_hsl(r2, g2, b2);
                    hsl_to_rgb(h2, s1, l2)
                }
                14 => {
                    let (h1, s1, _) = rgb_to_hsl(r1, g1, b1);
                    let l2 = luminosity(r2, g2, b2);
                    let (r, g, b) = hsl_to_rgb(h1, s1, 0.5);
                    set_lum(r, g, b, l2)
                }
                15 => {
                    let l1 = luminosity(r1, g1, b1);
                    set_lum(r2, g2, b2, l1)
                }
                _ => {
                    let blend = |c1: f32, c2: f32| -> f32 {
                        match mode {
                            0 => c1,
                            1 => c1 * c2,
                            2 => 1.0 - (1.0 - c1) * (1.0 - c2),
                            3 => c1.min(c2),
                            4 => c1.max(c2),
                            5 => if c2 < 0.5 { 2.0 * c1 * c2 } else { 1.0 - 2.0 * (1.0 - c1) * (1.0 - c2) },
                            6 => if c2 == 1.0 { c2 } else { (c1 / (1.0 - c2)).min(1.0) },
                            7 => if c2 == 0.0 { 0.0 } else { 1.0 - ((1.0 - c1) / c2).min(1.0) },
                            8 => if c1 < 0.5 { 2.0 * c1 * c2 } else { 1.0 - 2.0 * (1.0 - c1) * (1.0 - c2) },
                            9 => {
                                if c1 < 0.5 {
                                    c2 - (1.0 - 2.0 * c1) * c2 * (1.0 - c2)
                                } else {
                                    let d = if c2 < 0.25 {
                                        ((16.0 * c2 - 12.0) * c2 + 4.0) * c2
                                    } else {
                                        c2.sqrt()
                                    };
                                    c2 + (2.0 * c1 - 1.0) * (d - c2)
                                }
                            }
                            10 => (c1 - c2).abs(),
                            11 => c1 + c2 - 2.0 * c1 * c2,
                            _ => c1,
                        }
                    };
                    (blend(r1, r2), blend(g1, g2), blend(b1, b2))
                }
            };

            let out_a = a1 + a2 * (1.0 - a1);
            let term1 = 1.0 - a2;
            let term2 = 1.0 - a1;
            let term3 = a1 * a2;

            dst[[y, x, 0]] = (pca1_r * term1 + pca2_r * term2 + br * term3).clamp(0.0, 1.0);
            dst[[y, x, 1]] = (pca1_g * term1 + pca2_g * term2 + bg * term3).clamp(0.0, 1.0);
            dst[[y, x, 2]] = (pca1_b * term1 + pca2_b * term2 + bb * term3).clamp(0.0, 1.0);
            dst[[y, x, 3]] = out_a.clamp(0.0, 1.0);
        }
    }
    dst
}

pub fn fe_blend_impl(arr1: &ndarray::ArrayView3<u8>, arr2: &ndarray::ArrayView3<u8>, mode: u8) -> Array3<u8> {
    let (h, w, _) = (arr1.shape()[0], arr1.shape()[1], arr1.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            // Inputs are Premultiplied
            let pca1_r = arr1[[y, x, 0]] as f32 / 255.0;
            let pca1_g = arr1[[y, x, 1]] as f32 / 255.0;
            let pca1_b = arr1[[y, x, 2]] as f32 / 255.0;
            let a1 = arr1[[y, x, 3]] as f32 / 255.0;

            let pca2_r = arr2[[y, x, 0]] as f32 / 255.0;
            let pca2_g = arr2[[y, x, 1]] as f32 / 255.0;
            let pca2_b = arr2[[y, x, 2]] as f32 / 255.0;
            let a2 = arr2[[y, x, 3]] as f32 / 255.0;

            // Unpremultiply for blending function
            let (r1, g1, b1) = if a1 > 0.0 {
                (pca1_r / a1, pca1_g / a1, pca1_b / a1)
            } else {
                (0.0, 0.0, 0.0)
            };

            let (r2, g2, b2) = if a2 > 0.0 {
                (pca2_r / a2, pca2_g / a2, pca2_b / a2)
            } else {
                (0.0, 0.0, 0.0)
            };

            // HSL-based blend modes (12-15)
            let (br, bg, bb) = match mode {
                12 => {  // hue
                    let (h1, _, _) = rgb_to_hsl(r1, g1, b1);
                    let (_, s2, l2) = rgb_to_hsl(r2, g2, b2);
                    hsl_to_rgb(h1, s2, l2)
                }
                13 => {  // saturation
                    let (_, s1, _) = rgb_to_hsl(r1, g1, b1);
                    let (h2, _, l2) = rgb_to_hsl(r2, g2, b2);
                    hsl_to_rgb(h2, s1, l2)
                }
                14 => {  // color
                    let (h1, s1, _) = rgb_to_hsl(r1, g1, b1);
                    let l2 = luminosity(r2, g2, b2);
                    let (r, g, b) = hsl_to_rgb(h1, s1, 0.5);
                    set_lum(r, g, b, l2)
                }
                15 => {  // luminosity
                    let l1 = luminosity(r1, g1, b1);
                    set_lum(r2, g2, b2, l1)
                }
                _ => {
                    // Standard blend modes
                    let blend = |c1: f32, c2: f32| -> f32 {
                        match mode {
                            0 => c1,  // normal
                            1 => c1 * c2,  // multiply
                            2 => 1.0 - (1.0 - c1) * (1.0 - c2),  // screen
                            3 => c1.min(c2),  // darken
                            4 => c1.max(c2),  // lighten
                            5 => if c2 < 0.5 { 2.0 * c1 * c2 } else { 1.0 - 2.0 * (1.0 - c1) * (1.0 - c2) },  // overlay
                            6 => if c2 == 1.0 { c2 } else { (c1 / (1.0 - c2)).min(1.0) },  // color-dodge
                            7 => if c2 == 0.0 { 0.0 } else { 1.0 - ((1.0 - c1) / c2).min(1.0) },  // color-burn
                            8 => if c1 < 0.5 { 2.0 * c1 * c2 } else { 1.0 - 2.0 * (1.0 - c1) * (1.0 - c2) },  // hard-light
                            9 => {  // soft-light
                                if c1 < 0.5 {
                                    c2 - (1.0 - 2.0 * c1) * c2 * (1.0 - c2)
                                } else {
                                    let d = if c2 < 0.25 {
                                        ((16.0 * c2 - 12.0) * c2 + 4.0) * c2
                                    } else {
                                        c2.sqrt()
                                    };
                                    c2 + (2.0 * c1 - 1.0) * (d - c2)
                                }
                            }
                            10 => (c1 - c2).abs(),  // difference
                            11 => c1 + c2 - 2.0 * c1 * c2,  // exclusion
                            _ => c1,
                        }
                    };
                    (blend(r1, r2), blend(g1, g2), blend(b1, b2))
                }
            };

            let out_a = a1 + a2 * (1.0 - a1);
            
            // Compositing formula: qr = (1-qa)*cb + (1-qb)*ca + qa*qb*B(ca/qa, cb/qb)
            // ca = pca1 (premultiplied), qa = a1
            // cb = pca2 (premultiplied), qb = a2
            // B(...) = (br, bg, bb)
            
            let term1 = 1.0 - a2;
            let term2 = 1.0 - a1;
            let term3 = a1 * a2;
            
            let out_r = pca1_r * term1 + pca2_r * term2 + br * term3;
            let out_g = pca1_g * term1 + pca2_g * term2 + bg * term3;
            let out_b = pca1_b * term1 + pca2_b * term2 + bb * term3;

            dst[[y, x, 0]] = (out_r * 255.0).clamp(0.0, 255.0) as u8;
            dst[[y, x, 1]] = (out_g * 255.0).clamp(0.0, 255.0) as u8;
            dst[[y, x, 2]] = (out_b * 255.0).clamp(0.0, 255.0) as u8;
            dst[[y, x, 3]] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

/// feBlend - blend two images using blend mode
#[pyfunction]
pub fn fe_blend<'py>(
    py: Python<'py>,
    in1: numpy::PyReadonlyArray3<'py, u8>,
    in2: numpy::PyReadonlyArray3<'py, u8>,
    mode: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr1 = in1.as_array();
    let arr2 = in2.as_array();
    fe_blend_impl(&arr1, &arr2, mode).into_pyarray(py)
}

pub fn fe_composite_impl_f32(arr1: &ndarray::ArrayView3<f32>, arr2: &ndarray::ArrayView3<f32>, operator: u8, k1: f32, k2: f32, k3: f32, k4: f32) -> Array3<f32> {
    let (h, w, _) = (arr1.shape()[0], arr1.shape()[1], arr1.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a1 = arr1[[y, x, 3]];
            let a2 = arr2[[y, x, 3]];

            let (fa, fb) = match operator {
                0 => (1.0, 1.0 - a1),
                1 => (a2, 0.0),
                2 => (1.0 - a2, 0.0),
                3 => (a2, 1.0 - a1),
                4 => (1.0 - a2, 1.0 - a1),
                5 => (0.0, 0.0),
                _ => (1.0, 1.0 - a1),
            };

            for c in 0..4 {
                let c1 = arr1[[y, x, c]];
                let c2 = arr2[[y, x, c]];

                let out = if operator == 5 {
                    k1 * c1 * c2 + k2 * c1 + k3 * c2 + k4
                } else {
                    c1 * fa + c2 * fb
                };

                dst[[y, x, c]] = out.clamp(0.0, 1.0);
            }
        }
    }
    dst
}

pub fn fe_composite_impl(arr1: &ndarray::ArrayView3<u8>, arr2: &ndarray::ArrayView3<u8>, operator: u8, k1: f32, k2: f32, k3: f32, k4: f32) -> Array3<u8> {
    let (h, w, _) = (arr1.shape()[0], arr1.shape()[1], arr1.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a1 = arr1[[y, x, 3]] as f32 / 255.0;
            let a2 = arr2[[y, x, 3]] as f32 / 255.0;

            let (fa, fb) = match operator {
                0 => (1.0, 1.0 - a1),  // over
                1 => (a2, 0.0),  // in
                2 => (1.0 - a2, 0.0),  // out
                3 => (a2, 1.0 - a1),  // atop
                4 => (1.0 - a2, 1.0 - a1),  // xor
                5 => (0.0, 0.0),  // arithmetic
                _ => (1.0, 1.0 - a1),
            };

            for c in 0..4 {
                let c1 = arr1[[y, x, c]] as f32 / 255.0;
                let c2 = arr2[[y, x, c]] as f32 / 255.0;

                let out = if operator == 5 {
                    k1 * c1 * c2 + k2 * c1 + k3 * c2 + k4
                } else {
                    c1 * fa + c2 * fb
                };

                dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// feComposite - Porter-Duff compositing
#[pyfunction]
pub fn fe_composite<'py>(
    py: Python<'py>,
    in1: numpy::PyReadonlyArray3<'py, u8>,
    in2: numpy::PyReadonlyArray3<'py, u8>,
    operator: u8,
    k1: f32, k2: f32, k3: f32, k4: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr1 = in1.as_array();
    let arr2 = in2.as_array();
    fe_composite_impl(&arr1, &arr2, operator, k1, k2, k3, k4).into_pyarray(py)
}

pub fn fe_merge_impl_f32(layers: &[ndarray::ArrayView3<f32>]) -> Array3<f32> {
    if layers.is_empty() {
        return Array3::<f32>::zeros((1, 1, 4));
    }

    let first = &layers[0];
    let (h, w, _) = (first.shape()[0], first.shape()[1], first.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                dst[[y, x, c]] = first[[y, x, c]];
            }
        }
    }

    for layer in layers.iter().skip(1) {
        let src = layer;
        for y in 0..h {
            for x in 0..w {
                let src_a = src[[y, x, 3]];
                let inv_sa = 1.0 - src_a;
                
                for c in 0..4 {
                    let dst_c = dst[[y, x, c]];
                    let src_c = src[[y, x, c]];
                    let out = src_c + dst_c * inv_sa;
                    dst[[y, x, c]] = out.clamp(0.0, 1.0);
                }
            }
        }
    }
    dst
}

pub fn fe_merge_impl(layers: &[ndarray::ArrayView3<u8>]) -> Array3<u8> {
    if layers.is_empty() {
        return Array3::<u8>::zeros((1, 1, 4));
    }

    let first = &layers[0];
    let (h, w, _) = (first.shape()[0], first.shape()[1], first.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    // Initialize with first layer
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                dst[[y, x, c]] = first[[y, x, c]];
            }
        }
    }

    // Blend subsequent layers using Over operator (Premultiplied)
    // out = src + dst * (1 - src_a)
    for layer in layers.iter().skip(1) {
        let src = layer;
        for y in 0..h {
            for x in 0..w {
                let src_a = src[[y, x, 3]] as f32 / 255.0;
                let inv_sa = 1.0 - src_a;
                
                for c in 0..4 {
                    let dst_c = dst[[y, x, c]] as f32 / 255.0;
                    let src_c = src[[y, x, c]] as f32 / 255.0;
                    let out = src_c + dst_c * inv_sa;
                    dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    dst
}

/// feMerge - merge multiple layers
#[pyfunction]
pub fn fe_merge<'py>(
    py: Python<'py>,
    layers: Vec<numpy::PyReadonlyArray3<'py, u8>>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let layer_views: Vec<_> = layers.iter().map(|l| l.as_array()).collect();
    fe_merge_impl(&layer_views).into_pyarray(py)
}

pub fn fe_color_matrix_impl_f32(src: &ndarray::ArrayView3<f32>, matrix_type: u8, values: &[f32]) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    let mut m = [[0.0f32; 5]; 4];

    // Helper to set identity matrix
    let set_identity = |mat: &mut [[f32; 5]; 4]| {
        mat[0] = [1.0, 0.0, 0.0, 0.0, 0.0];
        mat[1] = [0.0, 1.0, 0.0, 0.0, 0.0];
        mat[2] = [0.0, 0.0, 1.0, 0.0, 0.0];
        mat[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
    };

    match matrix_type {
        0 => { // matrix
            if values.len() != 20 {
                set_identity(&mut m);
            } else {
                for i in 0..4 {
                    for j in 0..5 {
                        m[i][j] = values[i * 5 + j];
                    }
                }
            }
        }
        1 => { // saturate
            let s = if values.len() == 1 { values[0] } else { 1.0 }; 
            m[0] = [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[1] = [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[2] = [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        2 => { // hueRotate
            let angle_deg = if values.len() == 1 { values[0] } else { 0.0 };
            let angle = angle_deg.to_radians();
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            m[0] = [0.213 + cos_a * 0.787 - sin_a * 0.213, 0.715 - cos_a * 0.715 - sin_a * 0.715, 0.072 - cos_a * 0.072 + sin_a * 0.928, 0.0, 0.0];
            m[1] = [0.213 - cos_a * 0.213 + sin_a * 0.143, 0.715 + cos_a * 0.285 + sin_a * 0.140, 0.072 - cos_a * 0.072 - sin_a * 0.283, 0.0, 0.0];
            m[2] = [0.213 - cos_a * 0.213 - sin_a * 0.787, 0.715 - cos_a * 0.715 + sin_a * 0.715, 0.072 + cos_a * 0.928 + sin_a * 0.072, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        3 => { // luminanceToAlpha
            m[0] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[1] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[2] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[3] = [0.2126, 0.7152, 0.0722, 0.0, 0.0];
        }
        _ => {
            set_identity(&mut m);
        }
    }

    for y in 0..h {
        for x in 0..w {
            let r = src[[y, x, 0]];
            let g = src[[y, x, 1]];
            let b = src[[y, x, 2]];
            let a = src[[y, x, 3]];

            let out_r = (m[0][0] * r + m[0][1] * g + m[0][2] * b + m[0][3] * a + m[0][4]).clamp(0.0, 1.0);
            let out_g = (m[1][0] * r + m[1][1] * g + m[1][2] * b + m[1][3] * a + m[1][4]).clamp(0.0, 1.0);
            let out_b = (m[2][0] * r + m[2][1] * g + m[2][2] * b + m[2][3] * a + m[2][4]).clamp(0.0, 1.0);
            let out_a = (m[3][0] * r + m[3][1] * g + m[3][2] * b + m[3][3] * a + m[3][4]).clamp(0.0, 1.0);

            dst[[y, x, 0]] = out_r;
            dst[[y, x, 1]] = out_g;
            dst[[y, x, 2]] = out_b;
            dst[[y, x, 3]] = out_a;
        }
    }
    dst
}

pub fn fe_color_matrix_impl(src: &ndarray::ArrayView3<u8>, matrix_type: u8, values: &[f32]) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let mut m = [[0.0f32; 5]; 4];

    // Helper to set identity matrix
    let set_identity = |mat: &mut [[f32; 5]; 4]| {
        mat[0] = [1.0, 0.0, 0.0, 0.0, 0.0];
        mat[1] = [0.0, 1.0, 0.0, 0.0, 0.0];
        mat[2] = [0.0, 0.0, 1.0, 0.0, 0.0];
        mat[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
    };

    match matrix_type {
        0 => { // matrix
            if values.len() != 20 {
                set_identity(&mut m);
            } else {
                for i in 0..4 {
                    for j in 0..5 {
                        m[i][j] = values[i * 5 + j];
                    }
                }
            }
        }
        1 => { // saturate
            let s = if values.len() == 1 { values[0] } else { 1.0 }; 
            m[0] = [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[1] = [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s, 0.0, 0.0];
            m[2] = [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        2 => { // hueRotate
            let angle_deg = if values.len() == 1 { values[0] } else { 0.0 };
            let angle = angle_deg.to_radians();
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            m[0] = [0.213 + cos_a * 0.787 - sin_a * 0.213, 0.715 - cos_a * 0.715 - sin_a * 0.715, 0.072 - cos_a * 0.072 + sin_a * 0.928, 0.0, 0.0];
            m[1] = [0.213 - cos_a * 0.213 + sin_a * 0.143, 0.715 + cos_a * 0.285 + sin_a * 0.140, 0.072 - cos_a * 0.072 - sin_a * 0.283, 0.0, 0.0];
            m[2] = [0.213 - cos_a * 0.213 - sin_a * 0.787, 0.715 - cos_a * 0.715 + sin_a * 0.715, 0.072 + cos_a * 0.928 + sin_a * 0.072, 0.0, 0.0];
            m[3] = [0.0, 0.0, 0.0, 1.0, 0.0];
        }
        3 => { // luminanceToAlpha
            m[0] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[1] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[2] = [0.0, 0.0, 0.0, 0.0, 0.0];
            m[3] = [0.2126, 0.7152, 0.0722, 0.0, 0.0];
        }
        _ => {
            set_identity(&mut m);
        }
    }

    for y in 0..h {
        for x in 0..w {
            let r = src[[y, x, 0]] as f32 / 255.0;
            let g = src[[y, x, 1]] as f32 / 255.0;
            let b = src[[y, x, 2]] as f32 / 255.0;
            let a = src[[y, x, 3]] as f32 / 255.0;

            let out_r = (m[0][0] * r + m[0][1] * g + m[0][2] * b + m[0][3] * a + m[0][4]).clamp(0.0, 1.0);
            let out_g = (m[1][0] * r + m[1][1] * g + m[1][2] * b + m[1][3] * a + m[1][4]).clamp(0.0, 1.0);
            let out_b = (m[2][0] * r + m[2][1] * g + m[2][2] * b + m[2][3] * a + m[2][4]).clamp(0.0, 1.0);
            let out_a = (m[3][0] * r + m[3][1] * g + m[3][2] * b + m[3][3] * a + m[3][4]).clamp(0.0, 1.0);

            dst[[y, x, 0]] = (out_r * 255.0) as u8;
            dst[[y, x, 1]] = (out_g * 255.0) as u8;
            dst[[y, x, 2]] = (out_b * 255.0) as u8;
            dst[[y, x, 3]] = (out_a * 255.0) as u8;
        }
    }
    dst
}

/// feColorMatrix - apply color transformation matrix
#[pyfunction]
pub fn fe_color_matrix<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    matrix_type: u8,
    values: Vec<f32>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_color_matrix_impl(&arr, matrix_type, &values).into_pyarray(py)
}

pub fn fe_component_transfer_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    func_r: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_g: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_b: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_a: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    fn apply_transfer(val: f32, func: &(u8, Vec<f32>, f32, f32, f32, f32, f32)) -> f32 {
        let (func_type, table, slope, intercept, amplitude, exponent, offset) = func;
        match func_type {
            0 => val,
            1 => {
                if table.len() < 2 { return val; }
                let n = table.len() - 1;
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                let frac = val * n as f32 - k as f32;
                table[k] * (1.0 - frac) + table[k + 1] * frac
            }
            2 => {
                if table.is_empty() { return val; }
                let n = table.len();
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                table[k]
            }
            3 => slope * val + intercept,
            4 => amplitude * val.powf(*exponent) + offset,
            _ => val,
        }
    }

    let funcs = [func_r, func_g, func_b, func_a];

    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let val = src[[y, x, c]];
                let out = apply_transfer(val, funcs[c]);
                dst[[y, x, c]] = out.clamp(0.0, 1.0);
            }
        }
    }
    dst
}

pub fn fe_component_transfer_impl(
    src: &ndarray::ArrayView3<u8>,
    func_r: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_g: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_b: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_a: &(u8, Vec<f32>, f32, f32, f32, f32, f32),
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    fn apply_transfer(val: f32, func: &(u8, Vec<f32>, f32, f32, f32, f32, f32)) -> f32 {
        let (func_type, table, slope, intercept, amplitude, exponent, offset) = func;
        match func_type {
            0 => val,
            1 => {
                if table.len() < 2 { return val; }
                let n = table.len() - 1;
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                let frac = val * n as f32 - k as f32;
                table[k] * (1.0 - frac) + table[k + 1] * frac
            }
            2 => {
                if table.is_empty() { return val; }
                let n = table.len();
                let k = (val * n as f32).floor() as usize;
                let k = k.min(n - 1);
                table[k]
            }
            3 => slope * val + intercept,
            4 => amplitude * val.powf(*exponent) + offset,
            _ => val,
        }
    }

    let funcs = [func_r, func_g, func_b, func_a];

    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let val = src[[y, x, c]] as f32 / 255.0;
                let out = apply_transfer(val, funcs[c]);
                dst[[y, x, c]] = (out * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// feComponentTransfer - apply transfer function to each channel
#[pyfunction]
pub fn fe_component_transfer<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    func_r: (u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_g: (u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_b: (u8, Vec<f32>, f32, f32, f32, f32, f32),
    func_a: (u8, Vec<f32>, f32, f32, f32, f32, f32),
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_component_transfer_impl(&arr, &func_r, &func_g, &func_b, &func_a).into_pyarray(py)
}

/// Van Herk-Gil-Werman algorithm for 1D sliding window min/max
#[inline]
fn vhg_sliding_minmax(data: &[u8], radius: usize, is_min: bool) -> Vec<u8> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let window = 2 * radius + 1;

    if window >= n {
        let result_val = if is_min {
            *data.iter().min().unwrap_or(&0)
        } else {
            *data.iter().max().unwrap_or(&0)
        };
        return vec![result_val; n];
    }

    let mut result = vec![0u8; n];
    let num_blocks = (n + window - 1) / window;
    let mut prefix = vec![0u8; n];
    let mut suffix = vec![0u8; n];

    for block in 0..num_blocks {
        let block_start = block * window;
        let block_end = ((block + 1) * window).min(n);

        let mut val = if is_min { 255u8 } else { 0u8 };
        for i in (block_start..block_end).rev() {
            if is_min { val = val.min(data[i]); } else { val = val.max(data[i]); }
            suffix[i] = val;
        }

        val = if is_min { 255u8 } else { 0u8 };
        for i in block_start..block_end {
            if is_min { val = val.min(data[i]); } else { val = val.max(data[i]); }
            prefix[i] = val;
        }
    }

    for i in 0..n {
        let left = if i >= radius { i - radius } else { 0 };
        let right = if i + radius < n { i + radius } else { n - 1 };
        if is_min {
            result[i] = suffix[left].min(prefix[right]);
        } else {
            result[i] = suffix[left].max(prefix[right]);
        }
    }

    result
}

/// Van Herk-Gil-Werman algorithm for 1D sliding window min/max (f32)
#[inline]
fn vhg_sliding_minmax_f32(data: &[f32], radius: usize, is_min: bool) -> Vec<f32> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let window = 2 * radius + 1;

    if window >= n {
        let result_val = if is_min {
            *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
        } else {
            *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
        };
        return vec![result_val; n];
    }

    let mut result = vec![0.0f32; n];
    let num_blocks = (n + window - 1) / window;
    let mut prefix = vec![0.0f32; n];
    let mut suffix = vec![0.0f32; n];

    for block in 0..num_blocks {
        let block_start = block * window;
        let block_end = ((block + 1) * window).min(n);

        let mut val = if is_min { f32::INFINITY } else { f32::NEG_INFINITY };
        for i in (block_start..block_end).rev() {
            if is_min { val = val.min(data[i]); } else { val = val.max(data[i]); }
            suffix[i] = val;
        }

        val = if is_min { f32::INFINITY } else { f32::NEG_INFINITY };
        for i in block_start..block_end {
            if is_min { val = val.min(data[i]); } else { val = val.max(data[i]); }
            prefix[i] = val;
        }
    }

    for i in 0..n {
        let left = if i >= radius { i - radius } else { 0 };
        let right = if i + radius < n { i + radius } else { n - 1 };
        if is_min {
            result[i] = suffix[left].min(prefix[right]);
        } else {
            result[i] = suffix[left].max(prefix[right]);
        }
    }

    result
}

pub fn fe_morphology_impl_f32(src: &ndarray::ArrayView3<f32>, operator: u8, radius_x: f32, radius_y: f32) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    let rx = radius_x.round() as usize;
    let ry = radius_y.round() as usize;
    let is_erode = operator == 0;

    if rx == 0 && ry == 0 {
        return src.to_owned();
    }

    let mut temp = Array3::<f32>::zeros((h, w, 4));

    if rx > 0 {
        for y in 0..h {
            for c in 0..4 {
                let row: Vec<f32> = (0..w).map(|x| src[[y, x, c]]).collect();
                let result = vhg_sliding_minmax_f32(&row, rx, is_erode);
                for x in 0..w { temp[[y, x, c]] = result[x]; }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { temp[[y, x, c]] = src[[y, x, c]]; }
            }
        }
    }

    let mut dst = Array3::<f32>::zeros((h, w, 4));

    if ry > 0 {
        for x in 0..w {
            for c in 0..4 {
                let col: Vec<f32> = (0..h).map(|y| temp[[y, x, c]]).collect();
                let result = vhg_sliding_minmax_f32(&col, ry, is_erode);
                for y in 0..h { dst[[y, x, c]] = result[y]; }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = temp[[y, x, c]]; }
            }
        }
    }
    dst
}

pub fn fe_morphology_impl(src: &ndarray::ArrayView3<u8>, operator: u8, radius_x: f32, radius_y: f32) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    let rx = radius_x.round() as usize;
    let ry = radius_y.round() as usize;
    let is_erode = operator == 0;

    if rx == 0 && ry == 0 {
        let mut dst = Array3::<u8>::zeros((h, w, 4));
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = src[[y, x, c]]; }
            }
        }
        return dst;
    }

    let mut temp = Array3::<u8>::zeros((h, w, 4));

    if rx > 0 {
        for y in 0..h {
            for c in 0..4 {
                let row: Vec<u8> = (0..w).map(|x| src[[y, x, c]]).collect();
                let result = vhg_sliding_minmax(&row, rx, is_erode);
                for x in 0..w { temp[[y, x, c]] = result[x]; }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { temp[[y, x, c]] = src[[y, x, c]]; }
            }
        }
    }

    let mut dst = Array3::<u8>::zeros((h, w, 4));

    if ry > 0 {
        for x in 0..w {
            for c in 0..4 {
                let col: Vec<u8> = (0..h).map(|y| temp[[y, x, c]]).collect();
                let result = vhg_sliding_minmax(&col, ry, is_erode);
                for y in 0..h { dst[[y, x, c]] = result[y]; }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = temp[[y, x, c]]; }
            }
        }
    }
    dst
}

/// feMorphology - erode or dilate
#[pyfunction]
pub fn fe_morphology<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    operator: u8,
    radius_x: f32,
    radius_y: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_morphology_impl(&arr, operator, radius_x, radius_y).into_pyarray(py)
}

pub fn fe_convolve_matrix_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    order_x: usize,
    order_y: usize,
    kernel: &[f32],
    divisor: f32,
    bias: f32,
    target_x: usize,
    target_y: usize,
    edge_mode: u8,
    preserve_alpha: bool,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    let div = if divisor.abs() < 1e-10 { 1.0 } else { divisor };
    let scaled_kernel: Vec<f32> = kernel.iter().map(|k| k / div).collect();
    
    let h_i = h as i32;
    let w_i = w as i32;
    let target_y_i = target_y as i32;
    let target_x_i = target_x as i32;

    let channels = if preserve_alpha { 3 } else { 4 };

    if scaled_kernel.is_empty() || order_x == 0 || order_y == 0 {
        return src.to_owned();
    }

    if edge_mode == 0 {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    let sy = (y_i + ky as i32 - target_y_i).clamp(0, h_i - 1) as usize;
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }
                        let sx = (x_i + kx as i32 - target_x_i).clamp(0, w_i - 1) as usize;
                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += src[[sy, sx, c]] * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    let add_bias = if c == 3 { 0.0 } else { bias };
                    dst[[y, x, c]] = (sum[c] + add_bias).clamp(0.0, 1.0);
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = src[[y, x, 3]];
                }
            }
        }
    } else {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }

                        let sy = y_i + ky as i32 - target_y_i;
                        let sx = x_i + kx as i32 - target_x_i;

                        let (sy, sx) = match edge_mode {
                            1 => (sy.rem_euclid(h_i), sx.rem_euclid(w_i)),
                            _ => {
                                if sy < 0 || sy >= h_i || sx < 0 || sx >= w_i { continue; }
                                (sy, sx)
                            }
                        };

                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += src[[sy as usize, sx as usize, c]] * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    let add_bias = if c == 3 { 0.0 } else { bias };
                    dst[[y, x, c]] = (sum[c] + add_bias).clamp(0.0, 1.0);
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = src[[y, x, 3]];
                }
            }
        }
    }
    dst
}

pub fn fe_convolve_matrix_impl(
    src: &ndarray::ArrayView3<u8>,
    order_x: usize,
    order_y: usize,
    kernel: &[f32],
    divisor: f32,
    bias: f32,
    target_x: usize,
    target_y: usize,
    edge_mode: u8,
    preserve_alpha: bool,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let div = if divisor.abs() < 1e-10 { 1.0 } else { divisor };
    let scaled_kernel: Vec<f32> = kernel.iter().map(|k| k / div).collect();
    let bias_255 = bias * 255.0;

    let h_i = h as i32;
    let w_i = w as i32;
    let target_y_i = target_y as i32;
    let target_x_i = target_x as i32;

    let channels = if preserve_alpha { 3 } else { 4 };

    if scaled_kernel.is_empty() || order_x == 0 || order_y == 0 {
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = src[[y, x, c]]; }
            }
        }
        return dst;
    }

    if edge_mode == 0 {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    let sy = (y_i + ky as i32 - target_y_i).clamp(0, h_i - 1) as usize;
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }
                        let sx = (x_i + kx as i32 - target_x_i).clamp(0, w_i - 1) as usize;
                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += src[[sy, sx, c]] as f32 * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    dst[[y, x, c]] = (sum[c] + bias_255).clamp(0.0, 255.0) as u8;
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = src[[y, x, 3]];
                }
            }
        }
    } else {
        for y in 0..h {
            let y_i = y as i32;
            for x in 0..w {
                let x_i = x as i32;
                let mut sum = [0.0f32; 4];

                for ky in 0..order_y {
                    for kx in 0..order_x {
                        let kernel_idx = ky * order_x + kx;
                        if kernel_idx >= scaled_kernel.len() { continue; }

                        let sy = y_i + ky as i32 - target_y_i;
                        let sx = x_i + kx as i32 - target_x_i;

                        let (sy, sx) = match edge_mode {
                            1 => (sy.rem_euclid(h_i), sx.rem_euclid(w_i)),
                            _ => {
                                if sy < 0 || sy >= h_i || sx < 0 || sx >= w_i { continue; }
                                (sy, sx)
                            }
                        };

                        let kernel_val = scaled_kernel[kernel_idx];
                        for c in 0..channels {
                            sum[c] += src[[sy as usize, sx as usize, c]] as f32 * kernel_val;
                        }
                    }
                }

                for c in 0..channels {
                    dst[[y, x, c]] = (sum[c] + bias_255).clamp(0.0, 255.0) as u8;
                }
                if preserve_alpha {
                    dst[[y, x, 3]] = src[[y, x, 3]];
                }
            }
        }
    }
    dst
}

/// feConvolveMatrix - apply convolution kernel
#[pyfunction]
pub fn fe_convolve_matrix<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    order_x: usize,
    order_y: usize,
    kernel: Vec<f32>,
    divisor: f32,
    bias: f32,
    target_x: usize,
    target_y: usize,
    edge_mode: u8,
    preserve_alpha: bool,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_convolve_matrix_impl(&arr, order_x, order_y, &kernel, divisor, bias, target_x, target_y, edge_mode, preserve_alpha).into_pyarray(py)
}

fn generate_gradients(seed: i32) -> [[f64; 2]; 256] {
    let mut gradients = [[0.0f64; 2]; 256];
    let mut rng = seed as u32;

    for i in 0..256 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let angle = (rng as f64 / u32::MAX as f64) * std::f64::consts::PI * 2.0;
        gradients[i] = [angle.cos(), angle.sin()];
    }

    gradients
}

fn perlin_noise(x: f64, y: f64, channel: usize, gradients: &[[f64; 2]; 256]) -> f64 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);

    let hash = |x: i32, y: i32, c: usize| -> usize {
        ((x.wrapping_mul(1619) ^ y.wrapping_mul(31337) ^ (c as i32 * 6971)) & 0xFF) as usize
    };

    let g00 = &gradients[hash(x0, y0, channel)];
    let g10 = &gradients[hash(x1, y0, channel)];
    let g01 = &gradients[hash(x0, y1, channel)];
    let g11 = &gradients[hash(x1, y1, channel)];

    let n00 = g00[0] * fx + g00[1] * fy;
    let n10 = g10[0] * (fx - 1.0) + g10[1] * fy;
    let n01 = g01[0] * fx + g01[1] * (fy - 1.0);
    let n11 = g11[0] * (fx - 1.0) + g11[1] * (fy - 1.0);

    let nx0 = n00 + u * (n10 - n00);
    let nx1 = n01 + u * (n11 - n01);

    nx0 + v * (nx1 - nx0)
}

pub fn fe_turbulence_impl_f32(
    width: usize,
    height: usize,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: usize,
    seed: i32,
    noise_type: u8,
    _stitch_tiles: bool,
) -> Array3<f32> {
    let mut pixels = Array3::<f32>::zeros((height, width, 4));
    let gradients = generate_gradients(seed);

    for y in 0..height {
        for x in 0..width {
            for c in 0..4 {
                let mut noise = 0.0f64;
                let mut amplitude = 1.0f64;
                let mut freq_x = base_freq_x;
                let mut freq_y = base_freq_y;

                for _ in 0..num_octaves {
                    let nx = x as f64 * freq_x;
                    let ny = y as f64 * freq_y;
                    let n = perlin_noise(nx, ny, c, &gradients);

                    if noise_type == 0 {
                        noise += n.abs() * amplitude;
                    } else {
                        noise += n * amplitude;
                    }

                    amplitude *= 0.5;
                    freq_x *= 2.0;
                    freq_y *= 2.0;
                }

                let val = if noise_type == 0 {
                    noise
                } else {
                    (noise + 1.0) * 0.5
                };

                pixels[[y, x, c]] = val.clamp(0.0, 1.0) as f32;
            }
        }
    }
    pixels
}

pub fn fe_turbulence_impl(
    width: usize,
    height: usize,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: usize,
    seed: i32,
    noise_type: u8,
    _stitch_tiles: bool,
) -> Array3<u8> {
    let mut pixels = Array3::<u8>::zeros((height, width, 4));
    let gradients = generate_gradients(seed);

    for y in 0..height {
        for x in 0..width {
            for c in 0..4 {
                let mut noise = 0.0f64;
                let mut amplitude = 1.0f64;
                let mut freq_x = base_freq_x;
                let mut freq_y = base_freq_y;

                for _ in 0..num_octaves {
                    let nx = x as f64 * freq_x;
                    let ny = y as f64 * freq_y;
                    let n = perlin_noise(nx, ny, c, &gradients);

                    if noise_type == 0 {
                        noise += n.abs() * amplitude;
                    } else {
                        noise += n * amplitude;
                    }

                    amplitude *= 0.5;
                    freq_x *= 2.0;
                    freq_y *= 2.0;
                }

                let val = if noise_type == 0 {
                    noise
                } else {
                    (noise + 1.0) * 0.5
                };

                pixels[[y, x, c]] = (val * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    pixels
}

/// feTurbulence - generate Perlin noise
#[pyfunction]
pub fn fe_turbulence<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: usize,
    seed: i32,
    noise_type: u8,
    stitch_tiles: bool,
) -> Bound<'py, numpy::PyArray3<u8>> {
    fe_turbulence_impl(width, height, base_freq_x, base_freq_y, num_octaves, seed, noise_type, stitch_tiles).into_pyarray(py)
}

pub fn fe_displacement_map_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    map: &ndarray::ArrayView3<f32>,
    scale: f32,
    x_channel: u8,
    y_channel: u8,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let dx_val = map[[y, x, x_channel as usize]] - 0.5;
            let dy_val = map[[y, x, y_channel as usize]] - 0.5;

            let src_x = x as f32 + dx_val * scale;
            let src_y = y as f32 + dy_val * scale;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            for c in 0..4 {
                let get_pixel = |px: i32, py: i32| -> f32 {
                    if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                        0.0
                    } else {
                        src[[py as usize, px as usize, c]]
                    }
                };

                let v00 = get_pixel(x0, y0);
                let v10 = get_pixel(x1, y0);
                let v01 = get_pixel(x0, y1);
                let v11 = get_pixel(x1, y1);

                let v0 = v00 + fx * (v10 - v00);
                let v1 = v01 + fx * (v11 - v01);
                let v = v0 + fy * (v1 - v0);

                dst[[y, x, c]] = v.clamp(0.0, 1.0);
            }
        }
    }
    dst
}

pub fn fe_displacement_map_impl(
    src: &ndarray::ArrayView3<u8>,
    map: &ndarray::ArrayView3<u8>,
    scale: f32,
    x_channel: u8,
    y_channel: u8,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let dx_val = map[[y, x, x_channel as usize]] as f32 / 255.0 - 0.5;
            let dy_val = map[[y, x, y_channel as usize]] as f32 / 255.0 - 0.5;

            let src_x = x as f32 + dx_val * scale;
            let src_y = y as f32 + dy_val * scale;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            for c in 0..4 {
                let get_pixel = |px: i32, py: i32| -> f32 {
                    if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                        0.0
                    } else {
                        src[[py as usize, px as usize, c]] as f32
                    }
                };

                let v00 = get_pixel(x0, y0);
                let v10 = get_pixel(x1, y0);
                let v01 = get_pixel(x0, y1);
                let v11 = get_pixel(x1, y1);

                let v0 = v00 + fx * (v10 - v00);
                let v1 = v01 + fx * (v11 - v01);
                let v = v0 + fy * (v1 - v0);

                dst[[y, x, c]] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// feDisplacementMap - displace pixels using map
#[pyfunction]
pub fn fe_displacement_map<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    map: numpy::PyReadonlyArray3<'py, u8>,
    scale: f32,
    x_channel: u8,
    y_channel: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let src_arr = src.as_array();
    let map_arr = map.as_array();
    fe_displacement_map_impl(&src_arr, &map_arr, scale, x_channel, y_channel).into_pyarray(py)
}

pub fn fe_tile_impl_f32(src: &ndarray::ArrayView3<f32>, out_width: usize, out_height: usize) -> Array3<f32> {
    let (src_h, src_w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((out_height, out_width, 4));

    if src_h == 0 || src_w == 0 {
        return dst;
    }

    for y in 0..out_height {
        let src_y = y % src_h;
        for x in 0..out_width {
            let src_x = x % src_w;
            for c in 0..4 {
                dst[[y, x, c]] = src[[src_y, src_x, c]];
            }
        }
    }
    dst
}

pub fn fe_tile_impl(src: &ndarray::ArrayView3<u8>, out_width: usize, out_height: usize) -> Array3<u8> {
    let (src_h, src_w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((out_height, out_width, 4));

    if src_h == 0 || src_w == 0 {
        return dst;
    }

    for y in 0..out_height {
        let src_y = y % src_h;
        for x in 0..out_width {
            let src_x = x % src_w;
            for c in 0..4 {
                dst[[y, x, c]] = src[[src_y, src_x, c]];
            }
        }
    }
    dst
}

/// feTile - tile input image to fill region
#[pyfunction]
pub fn fe_tile<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    out_width: usize,
    out_height: usize,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let src_arr = src.as_array();
    fe_tile_impl(&src_arr, out_width, out_height).into_pyarray(py)
}

pub fn fe_diffuse_lighting_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    surface_scale: f32,
    diffuse_constant: f32,
    light_color: (f32, f32, f32), // normalized
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    specular_exponent: f32, limiting_cone_angle: f32,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    // For lighting, we need alpha map from src alpha
    // We treat 'src' as the source image, and use its alpha channel as height map
    
    let get_height = |x: i32, y: i32| -> f32 {
        if x < 0 || x >= w as i32 || y < 0 || y >= h as i32 {
            0.0
        } else {
            src[[y as usize, x as usize, 3]] * surface_scale
        }
    };

    let (lx, ly, lz) = if light_type == 0 {
        // Distant light
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();
        (az_rad.cos() * el_rad.cos(), az_rad.sin() * el_rad.cos(), el_rad.sin())
    } else {
        (0.0, 0.0, 0.0)
    };
    let (dist_lx, dist_ly, dist_lz) = (lx, ly, lz);

    for y in 0..h {
        let iy = y as i32;
        for x in 0..w {
            let ix = x as i32;
            
            // Calculate normal vector
            let z_nw = get_height(ix - 1, iy - 1);
            let z_n  = get_height(ix,     iy - 1);
            let z_ne = get_height(ix + 1, iy - 1);
            let z_w  = get_height(ix - 1, iy);
            let z_e  = get_height(ix + 1, iy);
            let z_sw = get_height(ix - 1, iy + 1);
            let z_s  = get_height(ix,     iy + 1);
            let z_se = get_height(ix + 1, iy + 1);

            let nx = -((z_ne + 2.0 * z_e + z_se) - (z_nw + 2.0 * z_w + z_sw)) / 4.0;
            let ny = -((z_sw + 2.0 * z_s + z_se) - (z_nw + 2.0 * z_n + z_ne)) / 4.0;
            let nz = 1.0;
            
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = (nx / len, ny / len, nz / len);

            // Light vector
            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                let z = get_height(ix, iy);
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - z;
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (dist_lx, dist_ly, dist_lz)
            };

            let n_dot_l = (nx * lx + ny * ly + nz * lz).max(0.0);
            
            let intensity = if light_type == 2 {
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos { 0.0 } else { l_dot_s.powf(specular_exponent) }
            } else {
                1.0
            };

            let diffuse = diffuse_constant * n_dot_l * intensity;
            
            // Result is alpha 1.0, color modulated by diffuse factor
            dst[[y, x, 0]] = (light_color.0 * diffuse).clamp(0.0, 1.0);
            dst[[y, x, 1]] = (light_color.1 * diffuse).clamp(0.0, 1.0);
            dst[[y, x, 2]] = (light_color.2 * diffuse).clamp(0.0, 1.0);
            dst[[y, x, 3]] = 1.0;
        }
    }
    dst
}

pub fn fe_diffuse_lighting_impl(
    src: &ndarray::ArrayView3<u8>,
    surface_scale: f32,
    diffuse_constant: f32,
    light_color: (u8, u8, u8),
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    specular_exponent: f32, limiting_cone_angle: f32,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let (lx, ly, lz) = if light_type == 0 {
        let az = azimuth.to_radians();
        let el = elevation.to_radians();
        (az.cos() * el.cos(), az.sin() * el.cos(), el.sin())
    } else {
        (0.0, 0.0, 0.0)
    };

    for y in 0..h {
        for x in 0..w {
            let get_height = |px: i32, py: i32| -> f32 {
                if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 { return 0.0; }
                src[[py as usize, px as usize, 3]] as f32 / 255.0 * surface_scale
            };

            let ix = x as i32;
            let iy = y as i32;

            let dx = get_height(ix + 1, iy) - get_height(ix - 1, iy);
            let dy = get_height(ix, iy + 1) - get_height(ix, iy - 1);

            let nx = -dx;
            let ny = -dy;
            let nz = 1.0f32;
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = (nx / n_len, ny / n_len, nz / n_len);

            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - get_height(ix, iy);
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (lx, ly, lz)
            };

            let n_dot_l = (nx * lx + ny * ly + nz * lz).max(0.0);

            let intensity = if light_type == 2 {
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos { 0.0 } else { l_dot_s.powf(specular_exponent) }
            } else {
                1.0
            };

            let diffuse = n_dot_l * diffuse_constant * intensity;

            dst[[y, x, 0]] = (light_color.0 as f32 * diffuse).clamp(0.0, 255.0) as u8;
            dst[[y, x, 1]] = (light_color.1 as f32 * diffuse).clamp(0.0, 255.0) as u8;
            dst[[y, x, 2]] = (light_color.2 as f32 * diffuse).clamp(0.0, 255.0) as u8;
            dst[[y, x, 3]] = 255;
        }
    }
    dst
}

/// feDiffuseLighting - diffuse lighting effect
#[pyfunction]
pub fn fe_diffuse_lighting<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    surface_scale: f32,
    diffuse_constant: f32,
    light_color: (u8, u8, u8),
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    specular_exponent: f32, limiting_cone_angle: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_diffuse_lighting_impl(
        &arr, surface_scale, diffuse_constant, light_color, light_type,
        azimuth, elevation, light_x, light_y, light_z,
        points_at_x, points_at_y, points_at_z, specular_exponent, limiting_cone_angle
    ).into_pyarray(py)
}

pub fn fe_specular_lighting_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    surface_scale: f32,
    specular_constant: f32,
    specular_exponent: f32,
    light_color: (f32, f32, f32), // normalized
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    spot_exponent: f32, limiting_cone_angle: f32,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    let get_height = |x: i32, y: i32| -> f32 {
        if x < 0 || x >= w as i32 || y < 0 || y >= h as i32 {
            0.0
        } else {
            src[[y as usize, x as usize, 3]] * surface_scale
        }
    };

    let (dist_lx, dist_ly, dist_lz) = if light_type == 0 {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();
        (az_rad.cos() * el_rad.cos(), az_rad.sin() * el_rad.cos(), el_rad.sin())
    } else {
        (0.0, 0.0, 0.0)
    };

    // Eye vector (0, 0, 1) usually
    let (ex, ey, ez) = (0.0, 0.0, 1.0);

    for y in 0..h {
        let iy = y as i32;
        for x in 0..w {
            let ix = x as i32;
            
            // Calculate normal vector
            let z_nw = get_height(ix - 1, iy - 1);
            let z_n  = get_height(ix,     iy - 1);
            let z_ne = get_height(ix + 1, iy - 1);
            let z_w  = get_height(ix - 1, iy);
            let z_e  = get_height(ix + 1, iy);
            let z_sw = get_height(ix - 1, iy + 1);
            let z_s  = get_height(ix,     iy + 1);
            let z_se = get_height(ix + 1, iy + 1);

            let nx = -((z_ne + 2.0 * z_e + z_se) - (z_nw + 2.0 * z_w + z_sw)) / 4.0;
            let ny = -((z_sw + 2.0 * z_s + z_se) - (z_nw + 2.0 * z_n + z_ne)) / 4.0;
            let nz = 1.0;
            
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = (nx / len, ny / len, nz / len);

            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                let z = get_height(ix, iy);
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - z;
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (dist_lx, dist_ly, dist_lz)
            };

            let hx = lx + ex;
            let hy = ly + ey;
            let hz = lz + ez;
            let h_len = (hx * hx + hy * hy + hz * hz).sqrt();
            let (hx, hy, hz) = if h_len > 1e-6 { (hx / h_len, hy / h_len, hz / h_len) } else { (0.0, 0.0, 1.0) };

            let n_dot_h = (nx * hx + ny * hy + nz * hz).max(0.0);

            let intensity = if light_type == 2 {
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos { 0.0 } else { l_dot_s.powf(spot_exponent) }
            } else {
                1.0
            };

            let specular = n_dot_h.powf(specular_exponent) * specular_constant * intensity;

            dst[[y, x, 0]] = (light_color.0 * specular).clamp(0.0, 1.0);
            dst[[y, x, 1]] = (light_color.1 * specular).clamp(0.0, 1.0);
            dst[[y, x, 2]] = (light_color.2 * specular).clamp(0.0, 1.0);
            let max_rgb = dst[[y, x, 0]].max(dst[[y, x, 1]]).max(dst[[y, x, 2]]);
            dst[[y, x, 3]] = max_rgb;
        }
    }
    dst
}

pub fn fe_specular_lighting_impl(
    src: &ndarray::ArrayView3<u8>,
    surface_scale: f32,
    specular_constant: f32,
    specular_exponent_param: f32,
    light_color: (u8, u8, u8),
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    spot_exponent: f32, limiting_cone_angle: f32,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    let specular_exponent = specular_exponent_param.clamp(1.0, 128.0);
    let spec_constant = specular_constant.max(0.0);

    let (ex, ey, ez) = (0.0f32, 0.0, 1.0);

    let (dist_lx, dist_ly, dist_lz) = if light_type == 0 {
        let az = azimuth.to_radians();
        let el = elevation.to_radians();
        (az.cos() * el.cos(), az.sin() * el.cos(), el.sin())
    } else {
        (0.0, 0.0, 0.0)
    };

    for y in 0..h {
        for x in 0..w {
            let get_height = |px: i32, py: i32| -> f32 {
                if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 { return 0.0; }
                src[[py as usize, px as usize, 3]] as f32 / 255.0 * surface_scale
            };

            let ix = x as i32;
            let iy = y as i32;

            let dx = get_height(ix + 1, iy) - get_height(ix - 1, iy);
            let dy = get_height(ix, iy + 1) - get_height(ix, iy - 1);

            let nx = -dx;
            let ny = -dy;
            let nz = 1.0f32;
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = if n_len > 1e-6 { (nx / n_len, ny / n_len, nz / n_len) } else { (0.0, 0.0, 1.0) };

            let (lx, ly, lz) = if light_type == 1 || light_type == 2 {
                let z = get_height(ix, iy);
                let dx = light_x - x as f32;
                let dy = light_y - y as f32;
                let dz = light_z - z;
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                (dx / len, dy / len, dz / len)
            } else {
                (dist_lx, dist_ly, dist_lz)
            };

            let hx = lx + ex;
            let hy = ly + ey;
            let hz = lz + ez;
            let h_len = (hx * hx + hy * hy + hz * hz).sqrt();
            let (hx, hy, hz) = if h_len > 1e-6 { (hx / h_len, hy / h_len, hz / h_len) } else { (0.0, 0.0, 1.0) };

            let n_dot_h = (nx * hx + ny * hy + nz * hz).max(0.0);

            let intensity = if light_type == 2 {
                let sx = points_at_x - light_x;
                let sy = points_at_y - light_y;
                let sz = points_at_z - light_z;
                let s_len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
                let (sx, sy, sz) = (sx / s_len, sy / s_len, sz / s_len);
                let l_dot_s = -(lx * sx + ly * sy + lz * sz);
                let cone_cos = limiting_cone_angle.to_radians().cos();
                if l_dot_s < cone_cos { 0.0 } else { l_dot_s.powf(spot_exponent) }
            } else {
                1.0
            };

            let specular = n_dot_h.powf(specular_exponent) * spec_constant * intensity;

            dst[[y, x, 0]] = (light_color.0 as f32 * specular).clamp(0.0, 255.0) as u8;
            dst[[y, x, 1]] = (light_color.1 as f32 * specular).clamp(0.0, 255.0) as u8;
            dst[[y, x, 2]] = (light_color.2 as f32 * specular).clamp(0.0, 255.0) as u8;
            let max_rgb = dst[[y, x, 0]].max(dst[[y, x, 1]]).max(dst[[y, x, 2]]);
            dst[[y, x, 3]] = max_rgb;
        }
    }
    dst
}

/// feSpecularLighting - specular lighting effect
#[pyfunction]
pub fn fe_specular_lighting<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    surface_scale: f32,
    specular_constant: f32,
    specular_exponent_param: f32,
    light_color: (u8, u8, u8),
    light_type: u8,
    azimuth: f32, elevation: f32,
    light_x: f32, light_y: f32, light_z: f32,
    points_at_x: f32, points_at_y: f32, points_at_z: f32,
    spot_exponent: f32, limiting_cone_angle: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_specular_lighting_impl(
        &arr, surface_scale, specular_constant, specular_exponent_param, light_color, light_type,
        azimuth, elevation, light_x, light_y, light_z,
        points_at_x, points_at_y, points_at_z, spot_exponent, limiting_cone_angle
    ).into_pyarray(py)
}

/// Compute integral image for a single channel
#[inline]
fn compute_integral_image(src: &[f32], w: usize, h: usize, channel: usize) -> Vec<f64> {
    let mut integral = vec![0.0f64; (w + 1) * (h + 1)];
    let iw = w + 1;

    for y in 0..h {
        let mut row_sum = 0.0f64;
        for x in 0..w {
            row_sum += src[(y * w + x) * 4 + channel] as f64;
            integral[(y + 1) * iw + (x + 1)] = row_sum + integral[y * iw + (x + 1)];
        }
    }
    integral
}

#[inline]
fn integral_query(integral: &[f64], iw: usize, x1: usize, y1: usize, x2: usize, y2: usize) -> f64 {
    integral[y2 * iw + x2] - integral[y1 * iw + x2] - integral[y2 * iw + x1] + integral[y1 * iw + x1]
}

#[inline]
fn box_blur_integral(src: &[f32], dst: &mut [f32], w: usize, h: usize, rx: usize, ry: usize) {
    if w == 0 || h == 0 { return; }

    let integral_r = compute_integral_image(src, w, h, 0);
    let integral_g = compute_integral_image(src, w, h, 1);
    let integral_b = compute_integral_image(src, w, h, 2);
    let integral_a = compute_integral_image(src, w, h, 3);

    let iw = w + 1;

    for y in 0..h {
        let y1 = if y >= ry { y - ry } else { 0 };
        let y2 = (y + ry + 1).min(h);

        for x in 0..w {
            let x1 = if x >= rx { x - rx } else { 0 };
            let x2 = (x + rx + 1).min(w);

            let area = ((x2 - x1) * (y2 - y1)) as f64;
            let inv_area = if area > 0.0 { 1.0 / area } else { 0.0 };

            let idx = (y * w + x) * 4;
            dst[idx]     = (integral_query(&integral_r, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 1] = (integral_query(&integral_g, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 2] = (integral_query(&integral_b, iw, x1, y1, x2, y2) * inv_area) as f32;
            dst[idx + 3] = (integral_query(&integral_a, iw, x1, y1, x2, y2) * inv_area) as f32;
        }
    }
}

pub fn fe_gaussian_blur_impl_f32(src: &ndarray::ArrayView3<f32>, std_dev_x: f32, std_dev_y: f32) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    if std_dev_x < 0.5 && std_dev_y < 0.5 {
        return src.to_owned();
    }

    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    let box_radius_x = ((12.0 * std_dev_x * std_dev_x / 3.0).sqrt() + 0.5).floor() as usize;
    let box_radius_y = ((12.0 * std_dev_y * std_dev_y / 3.0).sqrt() + 0.5).floor() as usize;

    let total_pixels = h * w * 4;

    let mut buf_a = vec![0.0f32; total_pixels];
    let mut buf_b = vec![0.0f32; total_pixels];

    // Copy src to buffer
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            buf_a[idx] = src[[y, x, 0]];
            buf_a[idx + 1] = src[[y, x, 1]];
            buf_a[idx + 2] = src[[y, x, 2]];
            buf_a[idx + 3] = src[[y, x, 3]];
        }
    }

    let mut current = &mut buf_a;
    let mut next = &mut buf_b;

    if box_radius_x > 0 || box_radius_y > 0 {
        for _ in 0..3 {
            box_blur_integral(current, next, w, h, box_radius_x, box_radius_y);
            std::mem::swap(&mut current, &mut next);
        }
    }

    let mut dst = Array3::<f32>::zeros((h, w, 4));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            dst[[y, x, 0]] = current[idx].clamp(0.0, 1.0);
            dst[[y, x, 1]] = current[idx + 1].clamp(0.0, 1.0);
            dst[[y, x, 2]] = current[idx + 2].clamp(0.0, 1.0);
            dst[[y, x, 3]] = current[idx + 3].clamp(0.0, 1.0);
        }
    }
    dst
}

pub fn fe_gaussian_blur_impl(src: &ndarray::ArrayView3<u8>, std_dev_x: f32, std_dev_y: f32) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    if std_dev_x < 0.5 && std_dev_y < 0.5 {
        let mut dst = Array3::<u8>::zeros((h, w, 4));
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 { dst[[y, x, c]] = src[[y, x, c]]; }
            }
        }
        return dst;
    }

    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    let box_radius_x = ((12.0 * std_dev_x * std_dev_x / 3.0).sqrt() + 0.5).floor() as usize;
    let box_radius_y = ((12.0 * std_dev_y * std_dev_y / 3.0).sqrt() + 0.5).floor() as usize;

    let total_pixels = h * w * 4;

    let mut buf_a = vec![0.0f32; total_pixels];
    let mut buf_b = vec![0.0f32; total_pixels];

    // Copy src to buffer (assuming already premultiplied)
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            buf_a[idx] = src[[y, x, 0]] as f32;
            buf_a[idx + 1] = src[[y, x, 1]] as f32;
            buf_a[idx + 2] = src[[y, x, 2]] as f32;
            buf_a[idx + 3] = src[[y, x, 3]] as f32;
        }
    }

    let mut current = &mut buf_a;
    let mut next = &mut buf_b;

    if box_radius_x > 0 || box_radius_y > 0 {
        for _ in 0..3 {
            box_blur_integral(current, next, w, h, box_radius_x, box_radius_y);
            std::mem::swap(&mut current, &mut next);
        }
    }

    let mut dst = Array3::<u8>::zeros((h, w, 4));
    for i in 0..total_pixels {
        // Flat copy back to Array3 structure
        // We can optimize this but let's stick to the loop for now or reuse index
    }
    
    // Using explicit loops to map back to Array3
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            dst[[y, x, 0]] = current[idx].clamp(0.0, 255.0) as u8;
            dst[[y, x, 1]] = current[idx + 1].clamp(0.0, 255.0) as u8;
            dst[[y, x, 2]] = current[idx + 2].clamp(0.0, 255.0) as u8;
            dst[[y, x, 3]] = current[idx + 3].clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

/// feGaussianBlur - optimized Gaussian blur
#[pyfunction]
pub fn fe_gaussian_blur<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    std_dev_x: f32,
    std_dev_y: f32,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_gaussian_blur_impl(&arr, std_dev_x, std_dev_y).into_pyarray(py)
}

pub fn fe_drop_shadow_impl_f32(
    src: &ndarray::ArrayView3<f32>,
    dx: f32,
    dy: f32,
    std_dev_x: f32,
    std_dev_y: f32,
    flood_r: f32, flood_g: f32, flood_b: f32, flood_a: f32,
) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    let dx_i = dx.round() as i32;
    let dy_i = dy.round() as i32;

    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    let mut alpha = vec![0.0f32; h * w];
    for y in 0..h {
        let src_y = y as i32 - dy_i;
        if src_y < 0 || src_y >= h as i32 { continue; }
        for x in 0..w {
            let src_x = x as i32 - dx_i;
            if src_x < 0 || src_x >= w as i32 { continue; }
            let a = src[[src_y as usize, src_x as usize, 3]];
            alpha[y * w + x] = a * flood_a;
        }
    }

    if std_dev_x >= 0.5 || std_dev_y >= 0.5 {
        let box_radius_x = ((12.0 * std_dev_x * std_dev_x / 3.0).sqrt() + 0.5).floor() as usize;
        let box_radius_y = ((12.0 * std_dev_y * std_dev_y / 3.0).sqrt() + 0.5).floor() as usize;

        let mut current = &mut alpha;
        let mut next = &mut vec![0.0f32; h * w];

        for _ in 0..3 {
            box_blur_single_channel(current, next, w, h, box_radius_x, box_radius_y);
            std::mem::swap(&mut current, &mut next);
        }
    }

    let mut dst = Array3::<f32>::zeros((h, w, 4));
    for y in 0..h {
        for x in 0..w {
            let shadow_a = alpha[y * w + x];
            // Shadow
            if shadow_a > 0.0 {
                dst[[y, x, 0]] = flood_r;
                dst[[y, x, 1]] = flood_g;
                dst[[y, x, 2]] = flood_b;
                dst[[y, x, 3]] = shadow_a;
            }
            
            // Composite source over shadow
            let src_a = src[[y, x, 3]];
            if src_a > 0.0 {
                let inv_src_a = 1.0 - src_a;
                dst[[y, x, 0]] = src[[y, x, 0]] + dst[[y, x, 0]] * inv_src_a;
                dst[[y, x, 1]] = src[[y, x, 1]] + dst[[y, x, 1]] * inv_src_a;
                dst[[y, x, 2]] = src[[y, x, 2]] + dst[[y, x, 2]] * inv_src_a;
                dst[[y, x, 3]] = src_a + dst[[y, x, 3]] * inv_src_a;
            }
        }
    }
    dst
}

fn box_blur_single_channel(src: &[f32], dst: &mut [f32], w: usize, h: usize, rx: usize, ry: usize) {
    if w == 0 || h == 0 { return; }
    let integral = compute_integral_image_single(src, w, h);
    let iw = w + 1;
    let area_x = (2 * rx + 1) as f64;
    let area_y = (2 * ry + 1) as f64;
    let inv_area = 1.0 / (area_x * area_y);

    for y in 0..h {
        let y1 = if y >= ry { y - ry } else { 0 };
        let y2 = if y + ry < h { y + ry + 1 } else { h };
        for x in 0..w {
            let x1 = if x >= rx { x - rx } else { 0 };
            let x2 = if x + rx < w { x + rx + 1 } else { w };
            let sum = integral_query(&integral, iw, x1, y1, x2, y2);
            let count = ((x2 - x1) * (y2 - y1)) as f64;
            // Adjust for edge cases if necessary, or just use box area approx
            dst[y * w + x] = (sum / count) as f32; // Normalizing by actual kernel overlap area
        }
    }
}

fn compute_integral_image_single(src: &[f32], w: usize, h: usize) -> Vec<f64> {
    let mut integral = vec![0.0f64; (w + 1) * (h + 1)];
    let iw = w + 1;
    for y in 0..h {
        let mut row_sum = 0.0f64;
        for x in 0..w {
            row_sum += src[y * w + x] as f64;
            integral[(y + 1) * iw + (x + 1)] = row_sum + integral[y * iw + (x + 1)];
        }
    }
    integral
}

pub fn fe_drop_shadow_impl(
    src: &ndarray::ArrayView3<u8>,
    dx: f32,
    dy: f32,
    std_dev_x: f32,
    std_dev_y: f32,
    flood_r: u8, flood_g: u8, flood_b: u8, flood_a: u8,
) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);

    let dx_i = dx.round() as i32;
    let dy_i = dy.round() as i32;

    let std_dev_x = std_dev_x.min(100.0);
    let std_dev_y = std_dev_y.min(100.0);

    let mut alpha = vec![0.0f32; h * w];
    for y in 0..h {
        let src_y = y as i32 - dy_i;
        if src_y < 0 || src_y >= h as i32 { continue; }
        for x in 0..w {
            let src_x = x as i32 - dx_i;
            if src_x < 0 || src_x >= w as i32 { continue; }
            let a = src[[src_y as usize, src_x as usize, 3]] as f32;
            alpha[y * w + x] = a * (flood_a as f32 / 255.0);
        }
    }

    if std_dev_x >= 0.5 || std_dev_y >= 0.5 {
        let box_radius_x = ((12.0 * std_dev_x * std_dev_x / 3.0).sqrt() + 0.5).floor() as usize;
        let box_radius_y = ((12.0 * std_dev_y * std_dev_y / 3.0).sqrt() + 0.5).floor() as usize;

        for _ in 0..3 {
            if box_radius_x > 0 {
                let mut next = vec![0.0f32; h * w];
                for y in 0..h {
                    let mut sum = 0.0f32;
                    for i in 0..box_radius_x.min(w) + 1 {
                        sum += alpha[y * w + i];
                    }
                    let mut left = 0i32 - box_radius_x as i32;
                    let mut right = box_radius_x as i32;

                    for x in 0..w {
                        if right < w as i32 && right > x as i32 {
                            sum += alpha[y * w + right as usize];
                        }
                        let count = (right.min(w as i32 - 1) - left.max(0) + 1) as f32;
                        next[y * w + x] = sum / count;
                        if left >= 0 {
                            sum -= alpha[y * w + left as usize];
                        }
                        left += 1;
                        right += 1;
                    }
                }
                alpha = next;
            }

            if box_radius_y > 0 {
                let mut next = vec![0.0f32; h * w];
                for x in 0..w {
                    let mut sum = 0.0f32;
                    for i in 0..box_radius_y.min(h) + 1 {
                        sum += alpha[i * w + x];
                    }
                    let mut top = 0i32 - box_radius_y as i32;
                    let mut bottom = box_radius_y as i32;

                    for y in 0..h {
                        if bottom < h as i32 && bottom > y as i32 {
                            sum += alpha[bottom as usize * w + x];
                        }
                        let count = (bottom.min(h as i32 - 1) - top.max(0) + 1) as f32;
                        next[y * w + x] = sum / count;
                        if top >= 0 {
                            sum -= alpha[top as usize * w + x];
                        }
                        top += 1;
                        bottom += 1;
                    }
                }
                alpha = next;
            }
        }
    }

    let mut dst = Array3::<u8>::zeros((h, w, 4));
    for y in 0..h {
        for x in 0..w {
            let shadow_a = alpha[y * w + x] / 255.0;
            let src_a = src[[y, x, 3]] as f32 / 255.0;

            let out_a = src_a + shadow_a * (1.0 - src_a);

            if out_a > 0.0 {
                for c in 0..3 {
                    let src_c = src[[y, x, c]] as f32 / 255.0;
                    let shadow_c = match c { 0 => flood_r, 1 => flood_g, _ => flood_b } as f32 / 255.0;
                    let out_c = (src_c * src_a + shadow_c * shadow_a * (1.0 - src_a)) / out_a;
                    dst[[y, x, c]] = (out_c * 255.0).clamp(0.0, 255.0) as u8;
                }
                dst[[y, x, 3]] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// feDropShadow - create drop shadow effect
#[pyfunction]
pub fn fe_drop_shadow<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
    dx: f32,
    dy: f32,
    std_dev_x: f32,
    std_dev_y: f32,
    flood_r: u8, flood_g: u8, flood_b: u8, flood_a: u8,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    fe_drop_shadow_impl(&arr, dx, dy, std_dev_x, std_dev_y, flood_r, flood_g, flood_b, flood_a).into_pyarray(py)
}

pub fn get_source_alpha_impl_f32(src: &ndarray::ArrayView3<f32>) -> Array3<f32> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<f32>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a = src[[y, x, 3]];
            dst[[y, x, 0]] = 0.0;
            dst[[y, x, 1]] = 0.0;
            dst[[y, x, 2]] = 0.0;
            dst[[y, x, 3]] = a;
        }
    }
    dst
}

pub fn get_source_alpha_impl(src: &ndarray::ArrayView3<u8>) -> Array3<u8> {
    let (h, w, _) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((h, w, 4));

    for y in 0..h {
        for x in 0..w {
            let a = src[[y, x, 3]];
            dst[[y, x, 0]] = 0;
            dst[[y, x, 1]] = 0;
            dst[[y, x, 2]] = 0;
            dst[[y, x, 3]] = a;
        }
    }
    dst
}

/// Get SourceAlpha from input image
#[pyfunction]
pub fn get_source_alpha<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray3<'py, u8>,
) -> Bound<'py, numpy::PyArray3<u8>> {
    let arr = src.as_array();
    get_source_alpha_impl(&arr).into_pyarray(py)
}

/// Register filter module functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fe_flood, m)?)?;
    m.add_function(wrap_pyfunction!(fe_offset, m)?)?;
    m.add_function(wrap_pyfunction!(fe_blend, m)?)?;
    m.add_function(wrap_pyfunction!(fe_composite, m)?)?;
    m.add_function(wrap_pyfunction!(fe_merge, m)?)?;
    m.add_function(wrap_pyfunction!(fe_color_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fe_component_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(fe_morphology, m)?)?;
    m.add_function(wrap_pyfunction!(fe_convolve_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fe_turbulence, m)?)?;
    m.add_function(wrap_pyfunction!(fe_displacement_map, m)?)?;
    m.add_function(wrap_pyfunction!(fe_tile, m)?)?;
    m.add_function(wrap_pyfunction!(fe_diffuse_lighting, m)?)?;
    m.add_function(wrap_pyfunction!(fe_specular_lighting, m)?)?;
    m.add_function(wrap_pyfunction!(fe_gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(fe_drop_shadow, m)?)?;
    m.add_function(wrap_pyfunction!(get_source_alpha, m)?)?;
    Ok(())
}
