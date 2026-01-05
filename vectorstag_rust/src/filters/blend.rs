//! feBlend - blend two images using blend mode

use pyo3::prelude::*;
use numpy::IntoPyArray;
use ndarray::Array3;
use super::color_utils::{rgb_to_hsl, hsl_to_rgb, luminosity, set_lum};

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
            let pca1_r = arr1[[y, x, 0]] as f32 / 255.0;
            let pca1_g = arr1[[y, x, 1]] as f32 / 255.0;
            let pca1_b = arr1[[y, x, 2]] as f32 / 255.0;
            let a1 = arr1[[y, x, 3]] as f32 / 255.0;

            let pca2_r = arr2[[y, x, 0]] as f32 / 255.0;
            let pca2_g = arr2[[y, x, 1]] as f32 / 255.0;
            let pca2_b = arr2[[y, x, 2]] as f32 / 255.0;
            let a2 = arr2[[y, x, 3]] as f32 / 255.0;

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
