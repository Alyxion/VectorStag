//! HSL Color Space Helpers for blend modes

pub fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
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

pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
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

pub fn luminosity(r: f32, g: f32, b: f32) -> f32 {
    0.3 * r + 0.59 * g + 0.11 * b
}

pub fn set_lum(r: f32, g: f32, b: f32, l: f32) -> (f32, f32, f32) {
    let d = l - luminosity(r, g, b);
    clip_color(r + d, g + d, b + d)
}

pub fn clip_color(mut r: f32, mut g: f32, mut b: f32) -> (f32, f32, f32) {
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
pub fn saturation(r: f32, g: f32, b: f32) -> f32 {
    r.max(g).max(b) - r.min(g).min(b)
}

#[allow(dead_code)]
pub fn set_sat(r: f32, g: f32, b: f32, s: f32) -> (f32, f32, f32) {
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
