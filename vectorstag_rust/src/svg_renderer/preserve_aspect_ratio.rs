
use super::Transform;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Align {
    None,
    XMinYMin,
    XMidYMin,
    XMaxYMin,
    XMinYMid,
    XMidYMid,
    XMaxYMid,
    XMinYMax,
    XMidYMax,
    XMaxYMax,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeetOrSlice {
    Meet,
    Slice,
}

#[derive(Clone, Copy, Debug)]
pub struct PreserveAspectRatio {
    pub align: Align,
    pub meet_or_slice: MeetOrSlice,
}

impl Default for PreserveAspectRatio {
    fn default() -> Self {
        Self {
            align: Align::XMidYMid,
            meet_or_slice: MeetOrSlice::Meet,
        }
    }
}

pub fn parse_preserve_aspect_ratio(s: &str) -> PreserveAspectRatio {
    let s = s.trim();
    if s.is_empty() {
        return PreserveAspectRatio::default();
    }

    let mut parts = s.split_whitespace();
    let align_str = parts.next().unwrap_or("");
    let mos_str = parts.next().unwrap_or("");

    let align = match align_str {
        "none" => Align::None,
        "xMinYMin" => Align::XMinYMin,
        "xMidYMin" => Align::XMidYMin,
        "xMaxYMin" => Align::XMaxYMin,
        "xMinYMid" => Align::XMinYMid,
        "xMidYMid" => Align::XMidYMid,
        "xMaxYMid" => Align::XMaxYMid,
        "xMinYMax" => Align::XMinYMax,
        "xMidYMax" => Align::XMidYMax,
        "xMaxYMax" => Align::XMaxYMax,
        _ => Align::XMidYMid, // Fallback to default
    };

    let meet_or_slice = match mos_str {
        "slice" => MeetOrSlice::Slice,
        _ => MeetOrSlice::Meet,
    };

    PreserveAspectRatio { align, meet_or_slice }
}

pub fn compute_viewbox_transform(
    vb_x: f64, vb_y: f64, vb_w: f64, vb_h: f64,
    dest_w: f64, dest_h: f64,
    par: PreserveAspectRatio
) -> Transform {
    if vb_w <= 0.0 || vb_h <= 0.0 || dest_w <= 0.0 || dest_h <= 0.0 {
        return Transform::default();
    }

    if par.align == Align::None {
        let sx = dest_w / vb_w;
        let sy = dest_h / vb_h;
        return Transform::translate(-vb_x * sx, -vb_y * sy).multiply(&Transform::scale(sx, sy));
    }

    let scale_x = dest_w / vb_w;
    let scale_y = dest_h / vb_h;
    
    let scale = match par.meet_or_slice {
        MeetOrSlice::Meet => scale_x.min(scale_y),
        MeetOrSlice::Slice => scale_x.max(scale_y),
    };

    let scaled_w = vb_w * scale;
    let scaled_h = vb_h * scale;

    let dx = match par.align {
        Align::XMinYMin | Align::XMinYMid | Align::XMinYMax => 0.0,
        Align::XMidYMin | Align::XMidYMid | Align::XMidYMax => (dest_w - scaled_w) / 2.0,
        Align::XMaxYMin | Align::XMaxYMid | Align::XMaxYMax => dest_w - scaled_w,
        Align::None => 0.0, // Unreachable
    };

    let dy = match par.align {
        Align::XMinYMin | Align::XMidYMin | Align::XMaxYMin => 0.0,
        Align::XMinYMid | Align::XMidYMid | Align::XMaxYMid => (dest_h - scaled_h) / 2.0,
        Align::XMinYMax | Align::XMidYMax | Align::XMaxYMax => dest_h - scaled_h,
        Align::None => 0.0, // Unreachable
    };

    // Transform: translate(dx, dy) * scale(scale, scale) * translate(-vb_x, -vb_y)
    Transform::translate(dx, dy)
        .multiply(&Transform::scale(scale, scale))
        .multiply(&Transform::translate(-vb_x, -vb_y))
}
