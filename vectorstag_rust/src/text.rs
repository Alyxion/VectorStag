use fontdb::{Database, Family, Query, Weight, Style as FontStyle, Stretch, Source};
use ttf_parser::{Face, OutlineBuilder};
use rustybuzz::{UnicodeBuffer, shape, Face as RbFace};
use crate::path::PathCmd;

pub struct FontManager {
    db: Database,
}

impl FontManager {
    pub fn new() -> Self {
        let mut db = Database::new();
        db.load_system_fonts();
        
        // Ensure we have a sans-serif fallback if possible
        // (In a real app we might bundle one)
        
        Self { db }
    }

    pub fn load_font_data(&self, family: &str, weight: u16, italic: bool) -> Option<(Vec<u8>, usize)> {
        let query = Query {
            families: &[Family::Name(family), Family::SansSerif],
            weight: Weight(weight),
            stretch: Stretch::Normal,
            style: if italic { FontStyle::Italic } else { FontStyle::Normal },
        };

        if let Some(id) = self.db.query(&query) {
            if let Some(face_info) = self.db.face(id) {
                if let Source::File(path) = &face_info.source {
                    if let Ok(data) = std::fs::read(path) {
                        return Some((data, face_info.index as usize));
                    }
                }
                // Handle Source::Binary/SharedFile if needed, but load_system_fonts uses File usually
            }
        }
        None
    }
}

struct PathCmdBuilder {
    commands: Vec<PathCmd>,
    scale: f64,
    offset_x: f64,
    offset_y: f64,
}

impl OutlineBuilder for PathCmdBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.commands.push(PathCmd::M(
            x as f64 * self.scale + self.offset_x,
            -y as f64 * self.scale + self.offset_y // Flip Y for SVG coords
        ));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.commands.push(PathCmd::L(
            x as f64 * self.scale + self.offset_x,
            -y as f64 * self.scale + self.offset_y
        ));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.commands.push(PathCmd::Q(
            x1 as f64 * self.scale + self.offset_x,
            -y1 as f64 * self.scale + self.offset_y,
            x as f64 * self.scale + self.offset_x,
            -y as f64 * self.scale + self.offset_y
        ));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.commands.push(PathCmd::C(
            x1 as f64 * self.scale + self.offset_x,
            -y1 as f64 * self.scale + self.offset_y,
            x2 as f64 * self.scale + self.offset_x,
            -y2 as f64 * self.scale + self.offset_y,
            x as f64 * self.scale + self.offset_x,
            -y as f64 * self.scale + self.offset_y
        ));
    }

    fn close(&mut self) {
        self.commands.push(PathCmd::Z);
    }
}

pub struct Glyph {
    pub path: Vec<PathCmd>,
    // positioning info could be added here if needed separate from path
}

pub fn layout_text(
    text: &str,
    x: f64,
    y: f64,
    font_family: &str,
    font_size: f64,
    font_weight: u16,
    italic: bool,
    text_anchor: &str, // start, middle, end
    font_manager: &FontManager,
) -> Vec<Vec<PathCmd>> {
    // 1. Load Font
    let (font_data, index) = match font_manager.load_font_data(font_family, font_weight, italic) {
        Some(data) => data,
        None => return vec![], // No font found
    };

    let face = match Face::parse(&font_data, index as u32) {
        Ok(f) => f,
        Err(_) => return vec![],
    };

    let rb_face = match RbFace::from_slice(&font_data, index as u32) {
        Some(f) => f,
        None => return vec![],
    };

    // 2. Shape text
    let mut buffer = UnicodeBuffer::new();
    buffer.push_str(text);
    let glyph_buffer = shape(&rb_face, &[], buffer);
    let glyph_infos = glyph_buffer.glyph_infos();
    let glyph_positions = glyph_buffer.glyph_positions();

    let units_per_em = face.units_per_em() as f64;
    let scale = font_size / units_per_em;

    // 3. Calculate total width for anchor alignment
    let mut total_width = 0.0;
    for pos in glyph_positions {
        total_width += pos.x_advance as f64 * scale;
    }

    let start_x = match text_anchor {
        "middle" => x - total_width / 2.0,
        "end" => x - total_width,
        _ => x,
    };

    // 4. Generate paths
    let mut glyph_paths = Vec::new();
    let mut current_x = start_x;
    let current_y = y; // Baseline

    for (info, pos) in glyph_infos.iter().zip(glyph_positions.iter()) {
        let glyph_id = ttf_parser::GlyphId(info.glyph_id as u16);
        
        let offset_x = current_x + pos.x_offset as f64 * scale;
        let offset_y = current_y - pos.y_offset as f64 * scale; // Y is up in fonts, down in SVG

        let mut builder = PathCmdBuilder {
            commands: Vec::new(),
            scale,
            offset_x,
            offset_y,
        };

        if let Some(_) = face.outline_glyph(glyph_id, &mut builder) {
            if !builder.commands.is_empty() {
                glyph_paths.push(builder.commands);
            }
        }

        current_x += pos.x_advance as f64 * scale;
    }

    glyph_paths
}
