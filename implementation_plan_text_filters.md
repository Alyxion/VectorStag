# VectorStag: Text and Filters Implementation Plan

## 1. Text Rendering Support

**Status:** Completely Unimplemented
**Estimated Complexity:** High
**Estimated Effort:** 2-3 weeks

### Architecture
Text rendering in SVG is complex due to font selection, layout, shaping, and path conversion. For VectorStag, the best approach is to convert text to vector paths (outlines) and render them using the existing high-quality path renderer. This avoids needing a separate bitmap font rasterizer and ensures consistent anti-aliasing.

### Required Dependencies
*   **`fontdb`**: For system font loading and matching (handling `font-family`).
*   **`ttf-parser`** or **`freetype`** (via `rusttype`): To parse font files and extract glyph outlines.
*   **`rustybuzz`**: For text shaping (ligatures, kerning, complex scripts). Essential for correct rendering.

### Implementation Steps

1.  **Font Management System**
    *   Create a global `FontManager` struct that initializes `fontdb::Database`.
    *   Implement font matching logic based on CSS `font-family`, `font-weight`, `font-style`.
    *   Load system fonts and allow loading custom font files.

2.  **Text Layout Engine**
    *   Parse `<text>` and `<tspan>` elements.
    *   Handle `x`, `y`, `dx`, `dy`, and `rotate` attributes.
    *   Implement `text-anchor` (start, middle, end) alignment calculations.
    *   Integrate `rustybuzz` to shape text segments into positioned glyphs.

3.  **Glyph to Path Conversion**
    *   Use `ttf-parser` to retrieve the outline of each glyph.
    *   Convert font curves (Quadratic/Cubic Beziers) to VectorStag's `PathCmd` format.
    *   Apply glyph transforms (size, position, rotation).

4.  **Integration with Renderer**
    *   In `render_node`, handle `text` tag.
    *   Resolve styles (fill/stroke) for text.
    *   Feed generated glyph paths into `render_path`.

5.  **Advanced Text Features (Phase 2)**
    *   `<textPath>`: Warping text along an SVG path.
    *   `white-space` handling.
    *   Bi-directional text (Bidi).

---

## 2. Filter Effects Support

**Status:** Partially Implemented (Logic exists in `filters.rs` but coupled to Python)
**Estimated Complexity:** Medium-High
**Estimated Effort:** 2 weeks

### Current State
*   `vectorstag_rust/src/filters.rs` contains logic for many primitives: `feFlood`, `feOffset`, `feBlend`, `feComposite`, `feColorMatrix`, `feGaussianBlur` (via convolution?), `feMorphology`, `feTurbulence`, etc.
*   **Problem:** These functions currently accept and return `numpy::PyArray3`, making them unusable directly within the Rust `svg_renderer.rs` pipeline which uses internal buffers.

### Architecture Refactoring

1.  **Decouple Filter Logic**
    *   Refactor `src/filters.rs` to operate on generic slices `&[u8]` or `ndarray::ArrayView3` instead of Python types.
    *   Create a "core" Rust module for image processing.
    *   Keep the `#[pyfunction]` wrappers in `filters.rs` but make them call the pure Rust core functions.

2.  **Filter Graph Processor**
    *   Parse `<filter>` elements and their children (`fe*` primitives).
    *   Build a dependency graph (DAG) of filter primitives based on `in` and `result` attributes.
    *   Implement a `FilterContext` struct to manage intermediate image buffers (layers).

3.  **Integration with Renderer**
    *   When an element has `filter="url(#id)"`:
        1.  Determine filter region (bbox + margins).
        2.  Render the element into a temporary RGBA buffer (SourceGraphic).
        3.  Execute the filter graph on this buffer.
        4.  Composite the final result back onto the main canvas using `alpha_composite_inplace`.

### Coordinate Systems
*   Must handle `filterUnits` and `primitiveUnits`:
    *   `userSpaceOnUse`: Coordinates are global.
    *   `objectBoundingBox`: Coordinates are relative to the element (requires mapping 0..1 to pixels).

### Missing Primitives
*   **`feGaussianBlur`**: Need a highly optimized stack blur or IIR blur for large radii (convolution matrix is too slow for large blurs).
*   **`feSpecularLighting`**: Logic similar to diffuse but needs implementation.
*   **Input Sources**: Support `SourceAlpha`, `BackgroundImage` (maybe skip), `FillPaint`, `StrokePaint`.

---

## Summary of Recommendation

1.  **Immediate Next Step**: Refactor `filters.rs` to separate pure Rust implementation from Python bindings. This enables filter support without new dependencies.
2.  **Secondary Step**: Add `fontdb`, `ttf-parser`, and `rustybuzz` to `Cargo.toml` and verify compilation.
3.  **Tertiary Step**: Implement the `render_text` stub converting glyphs to paths.
