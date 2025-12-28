# VectorStag - Pure Python SVG Renderer

## Project Goal
Create an SVG renderer using only Pillow (and optionally OpenCV) without external SVG libraries, achieving visual parity with CairoSVG.

## Current Status: 91.9% Average Similarity

### Test Results (30 SVG samples from W3C)
- **27 Passing** (>80% similarity): android, atom, check, clippath, compass, copyleft, feed, gaussian1-3, heart, helloworld, italian-flag, lineargradient1-4, paths-data-08-t, paths-data-09-t, python, radialgradient1-2, rectangles, shapes-polygon-01-t, shapes-polyline-01-t, star, yinyang
- **2 Failing** (<50%): circles1, tiger
- **1 Error**: smile.svg (XML entity parsing issue)

---

## Implemented Features

### Shapes
- [x] `<rect>` - including rounded corners (rx, ry)
- [x] `<circle>`
- [x] `<ellipse>`
- [x] `<line>`
- [x] `<polygon>`
- [x] `<polyline>`
- [x] `<path>` - full path command support

### Path Commands
- [x] M/m - moveto
- [x] L/l - lineto
- [x] H/h - horizontal lineto
- [x] V/v - vertical lineto
- [x] C/c - cubic bezier
- [x] S/s - smooth cubic bezier
- [x] Q/q - quadratic bezier
- [x] T/t - smooth quadratic bezier
- [x] A/a - elliptical arc (converted to beziers)
- [x] Z/z - closepath

### Transforms
- [x] translate(tx, ty)
- [x] scale(sx, sy)
- [x] rotate(angle, cx, cy)
- [x] skewX(angle)
- [x] skewY(angle)
- [x] matrix(a, b, c, d, e, f)
- [x] Transform inheritance through groups

### Styles
- [x] fill (color, none, url() gradient reference)
- [x] stroke (color, none)
- [x] stroke-width
- [x] fill-opacity
- [x] stroke-opacity
- [x] opacity
- [x] fill-rule (nonzero, evenodd - parsed but not fully implemented)
- [x] Style inheritance from parent groups

### Gradients
- [x] `<linearGradient>` with stops
- [x] `<radialGradient>` with stops
- [x] gradientUnits: objectBoundingBox
- [x] gradientUnits: userSpaceOnUse
- [x] Gradient href inheritance (xlink:href)
- [x] stop-color, stop-opacity

### Text
- [x] Basic `<text>` rendering
- [x] x, y positioning
- [x] font-size
- [x] font-family (falls back to system fonts)

### Document
- [x] width/height attributes
- [x] viewBox with proper scaling
- [x] Namespace handling (svg, xlink)

---

## Missing Features (Priority Order)

### High Priority - Would Improve Accuracy

1. **Anti-aliasing**
   - Current: Hard pixel edges
   - Needed: Smooth anti-aliased edges for shapes and paths
   - Location: `renderer.py` - polygon/path drawing functions
   - Approach: Use supersampling or implement proper anti-aliasing

2. **Fill Rule Implementation**
   - Current: Parsed but always uses Pillow's default
   - Needed: Proper evenodd vs nonzero fill rules
   - Location: `renderer.py:_fill_and_stroke_polygon()`
   - Approach: Implement scanline fill with winding number calculation

3. **Stroke Properties**
   - Missing: stroke-linecap (butt, round, square)
   - Missing: stroke-linejoin (miter, round, bevel)
   - Missing: stroke-dasharray, stroke-dashoffset
   - Location: `renderer.py` - stroke drawing

4. **Unit Handling**
   - Current: Basic unit conversion
   - Issues: cm/mm units may not scale correctly in all contexts
   - Location: `parser.py:_parse_length()`
   - Fix circles1.svg issue

### Medium Priority - Extended SVG Support

5. **Clipping Paths**
   - Current: `<clipPath>` elements ignored
   - Needed: Apply clip paths to contained elements
   - Location: New `_apply_clip_path()` in renderer

6. **Masks**
   - Current: Not implemented
   - Needed: `<mask>` element support

7. **Filters**
   - Current: Filter elements ignored (gaussian blur works by accident)
   - Needed: `<filter>`, `<feGaussianBlur>`, etc.

8. **Use/Symbol**
   - Current: `<use>` and `<symbol>` not implemented
   - Needed: Reference and instantiate defined elements

9. **Markers**
   - Current: Not implemented
   - Needed: `<marker>` for arrowheads, etc.

### Low Priority - Edge Cases

10. **Text Advanced**
    - Missing: `<tspan>`, text-anchor, dominant-baseline
    - Missing: textPath, text on path
    - Missing: Font styling (bold, italic, weight)

11. **CSS Parsing**
    - Current: Inline style only
    - Missing: `<style>` block CSS rules
    - Missing: CSS selectors, classes

12. **Image Element**
    - Current: `<image>` not implemented
    - Needed: Embed raster images

13. **Pattern Fills**
    - Current: Not implemented
    - Needed: `<pattern>` element

---

## Known Issues

### circles1.svg (48% similarity)
- **Cause**: SVG has no width/height, uses cm units
- **Fix**: Compute bounding box from content when dimensions missing

### tiger.svg (49.8% similarity)
- **Cause**: Complex paths, thin strokes, anti-aliasing differences
- **Fix**: Implement anti-aliasing, verify stroke rendering

### smile.svg (Error)
- **Cause**: XML entity `&Smile;` not defined
- **Fix**: Add XML entity handling or skip malformed files

---

## Architecture

```
vectorstag/
├── __init__.py          # Public API exports
├── parser.py            # SVG DOM parsing
│   ├── SVGParser        # Main parser class
│   ├── Transform        # 2D affine transform
│   ├── Style            # Style attributes
│   ├── *Element         # Element dataclasses
│   └── *Gradient        # Gradient definitions
├── path_parser.py       # Path 'd' attribute parsing
│   ├── parse_path()     # Convert d string to commands
│   └── arc_to_bezier()  # Arc approximation
└── renderer.py          # Pillow rendering
    ├── SVGRenderer      # Main renderer class
    ├── RenderContext    # Rendering state
    └── _render_*()      # Element-specific renderers
```

---

## Next Steps

1. **Implement anti-aliasing** - Biggest visual improvement
2. **Fix stroke rendering** - linecap, linejoin
3. **Implement fill-rule** - evenodd support
4. **Add more test coverage** - Unit tests for parser/renderer
5. **Performance optimization** - Gradient rendering is slow (per-pixel loop)

---

## Usage

```python
from vectorstag import SVGRenderer

# Basic usage
renderer = SVGRenderer()
image = renderer.render_file("input.svg")
image.save("output.png")

# With scaling
image = renderer.render_file("input.svg", width=800, height=600)

# From string
svg_content = '<svg>...</svg>'
image = renderer.render(svg_content)

# Custom background
renderer = SVGRenderer(background=(0, 0, 0, 0))  # Transparent
```

## Comparison Script

```bash
poetry run python compare_render.py
```

Outputs comparison images to `samples/comparison/`.
