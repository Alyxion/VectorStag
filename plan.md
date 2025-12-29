# VectorStag - Pure Python SVG Renderer

## Project Goal
Create an SVG renderer using only Pillow (and optionally OpenCV) without external SVG libraries, achieving visual parity with CairoSVG and resvg.

## Current Status: 99.9% Emoji / 99.8% Flag Accuracy

**Date**: 2025-12-29

### Emoji Test Results (3427 Noto emojis)
- **Average Accuracy**: 99.9%
- **99%+ Accuracy**: 99.8% (3421 files)
- **95-99%**: 0.2% (6 files)
- **<80%**: 0.0% (0 files)
- **Throughput**: ~65 files/sec

### Flag Test Results (358 Noto flags)
- **Average Accuracy**: 99.8%
- **99%+ Accuracy**: 99.4% (356 files)
- **95-99%**: 0.6% (2 files)
- **90-95%**: 0.0% (0 files)
- **<80%**: 0.0% (0 files)

**Reference Renderer**: resvg (Rust-based)

### W3C SVG Samples (30 files)
- **True Accuracy**: 99.4% (excluding CairoSVG bugs)
- **Raw Comparison**: 98.9%

---

## Recent Optimizations & Fixes

### Performance (2025-12-29)
1. **Vectorized gradient rendering** - 5.7x faster using numpy
2. **Optimized self-intersection check** - Skip for >200 points, limit to 5000 pairs
3. **Vectorized scanline fill** - Numpy-based edge processing
4. **Multiprocessing** - 16-worker parallel testing

### Bug Fixes (2025-12-29)
1. **Gradient alpha compositing** - Fixed `paste` → `alpha_composite` for proper transparency
2. **Nonzero winding rule** - Fixed multi-polygon fills to create proper holes
3. **display:none support** - Added CSS display property parsing and rendering skip
4. **GradientTransform** - Now properly applied in renderer
5. **Radial gradient inverse transform** - Fixed matrix inversion formula for transformed radial gradients
6. **Stroke gradient support** - Added gradient fills for strokes (clown mouth, etc.)
7. **Stroke miterlimit** - Apply miterlimit to prevent infinitely long miters
8. **stroke-dasharray** - Added dashed/dotted stroke support
9. **Switch element** - Added `<switch>` element support for fallback rendering
10. **Nested SVG support** - Added `<svg>` element parsing inside other SVGs (SI flag)
11. **userSpaceOnUse gradient transforms** - Fixed element transform propagation for gradients
12. **Comparison workflow** - Render at aspect-ratio-preserving size to match resvg output

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
- [x] gradientTransform support

### Styles
- [x] fill (color, none, url() gradient reference)
- [x] stroke (color, none, url() gradient reference)
- [x] stroke-width
- [x] stroke-linecap (butt, round, square)
- [x] stroke-linejoin (miter, round, bevel)
- [x] stroke-miterlimit
- [x] stroke-dasharray
- [x] fill-opacity
- [x] stroke-opacity
- [x] opacity
- [x] fill-rule (nonzero, evenodd)
- [x] display (none, inline, block)
- [x] Style inheritance from parent groups

### Rendering
- [x] Anti-aliasing (configurable supersampling, default 4x)
- [x] Automatic bounding box computation
- [x] ClipPath support
- [x] Proper alpha compositing
- [x] Nonzero winding rule with holes
- [x] Evenodd fill rule

### Text
- [x] Basic `<text>` rendering
- [x] x, y positioning
- [x] font-size
- [x] font-family mapping

### Gradients
- [x] `<linearGradient>` with stops
- [x] `<radialGradient>` with stops
- [x] gradientUnits: objectBoundingBox
- [x] gradientUnits: userSpaceOnUse
- [x] gradientTransform
- [x] Gradient href inheritance
- [x] stop-color, stop-opacity
- [x] Vectorized rendering (numpy)

### Document
- [x] width/height attributes
- [x] viewBox with proper scaling
- [x] preserveAspectRatio
- [x] Namespace handling
- [x] CSS class parsing

### Use Element
- [x] `<use>` element support
- [x] x, y positioning

### Switch Element
- [x] `<switch>` element support
- [x] requiredExtensions fallback

---

## Known Issues

### Previously Fixed Issues
- **AS.svg**: Added `<switch>` element support - now renders correctly
- **BR.svg**: Was CairoSVG bug with negative viewBox - our rendering was correct
- **TW.svg**: Fixed with proper resvg reference generation (aspect ratio preserved)

### CairoSVG Bugs (verified with resvg)
- clippath.svg: CairoSVG renders intersection wrong
- lineargradient1/2.svg: CairoSVG fills gaps that don't exist
- Gaussian blur: CairoSVG doesn't apply properly
- Negative viewBox: CairoSVG stretches content incorrectly

---

## Missing Features

### Medium Priority
- [ ] stroke-dashoffset
- [ ] `<mask>` element
- [ ] `<filter>` elements (feGaussianBlur, etc.)
- [ ] `<marker>` for arrowheads

### Low Priority
- [ ] `<tspan>`, text-anchor, dominant-baseline
- [ ] `<style>` block CSS rules
- [ ] `<image>` element
- [ ] `<pattern>` fills

---

## Scripts

```bash
# Pre-render references (run once)
python prerender_references.py --emojis --flags

# Fast comparison against pre-rendered references
python compare_fast.py --emojis --flags -j 16

# Benchmark VectorStag performance
python benchmark_vectorstag.py --emojis --profile

# Generate comparison images
python compare_all.py --emojis --flags --limit 100
```

---

## Usage

```python
from vectorstag import SVGRenderer

# Basic usage
renderer = SVGRenderer()
image = renderer.render_file("input.svg")
image.save("output.png")

# With scaling and antialiasing
renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
image = renderer.render_file("input.svg", width=800, height=600)

# From string
image = renderer.render(svg_content)
```

---

## Architecture

```
vectorstag/
├── __init__.py          # Public API exports
├── parser.py            # SVG DOM parsing
│   ├── SVGParser        # Main parser class
│   ├── Transform        # 2D affine transform
│   ├── Style            # Style attributes (incl. display)
│   └── *Element         # Element dataclasses
├── path_parser.py       # Path 'd' attribute parsing
└── renderer.py          # Pillow rendering
    ├── SVGRenderer      # Main renderer class
    ├── _fill_multi_polygon_nonzero()  # Winding rule with holes
    ├── _create_*_gradient_image()     # Vectorized gradients
    └── _render_*()      # Element-specific renderers
```
