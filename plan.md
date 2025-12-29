# VectorStag - Pure Python SVG Renderer

## Project Goal
Create an SVG renderer using only Pillow (and optionally OpenCV) without external SVG libraries, achieving visual parity with CairoSVG and resvg.

## Current Status: 99.9% Emoji / 99.7% Flag Accuracy

**Date**: 2025-12-29

### Emoji Test Results (3427 Noto emojis)
- **Average Accuracy**: 99.9%
- **99%+ Accuracy**: 99.9% (3424 files)
- **95-99%**: 0.1% (3 files)
- **<80%**: 0.0% (0 files)
- **Throughput**: ~50 files/sec

### Flag Test Results (358 Noto flags)
- **Average Accuracy**: 99.7%
- **99%+ Accuracy**: 99.2% (355 files)
- **95-99%**: 0.6% (2 files)
- **<80%**: 0.3% (1 file - QA.svg edge case)

**Reference Renderer**: resvg (Rust-based)

### Material Icons Test (336 files)
- **Average Accuracy**: 99.0%
- **99%+ Accuracy**: 60.7% (204 files)
- **95-99%**: 39.3% (132 files)
- **<95%**: 0.0% (0 files)

### FontAwesome Icons (2028 files)
- **Average Accuracy**: 100.0%
- **99%+ Accuracy**: 100.0% (2028 files)
- **95-99%**: 0.0% (0 files)

### Lucide Icons (200 files)
- **Average Accuracy**: 98.2%
- **99%+ Accuracy**: 18.5% (37 files)
- **95-99%**: 81.0% (162 files)
- **90-95%**: 0.5% (1 file)

### W3C SVG Samples (30 files)
- **Average Accuracy**: 95.8% (vs resvg)
- **99%+ Accuracy**: 80.0% (24 files)
- **95-99%**: 13.3% (4 files)
- **<80%**: 6.7% (2 files - tiger.svg, circles1.svg edge cases)
- Note: tiger.svg (36.8%) has unusual attributes (height only, no viewBox)

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
13. **Closed path stroke rendering** - Always use polygon-based stroke for closed paths to ensure proper miter joins (NP flag fix)
14. **Non-convex stroke detection** - Detect reflex angles and use segment-based stroke for non-convex shapes
15. **Stroke ring rendering** - Fixed closed polygon strokes using quad-per-edge approach with proper miter points (IL Star of David fix)
16. **Comparison dimension calculation** - Use viewBox for aspect ratio, handle unit conversions and precision issues
17. **spreadMethod support** - Added pad/reflect/repeat for linear and radial gradients
18. **URL gradient parsing** - Fixed parsing of gradient URLs with fallback colors (e.g., `url(#id) rgb(0,0,0)`)
19. **currentColor keyword** - Added support for the CSS currentColor keyword (defaults to black)

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
- [x] spreadMethod (pad, reflect, repeat) for gradients *(implemented 2025-12-29)*
- [x] currentColor keyword support *(implemented 2025-12-29)*
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
# Unified comparison tool - svg_compare.py

# List available collections
python svg_compare.py list

# Pre-render references (run once per collection)
python svg_compare.py prerender --emojis --flags --material -j 16
python svg_compare.py prerender --all -j 16

# Fast comparison (no PNG output)
python svg_compare.py compare --emojis --flags -j 16
python svg_compare.py compare --all -j 16

# Generate comparison grid PNGs (VectorStag | resvg | diff)
python svg_compare.py compare --emojis --save -j 16

# Benchmark VectorStag performance
python benchmark_vectorstag.py --emojis --profile
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
