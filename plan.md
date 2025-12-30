# VectorStag - Pure Python SVG Renderer

## Project Goal
Create an SVG renderer using only Pillow (and optionally OpenCV) without external SVG libraries, achieving visual parity with CairoSVG and resvg.

## Current Status: 100% FontAwesome / 99.8% Emoji / 99.7% Flag Accuracy

**Date**: 2025-12-30

### Performance vs Reference Renderers
- **VectorStag**: ~32 files/sec at 400x400 with 4x antialiasing (single-thread)
- **CairoSVG**: ~41 files/sec (VectorStag is ~0.79x)
- **Resvg**: ~75 files/sec (VectorStag is ~0.43x)
- **Multiprocessing**: ~520 files/sec with 14 workers

### FontAwesome Icons (2028 files)
- **Average Accuracy**: 100.0%
- **99%+ Accuracy**: 100.0% (2028 files)
- **<99%**: 0.0% (0 files)

### Emoji Test Results (3427 Noto emojis)
- **Average Accuracy**: 99.8%
- **99%+ Accuracy**: 99.9% (3423 files)
- **95-99%**: 0.1% (4 files)
- **<80%**: 0.0% (0 files)

### Flag Test Results (295 Noto flags rendered)
- **Average Accuracy**: 99.7%
- **99%+ Accuracy**: 99.3% (293 files)
- **95-99%**: 0.3% (1 file)
- **<80%**: 0.3% (1 file - QA.svg edge case at 13.1%)
- **Errors**: 63 files (complex flags with memory issues)

**Reference Renderer**: resvg (Rust-based)

### Material Icons Test (336 files)
- **Average Accuracy**: 99.0%
- **99%+ Accuracy**: 58.0% (195 files)
- **95-99%**: 42.0% (141 files)
- **<95%**: 0.0% (0 files)

### Lucide Icons (200 files)
- **Average Accuracy**: 98.7%
- **99%+ Accuracy**: 44.0% (88 files)
- **95-99%**: 56.0% (112 files)
- **<95%**: 0.0% (0 files)

### W3C SVG Samples (30 files)
- **Average Accuracy**: 99.3% (vs resvg)
- **99%+ Accuracy**: 80.0% (24 files)
- **95-99%**: 16.7% (5 files)
- **90-95%**: 3.3% (1 file)
- **Note**: Lower scores are primarily due to text rendering differences (different fonts)

### Comparison Images
Comparison grids (VectorStag | resvg | diff) saved to:
- `comparisons/emojis/` - 3427 files
- `comparisons/flags/` - 358 files
- `comparisons/material/` - 336 files
- `comparisons/fontawesome/` - 2028 files
- `comparisons/lucide/` - 200 files
- `comparisons/w3c/` - 30 files

---

## Recent Optimizations & Fixes

### Performance (2025-12-30)
1. **Vectorized gradient rendering** - 5.7x faster using numpy
2. **Optimized self-intersection check** - Skip for >200 points, limit to 5000 pairs
3. **Vectorized scanline fill** - Numpy-based edge processing
4. **Multiprocessing** - 12-worker parallel testing (memory-optimized)
5. **Memory optimization** - Eliminated full-size temp images, use cropped regions
6. **Rust fill algorithms** - `fill_polygon_evenodd` and `fill_multi_polygon_evenodd` via Rust extension
7. **Float32 gradients** - Reduced memory from float64 to float32, row-by-row computation
8. **Rust resize** - Box filter resize in Rust, faster than PIL LANCZOS for 4x downscale
9. **Rust gradient images** - `create_linear_gradient_image` and `create_radial_gradient_image` in Rust
10. **Numpy clipping optimization** - Reuse numpy arrays between clipping and resize steps
11. **Rust polygon nonzero fill** - All polygon fill operations now use Rust when available
12. **Direct-to-array rendering** - `fill_polygon_to_array` and `fill_multi_polygon_to_array` composite directly to numpy arrays, bypassing PIL
13. **Numpy-first image pipeline** - Main image created as numpy array, avoiding PIL Image.new and conversions
14. **Rust alpha compositing** - All alpha compositing uses `alpha_composite_inplace` for numpy arrays
15. **Stroke polygon fast path** - Open/closed strokes use fast Rust polygon fill instead of PIL ImageDraw
16. **Gradient numpy passthrough** - Gradient functions return numpy arrays directly, avoiding PIL roundtrip

### Bug Fixes (2025-12-30)
1. **Recursion protection** - Added MAX_PARSE_DEPTH and MAX_RENDER_DEPTH limits to prevent stack overflow
2. **Circular `<use>` detection** - Track `_use_stack` to prevent infinite loops on circular references
3. **Arc-to-bezier control point fix** - Fixed Rust `arc_to_bezier` using `half_d.sin()` instead of `d_theta.sin()` for control point calculation. This was causing ~21 pixel errors for large arcs (FontAwesome 97.3% → 100%)
4. **Stroke round/bevel linejoin** - Added `linejoin` parameter to Rust `render_stroke_closed_polygon` for proper round/bevel corner handling (Android logo fix)
5. **Premultiplied alpha resize** - Fixed `resize_rgba` to use proper premultiplied alpha averaging instead of independent channel averaging, eliminating dark edges on anti-aliased shapes
6. **Gaussian blur SVG spec** - Implemented triple box-blur approximation per SVG spec (`d = floor(s * 3 * sqrt(2*pi) / 4 + 0.5)`), proper stdDeviation scaling with combined element+base transform (W3C 90% → 93.3% at 99%+)
7. **Font path fallbacks** - Added Noto fonts (NotoSerif, NotoSans, NotoMono) to font path fallbacks for systems without DejaVu fonts
8. **Filter + clip-path combination** - Fixed `_render_element_with_clip` to properly handle elements with both clip-path AND filter (gaussian blur now works inside clip regions)
9. **Round/bevel stroke segmented approach** - Use segmented stroke rendering for round/bevel joins to get correct per-edge perpendicular offsets (fixes Android belly cutout)
10. **Letterbox background color** - Fixed viewBox clipping to set letterbox areas to background color instead of transparent (was causing 8% similarity drop)
11. **Round join pie slices** - Changed round join corners from full circles to pie slice arcs, preventing over-fill at corners
12. **Skip miter triangles for round joins** - Miter triangles were filling corners with square shapes even when linejoin="round", overriding the pieslice arcs (Android 99.72% → 99.85%)

### Notes
- **Text Rendering**: Now working well - use CairoSVG as reference for text comparisons
- **Gaussian Blur**: Significantly improved, but slight color differences remain due to blending/blur algorithm differences - investigate resvg source code for exact implementation
- **Debugging**: Always create comparison images with `svg_compare.py compare --save` to verify fixes visually.
  - Thin pink outlines around ALL edges: Normal - PIL vs resvg rasterization difference (~0.25px)
  - Large solid pink areas inside shapes: Real rendering bug - investigate
  - Pink text: Expected - different font rendering between renderers

### Completed: Android Logo (99.85% similarity)
**Fixed**: Round corners now render correctly
- Changed from full circles to pie slice arcs for round joins
- Arc angles calculated from perpendicular directions of adjacent edges
- Segmented stroke rendering for round/bevel joins
- Skip miter triangles for round joins (was filling corners square instead of round)
- Remaining ~0.15% difference is due to PIL vs resvg rasterization (~0.25px edge variance)

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

## Performance Notes

VectorStag achieves ~60% of CairoSVG performance at 4x antialiasing. The remaining performance gap is due to:

1. **PIL alpha_composite overhead** (~25% of render time) - PIL's native compositing is called per-element. Cairo does this in native C with less overhead.

2. **Image.new allocation** (~16% of render time) - Creating temporary images for semi-transparent fills. Cairo reuses internal buffers.

3. **Python/C boundary crossing** - Each PIL/numpy call has overhead that pure C libraries avoid.

### Potential Future Optimizations
- Replace PIL compositing with native Rust implementation (tested, but conversion overhead made it slower)
- Implement full rendering pipeline in Rust (major rewrite)
- Reduce temporary image allocations by reusing buffers

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

## Tools

Three CLI tools for testing and rendering:

### 1. svg_compare.py - Comparison & Testing
```bash
# List available collections
python svg_compare.py list

# Pre-render references (run once per collection)
python svg_compare.py prerender --emojis --flags --material -j 16
python svg_compare.py prerender --all -j 16

# Compare VectorStag against references
python svg_compare.py compare --emojis --flags -j 16
python svg_compare.py compare --all -j 16

# Save comparison grid PNGs (VectorStag | resvg | diff)
python svg_compare.py compare --emojis --save -j 16
```

### 2. benchmark.py - Performance Testing
```bash
# Benchmark a collection
python benchmark.py --emojis -j 16
python benchmark.py --all --limit 500

# Profile a single file
python benchmark.py --file samples/svg/tiger.svg --profile

# Check Rust extension status
python benchmark.py --check-rust
```

### 3. render.py - Simple Rendering
```bash
# Render SVG to PNG
python render.py input.svg output.png

# Render at specific size
python render.py input.svg output.png --width 800 --height 600

# Render with options
python render.py input.svg output.png --antialias 8 --background white
```

### Collection Flags
All tools support the same collection flags:
- `--emojis` - Noto Color Emojis (3427 files)
- `--flags` - Noto Flags (358 files)
- `--material` - Material Design Icons (336 files)
- `--fontawesome` - FontAwesome Icons (2028 files)
- `--lucide` - Lucide Icons (200 files)
- `--w3c` - W3C SVG Samples (30 files)
- `--all` - All collections

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
