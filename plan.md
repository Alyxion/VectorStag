# VectorStag - Pure Python SVG Renderer

## Project Goal
Create an SVG renderer using only Pillow (and optionally OpenCV) without external SVG libraries, achieving visual parity with CairoSVG and resvg.

**Target**: 99.9% accuracy on resvg-test-suite (currently 94.3%)

## Current Status: 94.3% resvg-test-suite / 100% FontAwesome / 99.9% Emoji / 99.9% Flag Accuracy

**Date**: 2025-12-31

### Priority Issues to Reach 99.9%
1. **feSpecularLighting** (70.4%) - Complex lighting calculations need improvement
2. **feImage** (80.3%) - Embedded SVG images not supported
3. **feConvolveMatrix** (80.6%) - Pattern support needed
4. **structure/image** (~81%) - External/embedded image handling
5. **Filter subregion calculations** - Incomplete implementation
6. **Text rendering differences** - Font metrics/positioning

### resvg-test-suite Results (1,295 tests)
- **Average Accuracy**: 94.3%
- **99%+ Accuracy**: 51.9% (672 tests)
- **95-99%**: 20.8% (270 tests)
- **90-95%**: 9.4% (121 tests)
- **80-90%**: 9.0% (117 tests)
- **<80%**: 8.9% (115 tests)
- **Errors**: 4 (0.3%)

| Category | Accuracy | Tests |
|----------|----------|-------|
| shapes | 96.7% | 133 |
| painting | 95.9% | 144 |
| paint-servers | 94.5% | 149 |
| masking | 93.3% | 91 |
| filters | 93.1% | 396 |
| structure | 92.2% | 238 |

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

### Flag Test Results (358 Noto flags)
- **Average Accuracy**: 99.9%
- **99%+ Accuracy**: 99.4% (356 files)
- **95-99%**: 0.6% (2 files - AF.svg, complex emblems)
- **<95%**: 0.0% (0 files)
- **Errors**: 0 (single-threaded); memory issues with parallel workers on complex flags

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

### SVG Filter Implementation (2025-12-30)
Implemented comprehensive SVG filter support with all filter primitives in Rust for performance:

**Filter Primitives Implemented**:
- feGaussianBlur, feOffset, feFlood, feBlend, feComposite, feMerge
- feColorMatrix, feComponentTransfer, feMorphology, feConvolveMatrix
- feTurbulence, feDisplacementMap, feTile, feImage
- feDiffuseLighting, feSpecularLighting, feDropShadow
- Light sources: FeDistantLight, FePointLight, FeSpotLight

**Filter Accuracy by Subcategory** (260 tests):
| Filter | Accuracy | Notes |
|--------|----------|-------|
| feDisplacementMap | 100.0% | Excellent |
| feDistantLight | 99.7% | Excellent |
| flood-opacity | 99.8% | Excellent |
| feDropShadow | 97.6% | Excellent |
| feOffset | 97.4% | Excellent |
| feComponentTransfer | 97.2% | Excellent |
| enable-background | 97.2% | Excellent |
| filter-functions | 97.1% | Excellent |
| feGaussianBlur | 96.7% | O(1) sliding window blur |
| feMerge | 96.8% | Excellent |
| flood-color | 96.6% | Excellent |
| feFlood | 96.3% | Fixed subregion handling |
| feComposite | 95.9% | Excellent |
| feDiffuseLighting | 95.7% | Fixed no-light-source case |
| feBlend | 92.9% | All blend modes supported |
| feTile | 92.9% | Fixed subregion tiling |
| feTurbulence | 92.9% | Perlin noise implemented |
| feSpotLight | 92.2% | Good |
| fePointLight | 90.7% | Good |
| filter | 90.4% | Good |
| feColorMatrix | 89.7% | Some UB tests |
| feMorphology | 89.4% | Radius clamping |
| feConvolveMatrix | 80.6% | Limited by pattern support |
| feImage | 80.3% | Data URLs supported |
| feSpecularLighting | 70.4% | Complex lighting calculations |

**Performance Optimizations**:
1. **O(1) Gaussian blur** - Replaced O(radius) box blur with sliding window approach
2. **O(1) Drop shadow blur** - Same optimization for feDropShadow
3. **Radius clamping** - feMorphology radius clamped to prevent slow operations
4. **Filter inheritance** - xlink:href support for filter element inheritance

**Known Limitations**:
- Pattern fills not implemented (affects feConvolveMatrix tests)
- Filter subregion calculations incomplete
- feImage external file references not supported

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

### resvg-test-suite Improvements (2025-12-30)
1. **`<mask>` element** - Implemented SVG masking with luminance-based alpha (mask elements render content, convert to luminance, apply as alpha mask)
2. **`<symbol>` element** - Symbols now properly parse and render only when referenced by `<use>`, not directly
3. **`<image>` element** - Added support for embedded images via data URLs (base64 PNG/JPEG)
4. **RGBA color support** - Added parsing for `#RGBA`, `#RRGGBBAA`, and `rgba()` color formats with alpha channel
5. **HSL/HSLA colors** - Implemented HSL to RGB conversion for `hsl()` and `hsla()` color functions
6. **Radial gradient `fr`** - Added focal radius attribute support for radial gradients
7. **CSS unit parsing** - Added support for mm, rem, vmin, vmax, ch, rlh, vh, vw, q units
8. **visibility:hidden** - Added visibility attribute to Style, elements with visibility:hidden don't render
9. **Improved similarity calculation** - Benchmark now ignores RGB values when alpha=0 (transparent pixels)
10. **Error handling** - Fixed various parsing errors for edge cases (invalid hex colors, percentage opacity, etc.)

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
13. **preserveAspectRatio="none" in comparisons** - Fixed `render_vectorstag_for_comparison()` to check `should_stretch()` for SVGs with `preserveAspectRatio="none"` (QA.svg 13% → 99.99%)

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

### Symbol Element
- [x] `<symbol>` element support
- [x] Only renders when referenced by `<use>`

### Image Element
- [x] `<image>` element with data URLs
- [x] Base64 PNG/JPEG support
- [x] preserveAspectRatio

### Mask Element
- [x] `<mask>` element support
- [x] Luminance-based masking
- [x] maskUnits and maskContentUnits

### Filter Element
- [x] `<filter>` element with primitives
- [x] Filter chaining with named buffers (in, in2, result)
- [x] SourceGraphic, SourceAlpha inputs
- [x] Filter inheritance via xlink:href
- [x] feGaussianBlur (O(1) sliding window)
- [x] feOffset
- [x] feFlood
- [x] feBlend (all 16 blend modes including HSL)
- [x] feComposite (all Porter-Duff operators)
- [x] feMerge, feMergeNode
- [x] feColorMatrix (matrix, saturate, hueRotate, luminanceToAlpha)
- [x] feComponentTransfer (identity, table, discrete, linear, gamma)
- [x] feMorphology (erode, dilate)
- [x] feConvolveMatrix
- [x] feTurbulence (Perlin noise)
- [x] feDisplacementMap
- [x] feTile
- [x] feImage (data URLs)
- [x] feDiffuseLighting
- [x] feSpecularLighting
- [x] feDropShadow
- [x] Light sources: feDistantLight, fePointLight, feSpotLight

### Colors
- [x] Hex colors (#RGB, #RRGGBB, #RGBA, #RRGGBBAA)
- [x] rgb() and rgba() functions
- [x] hsl() and hsla() functions
- [x] Named colors (140+ CSS colors)
- [x] currentColor keyword

### CSS Units
- [x] px, pt, pc, mm, cm, in
- [x] em, ex, rem, ch
- [x] vw, vh, vmin, vmax
- [x] Percentages

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
- [x] `<mask>` element *(implemented 2025-12-30)*
- [x] `<image>` element with data URLs *(implemented 2025-12-30)*
- [x] `<symbol>` element *(implemented 2025-12-30)*
- [x] visibility attribute *(implemented 2025-12-30)*
- [x] RGBA colors (#RGBA, #RRGGBBAA, rgba()) *(implemented 2025-12-30)*
- [x] HSL/HSLA colors *(implemented 2025-12-30)*
- [x] Radial gradient focal radius (fr) *(implemented 2025-12-30)*
- [x] `<filter>` primitives *(implemented 2025-12-30)* - All major primitives in Rust (86.26% accuracy)
  - feGaussianBlur, feOffset, feFlood, feBlend, feComposite, feMerge
  - feColorMatrix, feComponentTransfer, feMorphology, feConvolveMatrix
  - feTurbulence, feDisplacementMap, feTile, feImage
  - feDiffuseLighting, feSpecularLighting, feDropShadow
  - Light sources: FeDistantLight, FePointLight, FeSpotLight
- [ ] stroke-dashoffset
- [ ] `<pattern>` fills (needed for higher filter accuracy)
- [ ] `<marker>` for arrowheads
- [ ] External file references in `<image>`
- [ ] Embedded SVG in `<image>`
- [ ] Filter subregion calculations

### Low Priority
- [ ] `<tspan>` advanced features
- [ ] `<style>` block CSS rules
- [ ] Symbol viewBox handling
- [ ] feSpecularLighting accuracy improvements
- [ ] feTile implementation improvements

---

## Testing & Quality Assurance

### Test Suite Locations

| Collection | SVG Source | Reference PNGs | Tests |
|------------|-----------|----------------|-------|
| emojis | `SciStagEssentialData/images/noto/emojis/svg/` | `references/emojis/resvg/` | 3427 |
| flags | `SciStagEssentialData/images/noto/flags/svg/` | `references/flags/resvg/` | 358 |
| fontawesome | `advanced_svg/fontawesome/fa/fontawesome-free-6.4.2-web/svgs/` | `references/fontawesome/resvg/` | 2028 |
| material | `advanced_svg/material/` | `references/material/resvg/` | 336 |
| lucide | `advanced_svg/lucide/` | `references/lucide/resvg/` | 200 |
| w3c | `samples/svg/` | `references/w3c/resvg/` | 30 |
| resvg-test-suite | `resvg-test-suite/tests/` | Built-in `.png` files | 1679 |

### Running Tests

**Quick verification (run after changes):**
```bash
# Verify accuracy hasn't regressed (should all be >99%)
python svg_compare.py compare --emojis --limit 200
python svg_compare.py compare --flags --limit 200
python svg_compare.py compare --fontawesome --limit 200
```

**Full test suite:**
```bash
# Full accuracy test all collections
python svg_compare.py compare --all -j 16

# resvg-test-suite benchmark (1679 tests)
python benchmark_resvg_tests.py

# Filter-specific benchmark
python quick_filter_bench.py
```

**Pre-render references (one-time setup):**
```bash
python svg_compare.py prerender --all -j 16
```

### Expected Results (2025-12-30)

| Test Suite | Accuracy | Threshold |
|------------|----------|-----------|
| Emojis | 99.9% | >99% |
| Flags | 99.9% | >99% |
| FontAwesome | 100.0% | >99% |
| Material | 99.0% | >95% |
| Lucide | 98.7% | >95% |
| W3C | 99.3% | >95% |
| resvg-test-suite | 89.1% | >85% |
| resvg filters | 90.7% | >85% |

### Performance Benchmarks

| Collection | VectorStag | resvg | Ratio |
|------------|------------|-------|-------|
| FontAwesome (128x128) | ~159 files/sec | ~156 files/sec | 1.0x (parity) |
| Emojis (200x200) | ~49 files/sec | ~295 files/sec | 5.6x slower |

Note: Complex SVGs (emojis with many gradients/paths) are slower due to Python overhead. Simple icons render at parity with native Rust resvg.

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

### 4. benchmark_resvg_tests.py - resvg Test Suite Benchmark
```bash
# Clone the test suite first (one-time setup)
git clone https://github.com/nicubunu/resvg-test-suite.git

# Run full benchmark (1679 tests)
python benchmark_resvg_tests.py

# Run specific category
python benchmark_resvg_tests.py --category shapes
python benchmark_resvg_tests.py --category text
python benchmark_resvg_tests.py --category filters
python benchmark_resvg_tests.py --category masking
python benchmark_resvg_tests.py --category paint-servers
python benchmark_resvg_tests.py --category painting
python benchmark_resvg_tests.py --category structure

# Run subcategory
python benchmark_resvg_tests.py --category structure/symbol
python benchmark_resvg_tests.py --category masking/mask

# Limit number of tests
python benchmark_resvg_tests.py --limit 100

# Use multiple workers (may have memory issues)
python benchmark_resvg_tests.py -j 4
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
