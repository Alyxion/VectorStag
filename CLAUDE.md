# VectorStag Development Notes

## Rendering Accuracy Notes

### Verified with resvg (Rust-based accurate renderer)
We installed resvg-python as an alternative reference renderer. resvg confirms:
- **clippath.svg**: Our RED rendering is correct. CairoSVG's BLACK is wrong.
- **lineargradient1/2.svg**: Our gap rendering is correct. CairoSVG filling it is wrong.

### clippath.svg - IGNORE IN RATING
Our clippath rendering is **CORRECT** (verified: Chrome, Firefox, resvg).
CairoSVG renders it incorrectly (shows black instead of red for intersection).

### lineargradient1/2.svg - IGNORE IN RATING
The SVG has a **real gap** (~3 units) between adjacent rectangles.
Our rendering is **CORRECT** (verified: Chrome, Firefox, resvg).
CairoSVG incorrectly fills this gap.

### Known CairoSVG Bugs/Quirks
- CairoSVG clippath intersection rendering is buggy (renders black instead of red)
- CairoSVG fills gaps between adjacent elements that don't exist in SVG
- CairoSVG doesn't properly apply Gaussian blur filters
- CairoSVG stretches content when viewBox has NEGATIVE origin coordinates (e.g., BR.svg)
  - With positive viewBox origin: preserves aspect ratio
  - With negative viewBox origin: stretches to fill output dimensions

## Completed Fixes
- **android.svg**: Fixed hard edges and gaps in stroke joins (Segment+Join approach). Score: 99.9%+
- **paths-data-08/09-t**: Fixed evenodd multi-polygon fill (87% → 97%)
- Triangle cutout now works correctly with fill-rule="evenodd"

## BR.svg (Brazilian Flag) - IGNORE IN RATING
Our rendering is **CORRECT**. CairoSVG sizes it incorrectly.
When in doubt, use resvg as the reference renderer.

## Comparison Settings
- Use at least 3x resolution for comparisons to preserve details
- Use resvg as reference when CairoSVG results are questionable

## Flag/Emoji Rendering Issues (To Fix)

### Stars Not Filled (fill-rule issue?)
- BA.svg: stars should be filled
- CF.svg: star should be filled
- CN.svg: star should be filled
- CU.svg: star should be filled
- DZ.svg: star should be filled
- GF.svg: star should be filled
- GH.svg: star should be filled
- KM.svg: stars should be filled
- LY.svg: star should be filled

### Line Thickness Issues
- AF.svg: lack of transparency or too thick lines
- AR.svg: too thick lines, sun details are finer
- BL.svg: too thick lines

### Color/Rendering Issues
- AS.svg: complete failure - eagle on blue flag not visible
- EC.svg: colors of symbol slightly off
- GB.svg: Union Jack issues - red bleeding on top/bottom of horizontal red stripes
- GL.svg: lower half of circle should be white on top of red stripe
- MX-SON.svg: triangle in center top should be white, not black
- NP.svg: blue/white triangle on center left where it doesn't belong

## Tools (Consolidated)
- `svg_compare.py` - Main comparison tool (prerender, compare, list)
- `benchmark.py` - Performance benchmarking and profiling
- `render.py` - Simple single-file rendering
- `benchmark_resvg_tests.py` - Benchmark against resvg-test-suite (1679 tests)

## Testing

### Quick Regression Check
```bash
# Check all icon collections (should maintain 98%+ accuracy)
python svg_compare.py compare --all

# Expected results:
# - Flags: 99.9%
# - Emojis: 99.9%
# - Material: 99.0%
# - Lucide: 98.6%
# - FontAwesome: 100%
# - W3C: 99.3%
```

### resvg-test-suite Benchmark
```bash
# Full benchmark (1679 tests, ~10 minutes single-threaded)
python benchmark_resvg_tests.py

# Expected: 90% average accuracy
# Categories: shapes (97%), text (96%), painting (94%), masking (93%),
#             paint-servers (90%), structure (88%), filters (80%)

# Test specific category
python benchmark_resvg_tests.py --category shapes
python benchmark_resvg_tests.py --category masking/mask
```

## Accuracy Expectations
- **Anti-aliasing differences account for AT MOST 0.1%** - anything beyond that indicates real rendering bugs
- **Target: 99.9%+ for all SVGs** - especially simple shapes like android.svg

## Development Guidelines
- **ALWAYS use 4x antialiasing**: The quality difference between 2x and 4x is massive. DO NOT CHANGE this default under any circumstances.
- **ALWAYS use at least 8 workers, preferably 16**: `-j 16` (never run single-threaded benchmarks)
- Use 4-8 workers if memory pressure occurs on complex SVGs: `-j 4` or `-j 8`
- Use `ProcessPoolExecutor` with `as_completed()` for timeout support
- Worker timeout is 30 seconds (WORKER_TIMEOUT in svg_compare.py)
- Parser/renderer have MAX_PARSE_DEPTH=100 and MAX_RENDER_DEPTH=100 limits
- Circular `<use>` references are detected and prevented via `_use_stack`
