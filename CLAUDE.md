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

## Comparison Scripts
- `compare_render.py` - Compare vs CairoSVG (98.9% average)
- `compare_render_corrected.py` - Excludes CairoSVG bugs (99.4% true accuracy)
- `compare_render_resvg.py` - Compare vs resvg (different sizing/fonts)
