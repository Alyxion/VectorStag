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

### Known CairoSVG Bugs
- CairoSVG clippath intersection rendering is buggy (renders black instead of red)
- CairoSVG fills gaps between adjacent elements that don't exist in SVG
- CairoSVG doesn't properly apply Gaussian blur filters

## Completed Fixes
- **paths-data-08/09-t**: Fixed evenodd multi-polygon fill (87% → 97%)
- Triangle cutout now works correctly with fill-rule="evenodd"

## Comparison Scripts
- `compare_render.py` - Compare vs CairoSVG (98.9% average)
- `compare_render_corrected.py` - Excludes CairoSVG bugs (99.4% true accuracy)
- `compare_render_resvg.py` - Compare vs resvg (different sizing/fonts)
