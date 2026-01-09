# VectorStag Development Notes

The goal is to development an SVG rendering Library in Python and RUST which is in regards
of quality on par with the libraries RESVG and Cairo. For rust we are building the sub package
vectorstag_rust. We can benchmark this quality with the tools below.

When we are working on a larger plan we always document our steps in plan.md. We are using
the poetry package manager for our environment.

## Rendering Accuracy Notes

### Verified with resvg (Rust-based accurate renderer)
We installed resvg-python as an alternative reference renderer. resvg confirms:
- **clippath.svg**: Our RED rendering is correct. CairoSVG's BLACK is wrong.
- **lineargradient1/2.svg**: Our gap rendering is correct. CairoSVG filling it is wrong.

## Tools

### scripts/benchmark.py - Fast Benchmark
Compares VectorStag against pre-rendered resvg PNGs from resvg-test-suite.
```bash
# Full benchmark with table output
poetry run python scripts/benchmark.py -j 16

# Just icon collections
poetry run python scripts/benchmark.py --collections -j 16

# Just resvg test suite
poetry run python scripts/benchmark.py --resvg -j 16

# Specific resvg category
poetry run python scripts/benchmark.py --resvg --resvg-category shapes -j 16

# List available resvg categories
poetry run python scripts/benchmark.py --list-resvg-categories
```

### svg_compare.py - Multi-Renderer Comparison
Compares VectorStag against multiple renderers (resvg, Cairo, Chrome).

**Commands:**
- `prerender` - Pre-render reference images with Cairo, resvg, and Chrome
- `compare` - Compare VectorStag against pre-rendered references
- `matrix` - Pairwise similarity matrix across all renderers
- `list` - List available collections

**Collections:** emojis, flags, material, lucide, fontawesome, w3c, resvgtests

```bash
# Pre-render references (run once per collection)
poetry run python svg_compare.py prerender --emojis --flags -j 8
poetry run python svg_compare.py prerender --all -j 8
poetry run python svg_compare.py prerender --emojis --no-cairo --no-resvg -j 8  # Chrome only
poetry run python svg_compare.py prerender --emojis --force -j 8  # Re-render existing

# Compare VectorStag against references
poetry run python svg_compare.py compare --emojis -j 8
poetry run python svg_compare.py compare --emojis --save -j 8  # Save comparison grids

# Multi-renderer similarity matrix
poetry run python svg_compare.py matrix --emojis -j 8
poetry run python svg_compare.py matrix --emojis --save-top 10 -j 8  # Save worst 10 grids

# List collections and their status
poetry run python svg_compare.py list
```

**Output directories:**
- `references/<collection>/{cairo,resvg,chrome}/` - Pre-rendered PNGs
- `comparisons/<collection>/` - Comparison grid images

### render.py - Single File Rendering
```bash
poetry run python render.py input.svg output.png -b white
poetry run python render.py input.svg output.png -w 500 -H 500
```

## Accuracy Expectations
- **Anti-aliasing differences account for AT MOST 0.1%** - anything beyond that indicates real rendering bugs
- **Target: 99.9%+ for all SVGs** - especially simple shapes like android.svg
- **Pink Pixel Rule**: Comparison grids must NOT have more than 9 pink pixels (differences) in any 3x3 pixel area. Anything beyond that is a rendering error, not aliasing.

## Development Guidelines
- **ALWAYS use 4x antialiasing**: The quality difference between 2x and 4x is massive. DO NOT CHANGE this default under any circumstances.
- **ALWAYS use at least 8 workers, preferably 16**: `-j 16` (never run single-threaded benchmarks)
- Use 4-8 workers if memory pressure occurs on complex SVGs: `-j 4` or `-j 8`
- Use `ProcessPoolExecutor` with `as_completed()` for timeout support
- Worker timeout is 30 seconds (WORKER_TIMEOUT in svg_compare.py)

## How to Build

```bash
cd /projects/VectorStag

# Install dependencies
poetry install

# Build the Rust extension
poetry run maturin develop --release -m vectorstag_rust/Cargo.toml
```

## Key Files

- Main renderer: `vectorstag_rust/src/svg_renderer.rs` (~2200 lines)
- Benchmark script: `scripts/benchmark.py`
- Multi-renderer comparison: `svg_compare.py`
- Python wrapper: `vectorstag/rust_renderer.py`
- Status doc: `plan.md`

## Todos
- [ ] Implement efficient clipPath with render-to-temp approach
- [ ] Implement mask support
- [ ] Implement basic filters (feGaussianBlur, feColorMatrix)
- [ ] Implement text rendering
- [ ] Implement preserveAspectRatio parsing
- [ ] Fix gradientTransform for objectBoundingBox