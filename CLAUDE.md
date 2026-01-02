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

## Tools (Consolidated)
- `scripts/benchmark.py` - Full benchmark (all collections + resvg-test-suite)
- `svg_compare.py` - Main comparison tool for errors on single images
- `render.py` - Simple single-file rendering

## Testing

### Full Benchmark (Collections + resvg-test-suite)
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

## Accuracy Expectations
- **Anti-aliasing differences account for AT MOST 0.1%** - anything beyond that indicates real rendering bugs
- **Target: 99.9%+ for all SVGs** - especially simple shapes like android.svg

## Development Guidelines
- **ALWAYS use 4x antialiasing**: The quality difference between 2x and 4x is massive. DO NOT CHANGE this default under any circumstances.
- **ALWAYS use at least 8 workers, preferably 16**: `-j 16` (never run single-threaded benchmarks)
- Use 4-8 workers if memory pressure occurs on complex SVGs: `-j 4` or `-j 8`
- Use `ProcessPoolExecutor` with `as_completed()` for timeout support
- Worker timeout is 30 seconds (WORKER_TIMEOUT in svg_compare.py)
