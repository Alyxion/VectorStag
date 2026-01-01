# VectorStag Agent Guide

This document describes how to work with VectorStag as an AI agent.

## Project Overview

VectorStag is a hybrid Python/Rust SVG renderer that uses Pillow for rendering and a high-performance Rust extension for analytical anti-aliasing, gradients, and path operations. The goal is visual parity with CairoSVG and resvg.

## Directory Structure

```
vectorstag/          # Main Python package
  __init__.py        # Public API
  parser.py          # SVG DOM parsing
  path_parser.py     # Path 'd' attribute parsing
  renderer.py        # Pillow rendering
vectorstag_rust/     # Rust extension for performance
  src/lib.rs         # Core algorithms, path parsing, gradients
  src/canvas.rs      # Analytical anti-aliasing & primitives
samples/             # Test SVG files
  svg/               # W3C samples
advanced_svg/        # Additional test suites
  fontawesome/       # FontAwesome icons
  material/          # Material Design icons
  lucide/            # Lucide icons
SciStagEssentialData/
  images/noto/
    emojis/svg/      # 3427 Noto emoji SVGs
    flags/svg/       # 358 Noto flag SVGs
references/          # Pre-rendered reference PNGs (resvg)
  emojis/resvg/
  flags/resvg/
  fontawesome/resvg/
resvg-test-suite/    # Clone of resvg test suite (1679 tests)
  tests/             # SVG test files with reference PNGs
```

## Running Tests

### Quick Regression Check (run after any change)

```bash
# These should all report >99% accuracy
python svg_compare.py compare --emojis --limit 200
python svg_compare.py compare --flags --limit 200
python svg_compare.py compare --fontawesome --limit 200
```

### Full Test Suite

```bash
# All collections against resvg references
python svg_compare.py compare --all -j 16

# resvg-test-suite (includes filters, masks, etc.)
python benchmark_resvg_tests.py

# Filter-specific tests
python quick_filter_bench.py
```

### Pre-rendering References (one-time)

```bash
python svg_compare.py prerender --all -j 16
```

## Expected Accuracy Thresholds

| Suite | Target | Current |
|-------|--------|---------|
| Emojis | >99% | 99.9% |
| Flags | >99% | 99.9% |
| FontAwesome | >99% | 100% |
| Material | >95% | 99% |
| resvg-test-suite | >85% | 89% |

## Common Development Tasks

### Adding a New Feature

1. Implement in `vectorstag/parser.py` (parsing) and/or `vectorstag/renderer.py` (rendering)
2. For performance-critical code, implement in `vectorstag_rust/src/lib.rs` or `vectorstag_rust/src/canvas.rs`
3. Run quick regression check
4. If adding filters, run `python quick_filter_bench.py`

### Rebuilding Rust Extension

```bash
cd vectorstag_rust && maturin develop --release && cd ..
```

### Debugging a Specific SVG

```bash
# Render and compare
python svg_compare.py compare --file path/to/file.svg --save

# View comparison grid in comparisons/ directory
```

### Performance Profiling

```bash
python benchmark.py --file samples/svg/tiger.svg --profile
```

## Source Code & Scripts Reference

### Source Code

| File | Purpose |
|------|---------|
| `vectorstag/parser.py` | SVG element parsing, transforms, and style inheritance |
| `vectorstag/renderer.py` | Main Python orchestration for shape rendering |
| `vectorstag_rust/src/lib.rs` | Rust core: path parsing, gradients, polygon ops |
| `vectorstag_rust/src/canvas.rs` | Rust canvas: analytical anti-aliasing, primitives |

### Root Scripts

#### `svg_compare.py`
**Goal:** The primary regression testing tool. Handles pre-rendering references (CairoSVG/resvg) and comparing VectorStag's output against them.
- **Commands:**
  - `prerender`: Generate reference PNGs.
    - `--all` or specific collections (`--emojis`, `--flags`, etc.)
    - `-j [N]`: Number of workers.
  - `compare`: Run comparison tests.
    - `--save`: Save comparison grids (VectorStag | Reference | Diff) for failures.
    - `--limit [N]`: Limit number of files per collection.
  - `list`: Show available collections and status.

#### `benchmark_resvg_tests.py`
**Goal:** Runs the comprehensive `resvg-test-suite` (1679 tests) to verify compliance across all SVG features (shapes, masking, filters, etc.).
- **Parameters:**
  - `-j [N]`: Workers (default 1, recommend 4-8).
  - `--category [name]`: Run only specific category (e.g., `filters`, `masking`).
  - `--save-failures`: Generate diff images for failing tests.
  - `--antialias [1|2|4]`: Set AA level (default 4).

#### `benchmark.py`
**Goal:** Measure rendering performance and profile execution time.
- **Parameters:**
  - `--file [path]`: Benchmark a single file.
  - `--profile`: Enable cProfile for detailed function-level timing.
  - `--all` / collection flags: Benchmark entire collections.
  - `-j [N]`: Worker count.

#### `render.py`
**Goal:** Simple CLI utility to render a single SVG to an image file or stdout.
- **Parameters:**
  - `input`: Input SVG path.
  - `output`: Output PNG path (or `-` for stdout).
  - `-w`, `-H`, `-s`: Width, height, or square size.
  - `-b`: Background color (default: transparent).
  - `-a`: Anti-aliasing level (default: 4).

#### `run_all_benchmarks.py`
**Goal:** CI/CD-style wrapper that runs both `benchmark_resvg_tests.py` and `svg_compare.py` to report a final pass/fail score.
- **Parameters:** None (hardcoded configuration).

#### `quick_filter_bench.py` / `quick_filter_test.py`
**Goal:** Specialized micro-benchmarks for developing and testing filter implementations.

### Helper Scripts (`scripts/`)

#### `scripts/detect_gaps.py`
**Goal:** Analyzes rendered images to detect specific artifacts like interior gaps or broken lines, which simple pixel diffs might miss or confuse with anti-aliasing.
- **Parameters:**
  - `--categories`: Select collections to check.
  - `--file`: Analyze a single rendered image/SVG.
  - `--save-bad`: Save images where gaps were detected.
  - `--threshold`: Sensitivity threshold.

#### `scripts/visual_inspect.py`
**Goal:** Generates a random sample of renders from various collections and creates an HTML gallery for manual visual verification.
- **Parameters:**
  - `--count`: Number of random samples per category.
  - `--categories`: Select collections.
  - `--output`: Output directory for the gallery.

## Performance Notes

- **Hybrid Architecture**: Critical paths (fill, stroke, gradients, blur) are implemented in Rust.
- **Analytical Anti-Aliasing**: Implemented in Rust (`canvas.rs`) for high quality without supersampling overhead.
- **Direct Compositing**: Rust extension writes directly to memory buffers to avoid Python/Pillow overhead.
- **Filters**: Rust implementations for expensive operations like Gaussian blur.

## Known Limitations

- Pattern fills not implemented
- Filter subregion calculations incomplete
- External file references in `<image>` not supported
- Some lighting calculations need improvement (feSpecularLighting)
