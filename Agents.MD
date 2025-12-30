# VectorStag Agent Guide

This document describes how to work with VectorStag as an AI agent.

## Project Overview

VectorStag is a pure Python SVG renderer that uses only Pillow (and optionally OpenCV) without external SVG libraries. The goal is visual parity with CairoSVG and resvg.

## Directory Structure

```
vectorstag/          # Main Python package
  __init__.py        # Public API
  parser.py          # SVG DOM parsing
  path_parser.py     # Path 'd' attribute parsing
  renderer.py        # Pillow rendering
vectorstag_rust/     # Rust extension for performance
  src/lib.rs         # All Rust implementations
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
2. For performance-critical code, implement in `vectorstag_rust/src/lib.rs`
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

## Key Files Reference

| File | Purpose |
|------|---------|
| `vectorstag/parser.py` | SVG element parsing, transforms, styles |
| `vectorstag/renderer.py` | Shape rendering, gradients, filters |
| `vectorstag_rust/src/lib.rs` | Rust implementations (blur, fill, resize) |
| `svg_compare.py` | Main test runner |
| `benchmark_resvg_tests.py` | resvg test suite runner |
| `plan.md` | Project status and implementation notes |
| `CLAUDE.md` | Quick notes on known rendering quirks |

## Performance Notes

- Simple icons (FontAwesome): On par with resvg
- Complex SVGs (emojis): ~5.6x slower due to Python overhead
- Filters use Rust for O(1) blur algorithms
- All polygon fills use Rust when available

## Known Limitations

- Pattern fills not implemented
- Filter subregion calculations incomplete
- External file references in `<image>` not supported
- Some lighting calculations need improvement (feSpecularLighting)
