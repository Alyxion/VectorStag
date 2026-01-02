#!/usr/bin/env python3
"""
Unified benchmark tool for VectorStag.

Runs comparisons against all SVG collections and resvg-test-suite,
outputs a formatted summary table with accuracy and performance metrics.

Usage:
    # Full benchmark (all collections + resvg tests)
    python benchmark.py -j 16

    # Just icon collections
    python benchmark.py --collections -j 16

    # Just resvg test suite
    python benchmark.py --resvg -j 16

    # Specific categories
    python benchmark.py --emojis --flags -j 16

    # Profile a single file
    python benchmark.py --file samples/svg/tiger.svg --profile
"""

import argparse
import cProfile
import gc
import io as sysio
import pstats
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports for renderers to avoid issues with multiprocessing fork
SVGRenderer = None
RustSVGRenderer = None


def get_renderer(use_rust: bool, **kwargs):
    """Get a renderer instance with lazy import."""
    global SVGRenderer, RustSVGRenderer
    if use_rust:
        if RustSVGRenderer is None:
            from vectorstag.rust_renderer import RustSVGRenderer as _RustSVGRenderer
            RustSVGRenderer = _RustSVGRenderer
        return RustSVGRenderer(**kwargs)
    else:
        if SVGRenderer is None:
            from vectorstag import SVGRenderer as _SVGRenderer
            SVGRenderer = _SVGRenderer
        return SVGRenderer(**kwargs)


# Constants
DEFAULT_TIMEOUT = 30  # seconds per test
BATCH_SIZE = 100  # restart workers after this many tests

# Global flag for using Rust renderer
USE_RUST_RENDERER = False

# SVGs known to cause infinite loops or extreme slowdowns
SKIP_SVGS = {
    "filters/feMorphology/huge-radius.svg",  # feMorphology radius=9999 causes infinite loop
    "filters/filter/huge-region.svg",  # Takes 4+ seconds due to huge filter region
}

# Categories to skip entirely (too slow with 4x supersampling)
SKIP_CATEGORIES = {
    "filters/feMorphology",  # Morphology operations are O(n*radius^2), extremely slow at 4x
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result for a single SVG test."""
    name: str
    category: str
    similarity: Optional[float] = None
    render_time_ms: float = 0.0
    resvg_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class CategoryStats:
    """Statistics for a category."""
    name: str
    count: int = 0
    valid: int = 0
    errors: int = 0
    avg_similarity: float = 0.0
    bucket_99_100: int = 0
    bucket_95_99: int = 0
    bucket_below_95: int = 0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    avg_resvg_time_ms: float = 0.0
    max_resvg_time_ms: float = 0.0
    total_time_s: float = 0.0


@dataclass
class Collection:
    """SVG collection configuration."""
    name: str
    svg_dir: Path
    ref_dir: Path
    size: int = 400
    description: str = ""


# =============================================================================
# Collection Definitions
# =============================================================================

def get_collections() -> Dict[str, Collection]:
    """Get all available SVG collections."""
    noto_dir = Path("SciStagEssentialData/images/noto")
    base_ref_dir = Path("references")

    return {
        "emojis": Collection(
            name="Emojis",
            svg_dir=noto_dir / "emojis" / "svg",
            ref_dir=base_ref_dir / "emojis",
            size=400,
            description="Noto Color Emojis"
        ),
        "flags": Collection(
            name="Flags",
            svg_dir=noto_dir / "flags" / "svg",
            ref_dir=base_ref_dir / "flags",
            size=400,
            description="Noto Flags"
        ),
        "material": Collection(
            name="Material",
            svg_dir=Path("advanced_svg/material"),
            ref_dir=base_ref_dir / "material",
            size=256,
            description="Material Design Icons"
        ),
        "fontawesome": Collection(
            name="FontAwesome",
            svg_dir=Path("advanced_svg/fontawesome"),
            ref_dir=base_ref_dir / "fontawesome",
            size=128,
            description="FontAwesome Icons"
        ),
        "lucide": Collection(
            name="Lucide",
            svg_dir=Path("advanced_svg/lucide"),
            ref_dir=base_ref_dir / "lucide",
            size=128,
            description="Lucide Icons"
        ),
        "w3c": Collection(
            name="W3C",
            svg_dir=Path("samples/svg"),
            ref_dir=base_ref_dir / "w3c",
            size=400,
            description="W3C SVG Test Suite"
        ),
    }


# =============================================================================
# Image Utilities
# =============================================================================

def fit_to_canvas(img: Image.Image, size: int) -> Image.Image:
    """Fit image to canvas, centered with transparent background."""
    if img.size == (size, size):
        return img

    scale = min(size / img.width, size / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)

    if new_w != img.width or new_h != img.height:
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    canvas.paste(img, (offset_x, offset_y))

    return canvas


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity between two images (0.0 - 1.0)."""
    if img1 is None or img2 is None:
        return 0.0

    size = (max(img1.width, img2.width), max(img1.height, img2.height))
    if img1.size != size:
        img1 = img1.resize(size, Image.Resampling.LANCZOS)
    if img2.size != size:
        img2 = img2.resize(size, Image.Resampling.LANCZOS)

    white = Image.new("RGBA", size, (255, 255, 255, 255))
    img1_comp = Image.alpha_composite(white, img1)
    img2_comp = Image.alpha_composite(white, img2)

    arr1 = np.array(img1_comp, dtype=np.float32)[:, :, :3] / 255.0
    arr2 = np.array(img2_comp, dtype=np.float32)[:, :, :3] / 255.0

    mse = np.mean((arr1 - arr2) ** 2)
    return max(0.0, 1.0 - min(1.0, mse * 4))


def compute_similarity_premultiplied(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity using premultiplied alpha (for resvg tests)."""
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.LANCZOS)

    arr1 = np.array(img1.convert('RGBA'), dtype=np.float32)
    arr2 = np.array(img2.convert('RGBA'), dtype=np.float32)

    # Check if reference has opaque white background
    ref_alpha_mean = arr2[:, :, 3].mean()
    if ref_alpha_mean > 250:
        corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        white_corners = sum(1 for y, x in corners
                           if arr2[y, x, 0] > 240 and arr2[y, x, 1] > 240 and arr2[y, x, 2] > 240)
        if white_corners >= 3:
            a1_ratio = arr1[:, :, 3:4] / 255.0
            white_bg = np.ones_like(arr1[:, :, :3]) * 255
            arr1[:, :, :3] = arr1[:, :, :3] * a1_ratio + white_bg * (1 - a1_ratio)
            arr1[:, :, 3] = 255

    a1 = arr1[:, :, 3:4] / 255.0
    a2 = arr2[:, :, 3:4] / 255.0

    rgb1 = arr1[:, :, :3] * a1
    rgb2 = arr2[:, :, :3] * a2

    rgb_diff = np.abs(rgb1 - rgb2).mean()
    alpha_diff = np.abs(arr1[:, :, 3] - arr2[:, :, 3]).mean()
    combined_diff = (rgb_diff + alpha_diff) / 2

    return max(0, 100 * (1 - combined_diff / 255)) / 100.0


# =============================================================================
# Collection Benchmark
# =============================================================================

def get_unique_name(svg_path: Path, base_dir: Path) -> str:
    """Get unique name for an SVG file."""
    try:
        rel_path = svg_path.relative_to(base_dir)
        parts = list(rel_path.parts)
        parts[-1] = parts[-1].replace('.svg', '')
        return '_'.join(parts)
    except ValueError:
        return svg_path.stem


def benchmark_collection_worker(args) -> BenchmarkResult:
    """Worker function for collection benchmark."""
    svg_path, base_dir, resvg_ref_dir, cairo_ref_dir, size, category, compare_resvg, use_rust = args
    name = get_unique_name(svg_path, base_dir)

    resvg_time = 0.0

    try:
        # Load references
        resvg_path = resvg_ref_dir / f"{name}.png"
        cairo_path = cairo_ref_dir / f"{name}.png" if cairo_ref_dir else None

        resvg_img = Image.open(resvg_path).convert("RGBA") if resvg_path.exists() else None
        cairo_img = Image.open(cairo_path).convert("RGBA") if cairo_path and cairo_path.exists() else None

        # Benchmark resvg if requested
        if compare_resvg:
            try:
                from resvg_python import svg_to_png
                with open(svg_path, 'r') as f:
                    svg_content = f.read()
                resvg_start = time.perf_counter()
                _ = svg_to_png(svg_content)
                resvg_time = (time.perf_counter() - resvg_start) * 1000
            except Exception:
                resvg_time = 0.0

        # Render with VectorStag
        start_time = time.perf_counter()
        renderer = get_renderer(use_rust, background=(0, 0, 0, 0), antialias=4)
        vs_img = renderer.render_file(str(svg_path))
        render_time = (time.perf_counter() - start_time) * 1000

        if vs_img is None:
            return BenchmarkResult(name, category, error="Render failed", render_time_ms=render_time, resvg_time_ms=resvg_time)

        vs_img = fit_to_canvas(vs_img, size)

        # Compute similarity
        sim_resvg = compute_similarity(vs_img, resvg_img) if resvg_img else 0.0
        sim_cairo = compute_similarity(vs_img, cairo_img) if cairo_img else 0.0
        sim = max(sim_resvg, sim_cairo)

        del renderer, vs_img, resvg_img, cairo_img
        gc.collect()

        return BenchmarkResult(name, category, similarity=sim, render_time_ms=render_time, resvg_time_ms=resvg_time)

    except Exception as e:
        return BenchmarkResult(name, category, error=str(e)[:100], render_time_ms=0.0, resvg_time_ms=resvg_time)


def benchmark_collection_worker_str(args) -> BenchmarkResult:
    """Worker function for collection benchmark (string paths for better pickling)."""
    svg_path_str, base_dir_str, resvg_ref_dir_str, cairo_ref_dir_str, size, category, compare_resvg, use_rust = args
    svg_path = Path(svg_path_str)
    base_dir = Path(base_dir_str)
    resvg_ref_dir = Path(resvg_ref_dir_str)
    cairo_ref_dir = Path(cairo_ref_dir_str) if cairo_ref_dir_str else None

    name = get_unique_name(svg_path, base_dir)

    resvg_time = 0.0

    try:
        # Load references
        resvg_path = resvg_ref_dir / f"{name}.png"
        cairo_path = cairo_ref_dir / f"{name}.png" if cairo_ref_dir else None

        resvg_img = Image.open(resvg_path).convert("RGBA") if resvg_path.exists() else None
        cairo_img = Image.open(cairo_path).convert("RGBA") if cairo_path and cairo_path.exists() else None

        # Benchmark resvg if requested
        if compare_resvg:
            try:
                from resvg_python import svg_to_png
                with open(svg_path_str, 'r') as f:
                    svg_content = f.read()
                resvg_start = time.perf_counter()
                _ = svg_to_png(svg_content)
                resvg_time = (time.perf_counter() - resvg_start) * 1000
            except Exception:
                resvg_time = 0.0

        # Render with VectorStag
        start_time = time.perf_counter()
        renderer = get_renderer(use_rust, background=(0, 0, 0, 0), antialias=4)
        vs_img = renderer.render_file(svg_path_str)
        render_time = (time.perf_counter() - start_time) * 1000

        if vs_img is None:
            return BenchmarkResult(name, category, error="Render failed", render_time_ms=render_time, resvg_time_ms=resvg_time)

        vs_img = fit_to_canvas(vs_img, size)

        # Compute similarity
        sim_resvg = compute_similarity(vs_img, resvg_img) if resvg_img else 0.0
        sim_cairo = compute_similarity(vs_img, cairo_img) if cairo_img else 0.0
        sim = max(sim_resvg, sim_cairo)

        del renderer, vs_img, resvg_img, cairo_img
        gc.collect()

        return BenchmarkResult(name, category, similarity=sim, render_time_ms=render_time, resvg_time_ms=resvg_time)

    except Exception as e:
        return BenchmarkResult(name, category, error=str(e)[:100], render_time_ms=0.0, resvg_time_ms=resvg_time)


def benchmark_collection(collection: Collection, num_workers: int, timeout: float = DEFAULT_TIMEOUT, compare_resvg: bool = False) -> CategoryStats:
    """Benchmark a single collection."""
    svg_files = sorted(collection.svg_dir.glob("**/*.svg"))

    if not svg_files:
        return CategoryStats(name=collection.name)

    resvg_ref_dir = collection.ref_dir / "resvg"
    cairo_ref_dir = collection.ref_dir / "cairo"

    if not resvg_ref_dir.exists():
        print(f"  Warning: No references for {collection.name}")
        return CategoryStats(name=collection.name)

    cairo_ref_dir_arg = cairo_ref_dir if cairo_ref_dir.exists() else None

    # Convert Path objects to strings for better pickling
    tasks = [(str(svg_path), str(collection.svg_dir), str(resvg_ref_dir),
              str(cairo_ref_dir) if cairo_ref_dir else None,
              collection.size, collection.name, compare_resvg, USE_RUST_RENDERER) for svg_path in svg_files]

    results: List[BenchmarkResult] = []
    start_time = time.time()

    # Use default context
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(benchmark_collection_worker_str, tasks, chunksize=20)):
            results.append(result)

            completed = i + 1
            if completed % 500 == 0 or completed == len(svg_files):
                valid = [r for r in results if r.similarity is not None]
                avg = sum(r.similarity for r in valid) / len(valid) if valid else 0
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  {completed}/{len(svg_files)} - Avg: {avg:.1%} - {rate:.1f} files/sec")

    total_time = time.time() - start_time

    # Compute stats
    valid_results = [r for r in results if r.similarity is not None]
    error_results = [r for r in results if r.error is not None]

    stats = CategoryStats(name=collection.name)
    stats.count = len(results)
    stats.valid = len(valid_results)
    stats.errors = len(error_results)
    stats.total_time_s = total_time

    if valid_results:
        similarities = [r.similarity for r in valid_results]
        times = [r.render_time_ms for r in valid_results]
        resvg_times = [r.resvg_time_ms for r in valid_results if r.resvg_time_ms > 0]

        stats.avg_similarity = sum(similarities) / len(similarities)
        stats.bucket_99_100 = sum(1 for s in similarities if s >= 0.99)
        stats.bucket_95_99 = sum(1 for s in similarities if 0.95 <= s < 0.99)
        stats.bucket_below_95 = sum(1 for s in similarities if s < 0.95)
        stats.avg_time_ms = sum(times) / len(times)
        stats.max_time_ms = max(times)

        if resvg_times:
            stats.avg_resvg_time_ms = sum(resvg_times) / len(resvg_times)
            stats.max_resvg_time_ms = max(resvg_times)

    return stats


# =============================================================================
# resvg-test-suite Benchmark
# =============================================================================

def benchmark_resvg_worker(args) -> BenchmarkResult:
    """Worker function for resvg test suite (Path objects)."""
    svg_path, ref_path, antialias, use_rust = args

    parts = svg_path.relative_to(Path("resvg-test-suite/tests")).parts
    category = parts[0]
    name = svg_path.stem

    start_time = time.perf_counter()

    try:
        ref_img = Image.open(ref_path).convert('RGBA')
        ref_size = ref_img.size

        renderer = get_renderer(use_rust, background=(0, 0, 0, 0), antialias=antialias)
        vs_img = renderer.render_file(str(svg_path), ref_size[0], ref_size[1])

        render_time = (time.perf_counter() - start_time) * 1000

        if vs_img is None:
            del renderer, ref_img
            gc.collect()
            return BenchmarkResult(name, f"resvg/{category}", error="Render failed", render_time_ms=render_time)

        vs_img = vs_img.convert('RGBA')
        similarity = compute_similarity_premultiplied(vs_img, ref_img)

        del renderer, vs_img, ref_img
        gc.collect()

        return BenchmarkResult(name, f"resvg/{category}", similarity=similarity, render_time_ms=render_time)

    except Exception as e:
        render_time = (time.perf_counter() - start_time) * 1000
        gc.collect()
        return BenchmarkResult(name, f"resvg/{category}", error=str(e)[:100], render_time_ms=render_time)


def benchmark_resvg_worker_str(args) -> BenchmarkResult:
    """Worker function for resvg test suite (string paths for better pickling)."""
    svg_path_str, ref_path_str, antialias, use_rust = args
    svg_path = Path(svg_path_str)
    ref_path = Path(ref_path_str)

    parts = svg_path.relative_to(Path("resvg-test-suite/tests")).parts
    category = parts[0]
    name = svg_path.stem

    start_time = time.perf_counter()

    try:
        ref_img = Image.open(ref_path).convert('RGBA')
        ref_size = ref_img.size

        renderer = get_renderer(use_rust, background=(0, 0, 0, 0), antialias=antialias)
        vs_img = renderer.render_file(svg_path_str, ref_size[0], ref_size[1])

        render_time = (time.perf_counter() - start_time) * 1000

        if vs_img is None:
            del renderer, ref_img
            gc.collect()
            return BenchmarkResult(name, f"resvg/{category}", error="Render failed", render_time_ms=render_time)

        vs_img = vs_img.convert('RGBA')
        similarity = compute_similarity_premultiplied(vs_img, ref_img)

        del renderer, vs_img, ref_img
        gc.collect()

        return BenchmarkResult(name, f"resvg/{category}", similarity=similarity, render_time_ms=render_time)

    except Exception as e:
        render_time = (time.perf_counter() - start_time) * 1000
        gc.collect()
        return BenchmarkResult(name, f"resvg/{category}", error=str(e)[:100], render_time_ms=render_time)


def find_resvg_tests(test_dir: Path, category_filter: str = None) -> List[Tuple[Path, Path]]:
    """Find all resvg test SVGs and their reference PNGs."""
    tests = []
    for svg_path in sorted(test_dir.rglob("*.svg")):
        # Skip known problematic files and categories
        rel_path = str(svg_path.relative_to(test_dir))
        if rel_path in SKIP_SVGS:
            continue
        if any(rel_path.startswith(cat) for cat in SKIP_CATEGORIES):
            continue

        if category_filter:
            # Match category as the first directory component
            parts = svg_path.relative_to(test_dir).parts
            if not parts or parts[0] != category_filter:
                continue
        ref_path = svg_path.with_suffix('.png')
        if ref_path.exists():
            tests.append((svg_path, ref_path))
    return tests


def get_resvg_categories(test_dir: Path) -> List[str]:
    """Get all available resvg test categories."""
    categories = set()
    for svg_path in test_dir.rglob("*.svg"):
        parts = svg_path.relative_to(test_dir).parts
        if parts:
            categories.add(parts[0])
    return sorted(categories)


def benchmark_resvg_suite(num_workers: int, category_filter: str = None, timeout: float = DEFAULT_TIMEOUT) -> List[CategoryStats]:
    """Benchmark against resvg-test-suite, returning stats per category."""
    test_dir = Path("resvg-test-suite/tests")
    if not test_dir.exists():
        print("  Warning: resvg-test-suite not found")
        return [CategoryStats(name="resvg-tests")]

    tests = find_resvg_tests(test_dir, category_filter)
    if not tests:
        return [CategoryStats(name="resvg-tests")]

    # Convert Path objects to strings for pickling
    test_args = [(str(t[0]), str(t[1]), 4, USE_RUST_RENDERER) for t in tests]

    results: List[BenchmarkResult] = []
    start_time = time.time()

    # Use default context
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(benchmark_resvg_worker_str, test_args, chunksize=10)):
            results.append(result)

            # Progress update
            if (i + 1) % 200 == 0:
                valid = [r for r in results if r.similarity is not None]
                avg = sum(r.similarity for r in valid) / len(valid) if valid else 0
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {i + 1}/{len(tests)} - Avg: {avg:.1%} - {rate:.1f} files/sec")

    # Final progress
    valid = [r for r in results if r.similarity is not None]
    avg = sum(r.similarity for r in valid) / len(valid) if valid else 0
    print(f"  {len(results)}/{len(tests)} - Avg: {avg:.1%}")
    gc.collect()

    total_time = time.time() - start_time

    # Group results by category
    category_results: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        # Extract category from "resvg/category" format
        cat = r.category.replace("resvg/", "") if r.category.startswith("resvg/") else r.category
        category_results[cat].append(r)

    # Compute stats per category
    all_stats: List[CategoryStats] = []
    for cat_name in sorted(category_results.keys()):
        cat_results = category_results[cat_name]
        valid_results = [r for r in cat_results if r.similarity is not None]
        error_results = [r for r in cat_results if r.error is not None]

        stats = CategoryStats(name=f"resvg/{cat_name}")
        stats.count = len(cat_results)
        stats.valid = len(valid_results)
        stats.errors = len(error_results)
        stats.total_time_s = total_time * len(cat_results) / len(results) if results else 0

        if valid_results:
            similarities = [r.similarity for r in valid_results]
            times = [r.render_time_ms for r in valid_results]

            stats.avg_similarity = sum(similarities) / len(similarities)
            stats.bucket_99_100 = sum(1 for s in similarities if s >= 0.99)
            stats.bucket_95_99 = sum(1 for s in similarities if 0.95 <= s < 0.99)
            stats.bucket_below_95 = sum(1 for s in similarities if s < 0.95)
            stats.avg_time_ms = sum(times) / len(times)
            stats.max_time_ms = max(times)

        all_stats.append(stats)

    return all_stats


# =============================================================================
# Output Formatting
# =============================================================================

def print_table(all_stats: List[CategoryStats], compare_resvg: bool = False):
    """Print formatted results table."""
    # Check if we have resvg timing data
    has_resvg = compare_resvg and any(s.avg_resvg_time_ms > 0 for s in all_stats)

    if has_resvg:
        width = 135
        print("\n" + "=" * width)
        print("BENCHMARK RESULTS (with resvg comparison)")
        print("=" * width)
        # Header with resvg columns
        print(f"\n| {'Category':<14} | {'Count':>6} | {'Avg':>6} | {'99-100%':>15} | {'95-99%':>13} | {'<95%':>10} | {'VS avg':>7} | {'VS max':>7} | {'resvg avg':>9} | {'resvg max':>9} |")
        print("|" + "-" * 16 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 17 + "|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 11 + "|")
    else:
        width = 105
        print("\n" + "=" * width)
        print("BENCHMARK RESULTS")
        print("=" * width)
        # Header without resvg columns
        print(f"\n| {'Category':<14} | {'Count':>6} | {'Avg':>6} | {'99-100%':>15} | {'95-99%':>13} | {'<95%':>10} | {'Avg ms':>7} | {'Max ms':>7} |")
        print("|" + "-" * 16 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 17 + "|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 9 + "|")

    total_count = 0
    total_valid = 0
    total_99_100 = 0
    total_95_99 = 0
    total_below_95 = 0
    weighted_sim_sum = 0.0

    for stats in all_stats:
        if stats.count == 0:
            continue

        total_count += stats.count
        total_valid += stats.valid
        total_99_100 += stats.bucket_99_100
        total_95_99 += stats.bucket_95_99
        total_below_95 += stats.bucket_below_95
        weighted_sim_sum += stats.avg_similarity * stats.valid

        # Format bucket percentages
        pct_99_100 = f"{stats.bucket_99_100} ({100*stats.bucket_99_100/stats.valid:.1f}%)" if stats.valid else "0"
        pct_95_99 = f"{stats.bucket_95_99} ({100*stats.bucket_95_99/stats.valid:.1f}%)" if stats.valid else "0"
        pct_below_95 = f"{stats.bucket_below_95}" if stats.valid else "0"

        if has_resvg:
            resvg_avg = f"{stats.avg_resvg_time_ms:.1f}" if stats.avg_resvg_time_ms > 0 else "-"
            resvg_max = f"{stats.max_resvg_time_ms:.0f}" if stats.max_resvg_time_ms > 0 else "-"
            print(f"| {stats.name:<14} | {stats.count:>6} | {stats.avg_similarity:>5.1%} | {pct_99_100:>15} | {pct_95_99:>13} | {pct_below_95:>10} | {stats.avg_time_ms:>6.1f} | {stats.max_time_ms:>6.0f} | {resvg_avg:>9} | {resvg_max:>9} |")
        else:
            print(f"| {stats.name:<14} | {stats.count:>6} | {stats.avg_similarity:>5.1%} | {pct_99_100:>15} | {pct_95_99:>13} | {pct_below_95:>10} | {stats.avg_time_ms:>6.1f} | {stats.max_time_ms:>6.0f} |")

    # Total row
    if has_resvg:
        print("|" + "-" * 16 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 17 + "|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 11 + "|")
    else:
        print("|" + "-" * 16 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 17 + "|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 9 + "|")

    total_avg = weighted_sim_sum / total_valid if total_valid else 0
    pct_99_100 = f"{total_99_100} ({100*total_99_100/total_valid:.1f}%)" if total_valid else "0"
    pct_95_99 = f"{total_95_99} ({100*total_95_99/total_valid:.1f}%)" if total_valid else "0"
    pct_below_95 = f"{total_below_95}" if total_valid else "0"

    avg_time = sum(s.avg_time_ms * s.valid for s in all_stats) / total_valid if total_valid else 0
    max_time = max((s.max_time_ms for s in all_stats if s.valid), default=0)

    if has_resvg:
        resvg_valid = [s for s in all_stats if s.avg_resvg_time_ms > 0]
        total_resvg_valid = sum(s.valid for s in resvg_valid)
        avg_resvg = sum(s.avg_resvg_time_ms * s.valid for s in resvg_valid) / total_resvg_valid if total_resvg_valid else 0
        max_resvg = max((s.max_resvg_time_ms for s in resvg_valid), default=0)
        print(f"| {'TOTAL':<14} | {total_count:>6} | {total_avg:>5.1%} | {pct_99_100:>15} | {pct_95_99:>13} | {pct_below_95:>10} | {avg_time:>6.1f} | {max_time:>6.0f} | {avg_resvg:>8.1f} | {max_resvg:>8.0f} |")
    else:
        print(f"| {'TOTAL':<14} | {total_count:>6} | {total_avg:>5.1%} | {pct_99_100:>15} | {pct_95_99:>13} | {pct_below_95:>10} | {avg_time:>6.1f} | {max_time:>6.0f} |")

    # Summary
    total_time = sum(s.total_time_s for s in all_stats)
    print(f"\nTotal benchmark time: {total_time:.1f}s")
    print(f"Overall throughput: {total_count/total_time:.1f} files/sec")

    # Speed comparison summary
    if has_resvg:
        resvg_valid = [s for s in all_stats if s.avg_resvg_time_ms > 0]
        if resvg_valid:
            total_resvg_valid = sum(s.valid for s in resvg_valid)
            avg_vs = sum(s.avg_time_ms * s.valid for s in resvg_valid) / total_resvg_valid
            avg_resvg = sum(s.avg_resvg_time_ms * s.valid for s in resvg_valid) / total_resvg_valid
            if avg_resvg > 0:
                ratio = avg_vs / avg_resvg
                if ratio > 1:
                    print(f"VectorStag is {ratio:.1f}x slower than resvg on average")
                else:
                    print(f"VectorStag is {1/ratio:.1f}x faster than resvg on average")


# =============================================================================
# Single File Profiling (from old benchmark.py)
# =============================================================================

def benchmark_single(svg_path: Path, size: int = 400) -> Tuple[str, float, bool]:
    """Benchmark rendering a single SVG. Returns (name, time_ms, success)."""
    name = svg_path.stem
    try:
        renderer = get_renderer(USE_RUST_RENDERER, background=(0, 0, 0, 0), antialias=4)
        start = time.perf_counter()
        img = renderer.render_file(str(svg_path), size, size)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        return name, elapsed, img is not None
    except Exception:
        return name, 0, False


def profile_file(svg_path: Path, size: int = 400) -> str:
    """Profile a single SVG render to find bottlenecks."""
    renderer = get_renderer(USE_RUST_RENDERER, background=(0, 0, 0, 0), antialias=4)

    profiler = cProfile.Profile()
    profiler.enable()
    renderer.render_file(str(svg_path), size, size)
    profiler.disable()

    stream = sysio.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    return stream.getvalue()


def find_slow_files(collection: Collection, threshold_ms: float, timeout: float) -> List[Tuple[str, float]]:
    """Find files taking longer than threshold_ms to render."""
    svg_files = sorted(collection.svg_dir.glob("**/*.svg"))

    slow_files = []
    renderer = get_renderer(USE_RUST_RENDERER, background=(0, 0, 0, 0), antialias=4)

    print(f"Scanning {len(svg_files)} files in {collection.name}...")

    for i, svg_path in enumerate(svg_files):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(svg_files)}...")

        start = time.perf_counter()
        try:
            img = renderer.render_file(str(svg_path), collection.size, collection.size)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if elapsed_ms > threshold_ms:
                slow_files.append((svg_path.name, elapsed_ms))
                print(f"  SLOW: {svg_path.name}: {elapsed_ms:.0f}ms")

            if elapsed_ms > timeout * 1000:
                print(f"  TIMEOUT: {svg_path.name} exceeded {timeout}s")

        except Exception as e:
            print(f"  ERROR: {svg_path.name}: {str(e)[:50]}")

    return sorted(slow_files, key=lambda x: -x[1])


def check_rust_extension():
    print("Checking Rust extension...")
    print("=" * 50)

    try:
        import vectorstag_rust
        print("Rust extension loaded successfully")
        funcs = [f for f in dir(vectorstag_rust) if not f.startswith('_')]
        print(f"Available functions: {funcs}")

        test_svg = Path("samples/svg/heart.svg")
        if test_svg.exists():
            print(f"\nBenchmarking with {test_svg.name}...")
            name, elapsed, success = benchmark_single(test_svg, 400)
            if success:
                print(f"Render time: {elapsed:.1f}ms")

        return True
    except ImportError as e:
        print(f"Rust extension NOT available: {e}")
        print("Running in pure Python mode")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark tool for VectorStag",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Collection flags
    parser.add_argument("--emojis", action="store_true", help="Include Noto emojis")
    parser.add_argument("--flags", action="store_true", help="Include Noto flags")
    parser.add_argument("--material", action="store_true", help="Include Material icons")
    parser.add_argument("--fontawesome", action="store_true", help="Include FontAwesome icons")
    parser.add_argument("--lucide", action="store_true", help="Include Lucide icons")
    parser.add_argument("--w3c", action="store_true", help="Include W3C samples")

    # Meta flags
    parser.add_argument("--collections", action="store_true", help="Include all icon collections")
    parser.add_argument("--resvg", action="store_true", help="Include resvg-test-suite")
    parser.add_argument("--all", action="store_true", help="Include everything (default)")

    # Options
    parser.add_argument("-j", "--workers", type=int, default=16,
                        help="Number of workers (default: 16)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-file timeout in seconds (default: 30)")
    parser.add_argument("--resvg-category", type=str,
                        help="Filter resvg tests by category (e.g., 'shapes', 'filters')")
    parser.add_argument("--list-resvg-categories", action="store_true",
                        help="List available resvg test categories and exit")
    parser.add_argument("--find-slow", type=float, metavar="MS",
                        help="Find files taking longer than MS milliseconds")
    parser.add_argument("--compare-resvg", action="store_true",
                        help="Include resvg timing for performance comparison")

    # Single file options
    parser.add_argument("--file", type=str, help="Benchmark/profile a single file")
    parser.add_argument("--size", type=int, default=400, help="Render size for single file")
    parser.add_argument("--profile", action="store_true", help="Profile the file")
    parser.add_argument("--check-rust", action="store_true", help="Check Rust extension status")
    parser.add_argument("--use-rust", action="store_true", help="Use full Rust renderer (resvg)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy Python renderer")

    args = parser.parse_args()

    # Set renderer mode
    global USE_RUST_RENDERER
    if args.use_rust and not args.legacy:
        USE_RUST_RENDERER = True
        print("Using full Rust renderer (resvg)")
    else:
        USE_RUST_RENDERER = False

    # Check Rust extension
    if args.check_rust:
        check_rust_extension()
        return

    # List resvg categories
    if args.list_resvg_categories:
        test_dir = Path("resvg-test-suite/tests")
        if not test_dir.exists():
            print("Error: resvg-test-suite not found")
            return
        categories = get_resvg_categories(test_dir)
        print("Available resvg test categories:")
        for cat in categories:
            tests = find_resvg_tests(test_dir, cat)
            print(f"  {cat}: {len(tests)} tests")
        return

    # Single file mode
    if args.file:
        svg_path = Path(args.file)
        if not svg_path.exists():
            print(f"File not found: {args.file}")
            return

        print(f"Benchmarking: {args.file}")
        print("=" * 70)

        name, elapsed, success = benchmark_single(svg_path, args.size)
        print(f"Render time: {elapsed:.1f}ms")
        print(f"Success: {success}")

        if args.profile:
            print("\nProfile:")
            print(profile_file(svg_path, args.size))
        return

    # Find slow files mode
    if args.find_slow:
        collections = get_collections()
        run_collections = []

        for name in ['emojis', 'flags', 'material', 'fontawesome', 'lucide', 'w3c']:
            if getattr(args, name, False):
                if name in collections:
                    run_collections.append(collections[name])

        if args.collections or not run_collections:
            run_collections = list(collections.values())

        print(f"Finding files slower than {args.find_slow}ms")
        print("=" * 70)

        all_slow = []
        for collection in run_collections:
            if not collection.svg_dir.exists():
                continue
            slow = find_slow_files(collection, args.find_slow, args.timeout)
            for name, ms in slow:
                all_slow.append((collection.name, name, ms))

        print("\n" + "=" * 70)
        print(f"SLOW FILES (>{args.find_slow}ms)")
        print("=" * 70)
        for cat, name, ms in sorted(all_slow, key=lambda x: -x[2]):
            print(f"  {ms:>8.0f}ms  {cat}/{name}")
        print(f"\nTotal: {len(all_slow)} slow files")
        return

    # Determine what to run
    run_collections = []
    run_resvg = False

    collections = get_collections()

    # Check specific collection flags
    specific_selected = False
    for name in ['emojis', 'flags', 'material', 'fontawesome', 'lucide', 'w3c']:
        if getattr(args, name, False):
            specific_selected = True
            if name in collections:
                run_collections.append(collections[name])

    if args.collections:
        run_collections = list(collections.values())
        specific_selected = True

    if args.resvg:
        run_resvg = True
        specific_selected = True

    # Default: run everything if nothing selected
    if args.all or not specific_selected:
        run_collections = list(collections.values())
        run_resvg = True

    print("=" * 105)
    print("VECTORSTAG BENCHMARK")
    print("=" * 105)
    print(f"Workers: {args.workers}")
    print(f"Timeout: {args.timeout}s")
    print(f"Collections: {len(run_collections)}")
    print(f"resvg-test-suite: {'Yes' if run_resvg else 'No'}")
    print(f"Compare resvg timing: {'Yes' if args.compare_resvg else 'No'}")
    print()

    all_stats: List[CategoryStats] = []

    # Run collection benchmarks
    for collection in run_collections:
        if not collection.svg_dir.exists():
            print(f"Skipping {collection.name}: directory not found")
            continue

        print(f"\n--- {collection.name.upper()} ---")
        stats = benchmark_collection(collection, args.workers, args.timeout, args.compare_resvg)
        all_stats.append(stats)

    # Run resvg test suite
    if run_resvg:
        # Force cleanup of any lingering resources from ProcessPoolExecutor
        gc.collect()
        time.sleep(0.5)

        print(f"\n--- RESVG-TEST-SUITE ---")
        resvg_stats = benchmark_resvg_suite(args.workers, args.resvg_category, args.timeout)
        all_stats.extend(resvg_stats)

    # Print summary table
    print_table(all_stats, args.compare_resvg)


if __name__ == "__main__":
    main()
