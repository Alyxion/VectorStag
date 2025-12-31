#!/usr/bin/env python3
"""
Benchmark VectorStag against resvg-test-suite.

Memory-efficient multiprocessing with worker restart and timeout support.
"""

import sys
import os
import gc
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import argparse
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from vectorstag import SVGRenderer

# Constants for memory management
WORKER_TIMEOUT = 30  # seconds per test
BATCH_SIZE = 100  # restart workers after this many tests to free memory
MAX_MEMORY_MB = 100  # warn if process exceeds this


def get_memory_mb():
    """Get current process memory in MB."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except:
        pass
    return 0


@dataclass
class TestResult:
    name: str
    category: str
    subcategory: str
    similarity: Optional[float]
    error: Optional[str] = None


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute structural similarity between two images using premultiplied alpha."""
    # Resize to same dimensions if needed
    if img1.size != img2.size:
        # Use the reference size
        img1 = img1.resize(img2.size, Image.LANCZOS)

    arr1 = np.array(img1.convert('RGBA'), dtype=np.float32)
    arr2 = np.array(img2.convert('RGBA'), dtype=np.float32)

    # Check if reference (arr2) has opaque white background
    # If so, composite our image over white for fair comparison
    ref_alpha_mean = arr2[:, :, 3].mean()
    if ref_alpha_mean > 250:  # Reference is mostly opaque
        # Check if reference background is white
        corners = [(0,0), (0,-1), (-1,0), (-1,-1)]
        white_corners = sum(1 for y,x in corners
                          if arr2[y, x, 0] > 240 and arr2[y, x, 1] > 240 and arr2[y, x, 2] > 240)
        if white_corners >= 3:
            # Composite our transparent image over white background
            a1_ratio = arr1[:, :, 3:4] / 255.0
            white_bg = np.ones_like(arr1[:, :, :3]) * 255
            arr1[:, :, :3] = arr1[:, :, :3] * a1_ratio + white_bg * (1 - a1_ratio)
            arr1[:, :, 3] = 255

    # Use premultiplied alpha comparison
    # This correctly handles transparent pixels (RGB doesn't matter when A=0)
    a1 = arr1[:, :, 3:4] / 255.0
    a2 = arr2[:, :, 3:4] / 255.0

    # Premultiply RGB by alpha
    rgb1 = arr1[:, :, :3] * a1
    rgb2 = arr2[:, :, :3] * a2

    # Compare premultiplied RGB
    rgb_diff = np.abs(rgb1 - rgb2).mean()

    # Also compare alpha channel
    alpha_diff = np.abs(arr1[:, :, 3] - arr2[:, :, 3]).mean()

    # Combine: both RGB and alpha contribute equally
    combined_diff = (rgb_diff + alpha_diff) / 2

    # Convert to similarity percentage (0-255 scale)
    similarity = 100 * (1 - combined_diff / 255)
    return max(0, similarity)


def render_test(svg_path: Path, ref_path: Path, antialias: int = 4) -> TestResult:
    """Render a single test and compare to reference.

    Args:
        svg_path: Path to SVG file
        ref_path: Path to reference PNG
        antialias: Anti-aliasing factor (default 2, use 4 for higher quality)
    """
    parts = svg_path.relative_to(Path("resvg-test-suite/tests")).parts
    category = parts[0]
    subcategory = parts[1] if len(parts) > 2 else ""
    name = svg_path.stem

    try:
        # Load reference PNG
        ref_img = Image.open(ref_path).convert('RGBA')
        ref_size = ref_img.size

        # Render with VectorStag at reference size
        # Use antialias=2 by default to save memory (still good quality)
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=antialias)
        vs_img = renderer.render_file(str(svg_path), ref_size[0], ref_size[1])

        if vs_img is None:
            del renderer, ref_img
            gc.collect()
            return TestResult(name, category, subcategory, None, "Render returned None")

        vs_img = vs_img.convert('RGBA')

        # Compute similarity
        similarity = compute_similarity(vs_img, ref_img)

        # Explicit cleanup to help memory management
        del renderer, vs_img, ref_img
        gc.collect()

        return TestResult(name, category, subcategory, similarity)

    except Exception as e:
        gc.collect()
        return TestResult(name, category, subcategory, None, str(e)[:100])


def render_test_wrapper(args):
    """Wrapper for multiprocessing."""
    if len(args) == 3:
        svg_path, ref_path, antialias = args
    else:
        svg_path, ref_path = args
        antialias = 4
    return render_test(svg_path, ref_path, antialias)


def find_all_tests(test_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all SVG tests and their reference PNGs."""
    tests = []
    for svg_path in sorted(test_dir.rglob("*.svg")):
        ref_path = svg_path.with_suffix('.png')
        if ref_path.exists():
            tests.append((svg_path, ref_path))
    return tests


def run_benchmark(test_dir: Path, workers: int = 1, category_filter: str = None,
                 save_failures: bool = False, limit: int = None,
                 antialias: int = 2) -> Dict:
    """Run benchmark on all tests.

    Args:
        test_dir: Directory containing test SVGs
        workers: Number of parallel workers (default 1)
        category_filter: Filter tests by category string
        save_failures: Save comparison images for failures
        limit: Maximum number of tests to run
        antialias: Anti-aliasing factor (2 or 4)
    """
    tests = find_all_tests(test_dir)

    if category_filter:
        tests = [(s, r) for s, r in tests if category_filter in str(s)]

    if limit:
        tests = tests[:limit]

    print(f"Found {len(tests)} tests")
    print(f"Workers: {workers}")
    print(f"Antialias: {antialias}x")
    print(f"Memory limit: {MAX_MEMORY_MB}MB per worker")
    print()

    results: List[TestResult] = []

    if workers == 1:
        for i, (svg_path, ref_path) in enumerate(tests):
            result = render_test(svg_path, ref_path, antialias)
            results.append(result)
            if (i + 1) % 50 == 0:
                valid = [r for r in results if r.similarity is not None]
                avg = sum(r.similarity for r in valid) / len(valid) if valid else 0
                mem = get_memory_mb()
                print(f"  {i+1}/{len(tests)} - Avg: {avg:.1f}% - Mem: {mem:.0f}MB")
    else:
        # Process in batches to allow worker restart and memory cleanup
        num_batches = (len(tests) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_start = 0

        for batch_idx in range(num_batches):
            batch_end = min(batch_start + BATCH_SIZE, len(tests))
            batch_tests = tests[batch_start:batch_end]

            # Create new executor for each batch to restart workers
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(render_test_wrapper, (t[0], t[1], antialias)): t
                          for t in batch_tests}

                for future in as_completed(futures, timeout=WORKER_TIMEOUT * len(batch_tests)):
                    try:
                        result = future.result(timeout=WORKER_TIMEOUT)
                        results.append(result)
                    except TimeoutError:
                        svg_path, ref_path = futures[future][:2]
                        parts = svg_path.relative_to(Path("resvg-test-suite/tests")).parts
                        result = TestResult(
                            svg_path.stem, parts[0],
                            parts[1] if len(parts) > 2 else "",
                            None, "Timeout"
                        )
                        results.append(result)
                    except Exception as e:
                        svg_path, ref_path = futures[future][:2]
                        parts = svg_path.relative_to(Path("resvg-test-suite/tests")).parts
                        result = TestResult(
                            svg_path.stem, parts[0],
                            parts[1] if len(parts) > 2 else "",
                            None, str(e)[:100]
                        )
                        results.append(result)

            # Progress update after each batch
            valid = [r for r in results if r.similarity is not None]
            avg = sum(r.similarity for r in valid) / len(valid) if valid else 0
            print(f"  {len(results)}/{len(tests)} - Avg: {avg:.1f}%")

            batch_start = batch_end
            gc.collect()  # Help memory cleanup between batches

    # Analyze results
    analyze_results(results, save_failures)

    return results


def analyze_results(results: List[TestResult], save_failures: bool = False):
    """Analyze and print results."""
    # Overall stats
    valid = [r for r in results if r.similarity is not None]
    errors = [r for r in results if r.error is not None]

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    if valid:
        avg = sum(r.similarity for r in valid) / len(valid)
        print(f"Total tests: {len(results)}")
        print(f"Successful: {len(valid)}")
        print(f"Errors: {len(errors)}")
        print(f"Average similarity: {avg:.2f}%")

        # Distribution
        ranges = [
            ("99-100%", 99, 100),
            ("95-99%", 95, 99),
            ("90-95%", 90, 95),
            ("80-90%", 80, 90),
            ("<80%", 0, 80),
        ]

        print("\nDistribution:")
        for label, low, high in ranges:
            count = len([r for r in valid if low <= r.similarity < high or (high == 100 and r.similarity == 100)])
            pct = 100 * count / len(valid)
            print(f"  {label}: {count} ({pct:.1f}%)")

    # Per-category breakdown
    print("\n" + "=" * 70)
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 70)

    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    for cat in sorted(categories.keys()):
        cat_results = categories[cat]
        cat_valid = [r for r in cat_results if r.similarity is not None]
        cat_errors = [r for r in cat_results if r.error is not None]

        if cat_valid:
            cat_avg = sum(r.similarity for r in cat_valid) / len(cat_valid)
            at_99 = len([r for r in cat_valid if r.similarity >= 99])
            print(f"\n{cat}:")
            print(f"  Tests: {len(cat_results)}, Valid: {len(cat_valid)}, Errors: {len(cat_errors)}")
            print(f"  Average: {cat_avg:.2f}%, 99%+: {at_99} ({100*at_99/len(cat_valid):.1f}%)")

            # Subcategory breakdown
            subcats = {}
            for r in cat_valid:
                if r.subcategory not in subcats:
                    subcats[r.subcategory] = []
                subcats[r.subcategory].append(r)

            if len(subcats) > 1:
                for subcat in sorted(subcats.keys()):
                    if subcat:
                        sub_results = subcats[subcat]
                        sub_avg = sum(r.similarity for r in sub_results) / len(sub_results)
                        print(f"    {subcat}: {sub_avg:.1f}% ({len(sub_results)} tests)")

    # Worst performing tests
    print("\n" + "=" * 70)
    print("WORST PERFORMING TESTS (< 90%)")
    print("=" * 70)

    worst = sorted([r for r in valid if r.similarity < 90], key=lambda x: x.similarity)[:30]
    for r in worst:
        print(f"  {r.similarity:.1f}%: {r.category}/{r.subcategory}/{r.name}")

    # Errors
    if errors:
        print("\n" + "=" * 70)
        print(f"ERRORS ({len(errors)})")
        print("=" * 70)

        error_types = {}
        for r in errors:
            err = r.error[:50] if r.error else "Unknown"
            if err not in error_types:
                error_types[err] = []
            error_types[err].append(r)

        for err, err_results in sorted(error_types.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  [{len(err_results)}] {err}")


def save_comparison(svg_path: Path, ref_path: Path, output_dir: Path):
    """Save a comparison image for debugging."""
    try:
        ref_img = Image.open(ref_path).convert('RGBA')
        ref_size = ref_img.size

        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
        vs_img = renderer.render_file(str(svg_path), ref_size[0], ref_size[1])

        if vs_img is None:
            return

        vs_img = vs_img.convert('RGBA')

        # Create comparison grid
        width = ref_size[0] * 3
        height = ref_size[1]
        grid = Image.new('RGBA', (width, height), (128, 128, 128, 255))

        grid.paste(vs_img, (0, 0))
        grid.paste(ref_img, (ref_size[0], 0))

        # Diff
        arr1 = np.array(vs_img, dtype=np.float32)
        arr2 = np.array(ref_img, dtype=np.float32)
        diff = np.abs(arr1 - arr2)
        diff_visible = (diff.max(axis=2) > 10).astype(np.uint8) * 255

        diff_img = Image.new('RGBA', ref_size, (0, 0, 0, 255))
        diff_arr = np.array(diff_img)
        diff_arr[:, :, 0] = diff_visible  # Red channel shows differences
        diff_arr[:, :, 2] = diff_visible  # Blue = magenta
        diff_arr[:, :, 3] = 255
        diff_img = Image.fromarray(diff_arr)

        grid.paste(diff_img, (ref_size[0] * 2, 0))

        # Save
        rel_path = svg_path.relative_to(Path("resvg-test-suite/tests"))
        out_path = output_dir / rel_path.with_suffix('.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(out_path)

    except Exception as e:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark against resvg-test-suite")
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Number of workers (default 1, use 4-8 for speed)")
    parser.add_argument("--category", type=str, help="Filter by category (e.g., 'shapes', 'filters')")
    parser.add_argument("--limit", type=int, help="Limit number of tests")
    parser.add_argument("--save-failures", action="store_true",
                        help="Save comparison images for failures")
    parser.add_argument("--antialias", type=int, default=4, choices=[1, 2, 4],
                        help="Anti-aliasing factor (1=none, 2=low, 4=high quality). Default 4x - DO NOT CHANGE")

    args = parser.parse_args()

    test_dir = Path("resvg-test-suite/tests")
    if not test_dir.exists():
        print("Error: resvg-test-suite not found. Clone it first.")
        sys.exit(1)

    # Recommend worker count based on task
    if args.workers > 8:
        print(f"Note: Using {args.workers} workers may cause memory issues.")
        print(f"      Consider using -j 4 or -j 8 for stability.\n")

    run_benchmark(test_dir, args.workers, args.category, args.save_failures,
                  args.limit, args.antialias)
