#!/usr/bin/env python3
"""
Benchmark VectorStag against resvg-test-suite.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import argparse
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from vectorstag import SVGRenderer


@dataclass
class TestResult:
    name: str
    category: str
    subcategory: str
    similarity: Optional[float]
    error: Optional[str] = None


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute structural similarity between two images."""
    # Resize to same dimensions if needed
    if img1.size != img2.size:
        # Use the reference size
        img1 = img1.resize(img2.size, Image.LANCZOS)

    arr1 = np.array(img1.convert('RGBA'), dtype=np.float32)
    arr2 = np.array(img2.convert('RGBA'), dtype=np.float32)

    # For transparent pixels (alpha=0), RGB values don't matter
    # Only compare RGB where at least one image has non-zero alpha
    alpha1 = arr1[:, :, 3]
    alpha2 = arr2[:, :, 3]

    # Create mask for pixels where we should compare
    visible = (alpha1 > 0) | (alpha2 > 0)

    if not visible.any():
        return 100.0  # Both images fully transparent

    # Compare RGB only for visible pixels, plus compare alpha for all
    rgb_diff = np.abs(arr1[:, :, :3] - arr2[:, :, :3])
    rgb_diff_masked = np.where(visible[:, :, np.newaxis], rgb_diff, 0)
    alpha_diff = np.abs(alpha1 - alpha2)

    # Weighted average: RGB diff for visible + alpha diff for all
    total_pixels = arr1.shape[0] * arr1.shape[1]
    visible_count = visible.sum()

    if visible_count > 0:
        rgb_mae = rgb_diff_masked.sum() / (visible_count * 3)
    else:
        rgb_mae = 0

    alpha_mae = alpha_diff.mean()

    # Combine: RGB contributes more but alpha also matters
    combined_mae = (rgb_mae * 0.75 + alpha_mae * 0.25)

    # Convert to similarity percentage (0-255 scale)
    similarity = 100 * (1 - combined_mae / 255)
    return max(0, similarity)


def render_test(svg_path: Path, ref_path: Path) -> TestResult:
    """Render a single test and compare to reference."""
    parts = svg_path.relative_to(Path("resvg-test-suite/tests")).parts
    category = parts[0]
    subcategory = parts[1] if len(parts) > 2 else ""
    name = svg_path.stem

    try:
        # Load reference PNG
        ref_img = Image.open(ref_path).convert('RGBA')
        ref_size = ref_img.size

        # Render with VectorStag at reference size
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
        vs_img = renderer.render_file(str(svg_path), ref_size[0], ref_size[1])

        if vs_img is None:
            return TestResult(name, category, subcategory, None, "Render returned None")

        vs_img = vs_img.convert('RGBA')

        # Compute similarity
        similarity = compute_similarity(vs_img, ref_img)

        return TestResult(name, category, subcategory, similarity)

    except Exception as e:
        return TestResult(name, category, subcategory, None, str(e)[:100])


def render_test_wrapper(args):
    """Wrapper for multiprocessing."""
    svg_path, ref_path = args
    return render_test(svg_path, ref_path)


def find_all_tests(test_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all SVG tests and their reference PNGs."""
    tests = []
    for svg_path in sorted(test_dir.rglob("*.svg")):
        ref_path = svg_path.with_suffix('.png')
        if ref_path.exists():
            tests.append((svg_path, ref_path))
    return tests


def run_benchmark(test_dir: Path, workers: int = 1, category_filter: str = None,
                 save_failures: bool = False, limit: int = None) -> Dict:
    """Run benchmark on all tests."""
    tests = find_all_tests(test_dir)

    if category_filter:
        tests = [(s, r) for s, r in tests if category_filter in str(s)]

    if limit:
        tests = tests[:limit]

    print(f"Found {len(tests)} tests")
    print(f"Workers: {workers}")
    print()

    results: List[TestResult] = []

    if workers == 1:
        for i, (svg_path, ref_path) in enumerate(tests):
            result = render_test(svg_path, ref_path)
            results.append(result)
            if (i + 1) % 50 == 0:
                valid = [r for r in results if r.similarity is not None]
                avg = sum(r.similarity for r in valid) / len(valid) if valid else 0
                print(f"  {i+1}/{len(tests)} - Avg: {avg:.1f}%")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(render_test_wrapper, t): t for t in tests}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 100 == 0:
                    valid = [r for r in results if r.similarity is not None]
                    avg = sum(r.similarity for r in valid) / len(valid) if valid else 0
                    print(f"  {completed}/{len(tests)} - Avg: {avg:.1f}%")

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
    parser.add_argument("-j", "--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--limit", type=int, help="Limit number of tests")
    parser.add_argument("--save-failures", action="store_true", help="Save comparison images for failures")

    args = parser.parse_args()

    test_dir = Path("resvg-test-suite/tests")
    if not test_dir.exists():
        print("Error: resvg-test-suite not found. Clone it first.")
        sys.exit(1)

    run_benchmark(test_dir, args.workers, args.category, args.save_failures, args.limit)
