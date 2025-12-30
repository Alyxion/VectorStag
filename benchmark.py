#!/usr/bin/env python3
"""
VectorStag Performance Benchmark Tool.

Features:
- Benchmark rendering performance across collections
- Profile individual files to identify bottlenecks
- Check Rust extension availability

Usage:
    # Benchmark a collection
    python benchmark.py --emojis -j 16
    python benchmark.py --all --limit 500

    # Profile a single file
    python benchmark.py --file samples/svg/tiger.svg --profile

    # Check Rust extension
    python benchmark.py --check-rust
"""

import argparse
import cProfile
import io as sysio
import pstats
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from vectorstag import SVGRenderer


# =============================================================================
# Configuration - matches svg_compare.py
# =============================================================================

def get_collection_paths() -> dict:
    """Get paths for all SVG collections."""
    noto_dir = Path("SciStagEssentialData/images/noto")

    return {
        "emojis": (noto_dir / "emojis" / "svg", 400),
        "flags": (noto_dir / "flags" / "svg", 400),
        "material": (Path("advanced_svg/material"), 256),
        "fontawesome": (Path("advanced_svg/fontawesome"), 128),
        "lucide": (Path("advanced_svg/lucide"), 128),
        "w3c": (Path("samples/svg"), 400),
    }


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_single(svg_path: Path, size: int = 400) -> Tuple[str, float, bool]:
    """Benchmark rendering a single SVG. Returns (name, time_ms, success)."""
    name = svg_path.stem
    try:
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
        start = time.perf_counter()
        img = renderer.render_file(str(svg_path), size, size)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        return name, elapsed, img is not None
    except Exception:
        return name, 0, False


def benchmark_worker(args: Tuple[Path, int]) -> Tuple[str, float, bool]:
    """Multiprocessing worker wrapper."""
    svg_path, size = args
    return benchmark_single(svg_path, size)


def profile_file(svg_path: Path, size: int = 400) -> str:
    """Profile a single SVG render to find bottlenecks."""
    renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)

    profiler = cProfile.Profile()
    profiler.enable()
    renderer.render_file(str(svg_path), size, size)
    profiler.disable()

    stream = sysio.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    return stream.getvalue()


def benchmark_collection(svg_dir: Path, size: int = 400, limit: int = None,
                         num_workers: int = None, profile_slowest: bool = False):
    """Benchmark all SVGs in a directory."""
    svg_files = sorted(svg_dir.glob("**/*.svg"))
    if limit:
        svg_files = svg_files[:limit]

    if not svg_files:
        print(f"No SVG files found in {svg_dir}")
        return []

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Benchmarking {len(svg_files)} SVGs from {svg_dir}")
    print(f"Resolution: {size}x{size}")
    print(f"Workers: {num_workers}\n")

    tasks = [(svg_path, size) for svg_path in svg_files]

    results = []
    errors = 0
    start_time = time.time()

    with Pool(num_workers) as pool:
        completed = 0
        for name, elapsed_ms, success in pool.imap_unordered(benchmark_worker, tasks):
            completed += 1
            if success:
                results.append((name, elapsed_ms))
            else:
                errors += 1

            if completed % 500 == 0 or completed == len(svg_files):
                total_elapsed = time.time() - start_time
                rate = completed / total_elapsed if total_elapsed > 0 else 0
                print(f"  {completed}/{len(svg_files)} - {rate:.1f} files/sec")

    total_elapsed = time.time() - start_time

    # Print results
    if results:
        times = [t for _, t in results]

        print(f"\n{'=' * 70}")
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Total files: {len(results)}")
        print(f"Total time: {total_elapsed:.1f}s ({len(results)/total_elapsed:.1f} files/sec)")
        print(f"Errors: {errors}")
        print()
        print(f"Render times (ms):")
        print(f"  Min:    {min(times):.1f} ms")
        print(f"  Max:    {max(times):.1f} ms")
        print(f"  Mean:   {np.mean(times):.1f} ms")
        print(f"  Median: {np.median(times):.1f} ms")
        print(f"  P95:    {np.percentile(times, 95):.1f} ms")
        print(f"  P99:    {np.percentile(times, 99):.1f} ms")

        # Time distribution
        print(f"\nTime distribution:")
        brackets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, float('inf'))]
        for low, high in brackets:
            count = sum(1 for _, t in results if low <= t < high)
            if count > 0:
                pct = count / len(results) * 100
                label = f"{low}-{high}ms" if high != float('inf') else f">{low}ms"
                bar = "#" * int(pct / 2)
                print(f"  {label:>12}: {count:5} ({pct:5.1f}%) {bar}")

        # Slowest files
        slowest = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        print(f"\nSlowest files:")
        for name, t in slowest:
            print(f"  {name}: {t:.1f} ms")

        # Profile slowest if requested
        if profile_slowest and slowest:
            slowest_name = slowest[0][0]
            # Find the full path
            for svg_path in svg_files:
                if svg_path.stem == slowest_name:
                    print(f"\n{'=' * 70}")
                    print(f"PROFILING SLOWEST: {slowest_name}")
                    print("=" * 70)
                    print(profile_file(svg_path, size))
                    break

    return results


def check_rust_extension():
    """Check if Rust extension is available and working."""
    print("Checking Rust extension...")
    print("=" * 50)

    try:
        import vectorstag_rust
        print("Rust extension loaded successfully")
        funcs = [f for f in dir(vectorstag_rust) if not f.startswith('_')]
        print(f"Available functions: {funcs}")

        # Quick benchmark comparison
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
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VectorStag Performance Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Collection selection (matches svg_compare.py)
    parser.add_argument("--emojis", action="store_true", help="Benchmark Noto emojis")
    parser.add_argument("--flags", action="store_true", help="Benchmark Noto flags")
    parser.add_argument("--material", action="store_true", help="Benchmark Material icons")
    parser.add_argument("--fontawesome", action="store_true", help="Benchmark FontAwesome icons")
    parser.add_argument("--lucide", action="store_true", help="Benchmark Lucide icons")
    parser.add_argument("--w3c", action="store_true", help="Benchmark W3C samples")
    parser.add_argument("--all", action="store_true", help="Benchmark all collections")

    # Single file
    parser.add_argument("--file", type=str, help="Benchmark/profile a single file")

    # Options
    parser.add_argument("--limit", type=int, help="Limit number of files per collection")
    parser.add_argument("--size", type=int, help="Override render size")
    parser.add_argument("-j", "--workers", type=int, help="Number of workers")
    parser.add_argument("--profile", action="store_true", help="Profile slowest file (or --file)")
    parser.add_argument("--check-rust", action="store_true", help="Check Rust extension status")

    args = parser.parse_args()

    # Check Rust extension
    if args.check_rust:
        check_rust_extension()
        return

    # Single file mode
    if args.file:
        svg_path = Path(args.file)
        if not svg_path.exists():
            print(f"File not found: {args.file}")
            return

        size = args.size or 400
        print(f"Benchmarking: {args.file}")
        print("=" * 70)

        name, elapsed, success = benchmark_single(svg_path, size)
        print(f"Render time: {elapsed:.1f}ms")
        print(f"Success: {success}")

        if args.profile:
            print("\nProfile:")
            print(profile_file(svg_path, size))
        return

    # Collection mode
    collections = get_collection_paths()
    selected = []

    if args.all:
        selected = list(collections.keys())
    else:
        for name in collections:
            if getattr(args, name, False):
                selected.append(name)

    if not selected:
        parser.print_help()
        print("\nExamples:")
        print("  python benchmark.py --emojis -j 16")
        print("  python benchmark.py --all --limit 500")
        print("  python benchmark.py --file samples/svg/tiger.svg --profile")
        print("  python benchmark.py --check-rust")
        return

    for name in selected:
        svg_dir, default_size = collections[name]

        if not svg_dir.exists():
            print(f"Skipping {name}: {svg_dir} not found")
            continue

        print("\n" + "=" * 70)
        print(f"BENCHMARKING: {name.upper()}")
        print("=" * 70)

        benchmark_collection(
            svg_dir,
            size=args.size or default_size,
            limit=args.limit,
            num_workers=args.workers,
            profile_slowest=args.profile
        )


if __name__ == "__main__":
    main()
