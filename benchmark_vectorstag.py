#!/usr/bin/env python3
"""Benchmark VectorStag rendering performance and identify bottlenecks."""

import os
import sys
from pathlib import Path
from PIL import Image
import time
import argparse
import cProfile
import pstats
import io as sysio
from multiprocessing import Pool, cpu_count
import numpy as np

from vectorstag import SVGRenderer


def benchmark_single(svg_path: Path, size: int = 400) -> tuple:
    """Benchmark rendering a single SVG. Returns (name, time_ms, success)."""
    name = svg_path.stem
    try:
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
        start = time.perf_counter()
        img = renderer.render_file(str(svg_path), size, size)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        return name, elapsed, True
    except Exception as e:
        return name, 0, False


def benchmark_single_wrapper(args):
    """Wrapper for multiprocessing."""
    svg_path, size = args
    return benchmark_single(svg_path, size)


def profile_single(svg_path: Path, size: int = 400):
    """Profile a single SVG render to find bottlenecks."""
    renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)

    profiler = cProfile.Profile()
    profiler.enable()

    img = renderer.render_file(str(svg_path), size, size)

    profiler.disable()

    # Get stats
    stream = sysio.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    return stream.getvalue()


def benchmark_directory(svg_dir: Path, size: int = 400, limit: int = None,
                        num_workers: int = None, profile_slow: bool = False):
    """Benchmark all SVGs in a directory."""
    svg_files = sorted(svg_dir.glob("*.svg"))
    if limit:
        svg_files = svg_files[:limit]

    if not svg_files:
        print(f"No SVG files found in {svg_dir}")
        return

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Benchmarking {len(svg_files)} SVGs from {svg_dir}")
    print(f"Resolution: {size}x{size}")
    print(f"Workers: {num_workers}\n")

    tasks = [(svg_path, size) for svg_path in svg_files]

    results = []
    errors = []
    start_time = time.time()

    with Pool(num_workers) as pool:
        completed = 0
        for name, elapsed_ms, success in pool.imap_unordered(benchmark_single_wrapper, tasks):
            completed += 1
            if success:
                results.append((name, elapsed_ms))
            else:
                errors.append(name)

            if completed % 500 == 0 or completed == len(svg_files):
                total_elapsed = time.time() - start_time
                rate = completed / total_elapsed if total_elapsed > 0 else 0
                print(f"  {completed}/{len(svg_files)} - {rate:.1f} files/sec")

    total_elapsed = time.time() - start_time

    # Analyze results
    times = [t for _, t in results]

    print(f"\n{'=' * 70}")
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total files: {len(results)}")
    print(f"Total time: {total_elapsed:.1f}s ({len(results)/total_elapsed:.1f} files/sec)")
    print(f"Errors: {len(errors)}")
    print()
    print(f"Render times (ms):")
    print(f"  Min:    {min(times):.1f} ms")
    print(f"  Max:    {max(times):.1f} ms")
    print(f"  Mean:   {np.mean(times):.1f} ms")
    print(f"  Median: {np.median(times):.1f} ms")
    print(f"  P95:    {np.percentile(times, 95):.1f} ms")
    print(f"  P99:    {np.percentile(times, 99):.1f} ms")

    # Distribution
    print(f"\nTime distribution:")
    brackets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, float('inf'))]
    for low, high in brackets:
        count = sum(1 for _, t in results if low <= t < high)
        pct = count / len(results) * 100
        label = f"{low}-{high}ms" if high != float('inf') else f">{low}ms"
        bar = "#" * int(pct / 2)
        print(f"  {label:>12}: {count:5} ({pct:5.1f}%) {bar}")

    # Show slowest files
    slowest = sorted(results, key=lambda x: x[1], reverse=True)[:20]
    print(f"\nSlowest files:")
    for name, t in slowest:
        print(f"  {name}: {t:.1f} ms")

    # Profile the slowest file
    if profile_slow and slowest:
        slowest_name = slowest[0][0]
        slowest_path = svg_dir / f"{slowest_name}.svg"
        print(f"\n{'=' * 70}")
        print(f"PROFILING SLOWEST: {slowest_name}")
        print("=" * 70)
        profile_output = profile_single(slowest_path, size)
        print(profile_output)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark VectorStag rendering")
    parser.add_argument("--emojis", action="store_true", help="Test emojis")
    parser.add_argument("--flags", action="store_true", help="Test flags")
    parser.add_argument("--file", type=str, help="Test single file")
    parser.add_argument("--limit", type=int, help="Limit number of files")
    parser.add_argument("--size", type=int, default=400, help="Render size (default: 400)")
    parser.add_argument("--workers", "-j", type=int, default=None, help="Number of workers")
    parser.add_argument("--profile", action="store_true", help="Profile slowest file")
    args = parser.parse_args()

    noto_dir = Path("SciStagEssentialData/images/noto")

    if args.file:
        print(f"Profiling: {args.file}")
        print("=" * 70)

        # Time it
        name, elapsed, success = benchmark_single(Path(args.file), args.size)
        print(f"Render time: {elapsed:.1f} ms")

        if args.profile:
            print("\nProfile:")
            print(profile_single(Path(args.file), args.size))
        return

    if args.emojis:
        print("=" * 70)
        print("BENCHMARKING EMOJIS")
        print("=" * 70)
        benchmark_directory(
            noto_dir / "emojis" / "svg",
            size=args.size,
            limit=args.limit,
            num_workers=args.workers,
            profile_slow=args.profile
        )

    if args.flags:
        print("=" * 70)
        print("BENCHMARKING FLAGS")
        print("=" * 70)
        benchmark_directory(
            noto_dir / "flags" / "svg",
            size=args.size,
            limit=args.limit,
            num_workers=args.workers,
            profile_slow=args.profile
        )

    if not (args.emojis or args.flags or args.file):
        print("Usage: python benchmark_vectorstag.py [--emojis] [--flags] [--file FILE] [--limit N] [--profile]")


if __name__ == "__main__":
    main()
