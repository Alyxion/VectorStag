#!/usr/bin/env python3
"""
Run all VectorStag benchmarks and report overall performance.

Usage:
    python run_all_benchmarks.py
"""

import subprocess
import sys
import re
from dataclasses import dataclass
from typing import List, Tuple

WORKERS = 8


@dataclass
class BenchmarkResult:
    name: str
    tests: int
    score: float
    at_99_plus: int = 0


def run_resvg_benchmark() -> BenchmarkResult:
    """Run resvg-test-suite benchmark."""
    print("=" * 70)
    print("RUNNING: resvg-test-suite")
    print("=" * 70)

    result = subprocess.run(
        ["python", "benchmark_resvg_tests.py", "-j", str(WORKERS)],
        capture_output=True,
        text=True,
        timeout=900
    )

    output = result.stdout + result.stderr
    print(output)

    # Parse results
    tests = 0
    score = 0.0
    at_99 = 0

    for line in output.split('\n'):
        if 'Total tests:' in line:
            tests = int(re.search(r'Total tests:\s*(\d+)', line).group(1))
        elif 'Average similarity:' in line:
            score = float(re.search(r'Average similarity:\s*([\d.]+)', line).group(1))
        elif '99-100%:' in line:
            match = re.search(r'99-100%:\s*(\d+)', line)
            if match:
                at_99 = int(match.group(1))

    return BenchmarkResult("resvg-test-suite", tests, score, at_99)


def run_svg_compare() -> List[BenchmarkResult]:
    """Run svg_compare.py for all icon collections."""
    print("\n" + "=" * 70)
    print("RUNNING: Icon Collections (svg_compare.py)")
    print("=" * 70)

    result = subprocess.run(
        ["python", "svg_compare.py", "compare", "--all"],
        capture_output=True,
        text=True,
        timeout=600
    )

    output = result.stdout + result.stderr
    print(output)

    results = []
    current_name = None

    for line in output.split('\n'):
        if 'SUMMARY -' in line:
            current_name = line.split('SUMMARY -')[1].strip()
        elif current_name and 'Total:' in line:
            tests = int(re.search(r'Total:\s*(\d+)', line).group(1))
        elif current_name and 'Average:' in line and '%' in line:
            score = float(re.search(r'Average:\s*([\d.]+)%', line).group(1))
            results.append(BenchmarkResult(current_name, tests, score))
            current_name = None

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print overall summary."""
    print("\n" + "=" * 70)
    print("OVERALL BENCHMARK SUMMARY")
    print("=" * 70)

    total_tests = sum(r.tests for r in results)
    weighted_sum = sum(r.tests * r.score for r in results)
    overall_score = weighted_sum / total_tests if total_tests > 0 else 0

    print(f"\n{'Benchmark':<25} {'Tests':>8} {'Score':>10} {'Status':>10}")
    print("-" * 55)

    for r in results:
        status = "✓" if r.score >= 98.0 else "○" if r.score >= 95.0 else "✗"
        print(f"{r.name:<25} {r.tests:>8} {r.score:>9.1f}% {status:>10}")

    print("-" * 55)
    print(f"{'TOTAL':<25} {total_tests:>8} {overall_score:>9.2f}%")
    print()

    # Distribution summary
    at_99 = sum(1 for r in results if r.score >= 99.0)
    at_95 = sum(1 for r in results if 95.0 <= r.score < 99.0)
    below_95 = sum(1 for r in results if r.score < 95.0)

    print("Score Distribution:")
    print(f"  99%+:  {at_99} benchmarks")
    print(f"  95-99%: {at_95} benchmarks")
    print(f"  <95%:  {below_95} benchmarks")
    print()

    return overall_score, total_tests


def main():
    print("VectorStag Benchmark Suite")
    print(f"Workers: {WORKERS}")
    print()

    all_results = []

    # Run resvg-test-suite
    try:
        resvg_result = run_resvg_benchmark()
        all_results.append(resvg_result)
    except Exception as e:
        print(f"ERROR running resvg benchmark: {e}")

    # Run icon collections
    try:
        icon_results = run_svg_compare()
        all_results.extend(icon_results)
    except Exception as e:
        print(f"ERROR running icon benchmarks: {e}")

    # Print summary
    if all_results:
        overall_score, total_tests = print_summary(all_results)

        print("=" * 70)
        print(f"FINAL SCORE: {overall_score:.2f}% across {total_tests:,} tests")
        print("=" * 70)

        # Exit with error if below threshold
        if overall_score < 95.0:
            sys.exit(1)
    else:
        print("ERROR: No benchmark results collected")
        sys.exit(1)


if __name__ == "__main__":
    main()
