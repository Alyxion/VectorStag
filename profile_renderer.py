#!/usr/bin/env python3
"""Profile the VectorStag renderer to identify bottlenecks."""

import time
import cProfile
import pstats
import io
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from vectorstag import SVGRenderer
from vectorstag.svg_parser import SVGParser

def profile_single_file(svg_path: str, iterations: int = 10):
    """Profile rendering a single file."""
    renderer = SVGRenderer(antialias=4)
    parser = SVGParser()

    with open(svg_path, 'r') as f:
        svg_content = f.read()

    # Time parsing
    parse_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        doc = parser.parse(svg_content)
        parse_times.append(time.perf_counter() - start)

    # Time rendering
    render_times = []
    for _ in range(iterations):
        doc = parser.parse(svg_content)
        start = time.perf_counter()
        renderer.render_document(doc)
        render_times.append(time.perf_counter() - start)

    avg_parse = sum(parse_times) / len(parse_times) * 1000
    avg_render = sum(render_times) / len(render_times) * 1000

    print(f"File: {svg_path}")
    print(f"  Parse time:  {avg_parse:.2f}ms (avg of {iterations})")
    print(f"  Render time: {avg_render:.2f}ms (avg of {iterations})")
    print(f"  Total:       {avg_parse + avg_render:.2f}ms")
    print()

    return avg_parse, avg_render


def profile_with_cprofile(svg_path: str, iterations: int = 5):
    """Run cProfile on rendering."""
    renderer = SVGRenderer(antialias=4)

    with open(svg_path, 'r') as f:
        svg_content = f.read()

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(iterations):
        renderer.render(svg_content)

    pr.disable()

    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())


def profile_batch(svg_dir: str, max_files: int = 50):
    """Profile a batch of files."""
    renderer = SVGRenderer(antialias=4)
    parser = SVGParser()

    svg_files = list(Path(svg_dir).rglob("*.svg"))[:max_files]

    total_parse = 0
    total_render = 0

    for svg_path in svg_files:
        try:
            with open(svg_path, 'r') as f:
                svg_content = f.read()

            start = time.perf_counter()
            doc = parser.parse(svg_content)
            parse_time = time.perf_counter() - start

            start = time.perf_counter()
            renderer.render_document(doc)
            render_time = time.perf_counter() - start

            total_parse += parse_time
            total_render += render_time

        except Exception as e:
            print(f"Error with {svg_path}: {e}")

    print(f"\nBatch Profile ({len(svg_files)} files):")
    print(f"  Total parse time:  {total_parse*1000:.1f}ms ({total_parse/len(svg_files)*1000:.2f}ms avg)")
    print(f"  Total render time: {total_render*1000:.1f}ms ({total_render/len(svg_files)*1000:.2f}ms avg)")
    print(f"  Parse %: {total_parse/(total_parse+total_render)*100:.1f}%")
    print(f"  Render %: {total_render/(total_parse+total_render)*100:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", help="Profile single file")
    parser.add_argument("--dir", "-d", help="Profile directory of files")
    parser.add_argument("--cprofile", "-c", action="store_true", help="Use cProfile")
    parser.add_argument("--iterations", "-n", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=50)
    args = parser.parse_args()

    if args.file:
        if args.cprofile:
            profile_with_cprofile(args.file, args.iterations)
        else:
            profile_single_file(args.file, args.iterations)
    elif args.dir:
        profile_batch(args.dir, args.max_files)
    else:
        # Default: profile FontAwesome icons
        print("Profiling FontAwesome icons...")
        profile_batch("SciStagEssentialData/svg-collections/FontAwesome", 100)
