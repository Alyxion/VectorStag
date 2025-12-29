#!/usr/bin/env python3
"""Comprehensive comparison of VectorStag vs CairoSVG and resvg at 400px with multiprocessing."""

import os
import sys
from pathlib import Path
from PIL import Image
import cairosvg
from resvg_python import svg_to_png
import io
import numpy as np
from collections import defaultdict
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import time

from vectorstag import SVGRenderer


def render_with_cairo(svg_path: Path, width: int = 400, height: int = 400) -> Image.Image:
    """Render SVG using CairoSVG."""
    try:
        png_data = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
        return Image.open(io.BytesIO(png_data)).convert("RGBA")
    except Exception as e:
        return None


def render_with_resvg(svg_path: Path, width: int = 400, height: int = 400) -> Image.Image:
    """Render SVG using resvg."""
    try:
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        png_data = bytes(svg_to_png(svg_content))
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        # Resize to target dimensions
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        return None


def render_with_vectorstag(svg_path: Path, width: int = 400, height: int = 400) -> Image.Image:
    """Render SVG using VectorStag."""
    try:
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
        return renderer.render_file(str(svg_path), width, height)
    except Exception as e:
        return None


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity between two images (0-1, higher is better)."""
    if img1 is None or img2 is None:
        return 0.0

    size = (max(img1.width, img2.width), max(img1.height, img2.height))
    img1 = img1.resize(size, Image.Resampling.LANCZOS)
    img2 = img2.resize(size, Image.Resampling.LANCZOS)

    white_bg = Image.new("RGBA", size, (255, 255, 255, 255))
    img1_comp = Image.alpha_composite(white_bg, img1)
    img2_comp = Image.alpha_composite(white_bg, img2)

    arr1 = np.array(img1_comp, dtype=np.float32)[:, :, :3] / 255.0
    arr2 = np.array(img2_comp, dtype=np.float32)[:, :, :3] / 255.0

    mse = np.mean((arr1 - arr2) ** 2)
    similarity = 1.0 - min(1.0, mse * 4)
    return max(0.0, similarity)


def create_comparison_image(cairo_img, resvg_img, vs_img, name: str) -> Image.Image:
    """Create a 4-panel comparison image: Cairo | resvg | VectorStag | Diff."""
    width = 400
    height = 400

    # Composite all on white background
    white = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    cairo_comp = Image.alpha_composite(white.copy(), cairo_img) if cairo_img else white.copy()
    resvg_comp = Image.alpha_composite(white.copy(), resvg_img) if resvg_img else white.copy()
    vs_comp = Image.alpha_composite(white.copy(), vs_img) if vs_img else white.copy()

    # Create comparison canvas
    comp = Image.new("RGBA", (width * 4 + 30, height + 30), (240, 240, 240, 255))

    # Paste images
    comp.paste(cairo_comp, (0, 25))
    comp.paste(resvg_comp, (width + 10, 25))
    comp.paste(vs_comp, (width * 2 + 20, 25))

    # Compute diff (VectorStag vs Cairo)
    if cairo_img and vs_img:
        arr1 = np.array(cairo_comp, dtype=np.float32)
        arr2 = np.array(vs_comp, dtype=np.float32)
        diff = np.abs(arr1 - arr2) * 3  # Amplify
        diff_img = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
        comp.paste(diff_img, (width * 3 + 30, 25))

    return comp


def process_single_svg(args):
    """Process a single SVG file - used by multiprocessing pool."""
    svg_path, output_dir, size, save_all = args
    name = svg_path.stem

    try:
        cairo_img = render_with_cairo(svg_path, size, size)
        resvg_img = render_with_resvg(svg_path, size, size)
        vs_img = render_with_vectorstag(svg_path, size, size)

        if vs_img is None:
            return {"name": name, "error": "VectorStag render failed"}

        # Compute similarities
        sim_cairo = compute_similarity(cairo_img, vs_img) if cairo_img else 0.0
        sim_resvg = compute_similarity(resvg_img, vs_img) if resvg_img else 0.0

        # Use best similarity (some files have CairoSVG bugs)
        sim = max(sim_cairo, sim_resvg)

        # Save comparison images for files <99%
        if save_all or sim < 0.99:
            comp = create_comparison_image(cairo_img, resvg_img, vs_img, name)
            comp.save(output_dir / f"{name}_comparison.png")

        return {
            "name": name,
            "sim": sim,
            "sim_cairo": sim_cairo,
            "sim_resvg": sim_resvg
        }
    except Exception as e:
        return {"name": name, "error": str(e)[:50]}


def test_directory(svg_dir: Path, output_dir: Path, limit: int = None,
                   save_all: bool = False, size: int = 400, num_workers: int = None):
    """Test all SVGs in a directory using multiprocessing."""
    svg_files = sorted(svg_dir.glob("*.svg"))
    if limit:
        svg_files = svg_files[:limit]

    if not svg_files:
        print(f"No SVG files found in {svg_dir}")
        return [], [], {}

    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Testing {len(svg_files)} SVG files from {svg_dir}")
    print(f"Output: {output_dir}")
    print(f"Resolution: {size}x{size}")
    print(f"Workers: {num_workers}\n")

    # Prepare arguments for each task
    tasks = [(svg_path, output_dir, size, save_all) for svg_path in svg_files]

    results = []
    errors = []
    buckets = defaultdict(list)

    start_time = time.time()

    # Process in parallel
    with Pool(num_workers) as pool:
        completed = 0
        for result in pool.imap_unordered(process_single_svg, tasks):
            completed += 1

            if "error" in result:
                errors.append((result["name"], result["error"]))
            else:
                name = result["name"]
                sim = result["sim"]
                sim_cairo = result["sim_cairo"]
                sim_resvg = result["sim_resvg"]

                results.append((name, sim, sim_cairo, sim_resvg))

                # Bucket by similarity
                if sim >= 0.99:
                    buckets["99-100%"].append(name)
                elif sim >= 0.95:
                    buckets["95-99%"].append(name)
                elif sim >= 0.90:
                    buckets["90-95%"].append(name)
                elif sim >= 0.80:
                    buckets["80-90%"].append(name)
                else:
                    buckets["<80%"].append(name)

            # Progress every 100 files or at end
            if completed % 100 == 0 or completed == len(svg_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                avg = np.mean([s for _, s, _, _ in results]) if results else 0
                print(f"  Processed {completed}/{len(svg_files)} - Avg: {avg:.1%} - {rate:.1f} files/sec")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({len(svg_files)/elapsed:.1f} files/sec)")

    return results, errors, buckets


def print_summary(results, errors, buckets, title=""):
    """Print test summary."""
    print("\n" + "=" * 70)
    print(f"SUMMARY{' - ' + title if title else ''}")
    print("=" * 70)

    if results:
        avg_sim = np.mean([s for _, s, _, _ in results])
        print(f"\nTotal tested: {len(results)}")
        print(f"Average similarity: {avg_sim:.1%}")
        print(f"\nDistribution:")
        for bucket in ["99-100%", "95-99%", "90-95%", "80-90%", "<80%"]:
            count = len(buckets[bucket])
            pct = count / len(results) * 100 if results else 0
            print(f"  {bucket}: {count} ({pct:.1f}%)")

        # Show worst performers
        if buckets["<80%"]:
            print(f"\nWorst performers (<80%):")
            worst = sorted([(n, s) for n, s, _, _ in results if s < 0.80], key=lambda x: x[1])[:15]
            for name, sim in worst:
                print(f"  {name}: {sim:.1%}")

        # Show files needing improvement (80-99%)
        needs_work = [(n, s, sc, sr) for n, s, sc, sr in results if 0.80 <= s < 0.99]
        if needs_work and len(needs_work) <= 30:
            print(f"\nFiles needing improvement (80-99%): {len(needs_work)}")
            for name, sim, sim_cairo, sim_resvg in sorted(needs_work, key=lambda x: x[1]):
                print(f"  {name}: {sim:.1%} (cairo={sim_cairo:.1%}, resvg={sim_resvg:.1%})")

    if errors:
        print(f"\nErrors: {len(errors)}")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare VectorStag vs CairoSVG and resvg")
    parser.add_argument("--flags", action="store_true", help="Test flags")
    parser.add_argument("--emojis", action="store_true", help="Test emojis")
    parser.add_argument("--svg", action="store_true", help="Test W3C samples")
    parser.add_argument("--all", action="store_true", help="Test all")
    parser.add_argument("--limit", type=int, help="Limit number of files")
    parser.add_argument("--size", type=int, default=400, help="Render size (default: 400)")
    parser.add_argument("--save-all", action="store_true", help="Save all comparisons")
    parser.add_argument("--workers", "-j", type=int, default=None,
                        help=f"Number of worker processes (default: {min(cpu_count(), 16)})")
    args = parser.parse_args()

    noto_dir = Path("SciStagEssentialData/images/noto")
    samples_dir = Path("samples/svg")
    output_base = Path("samples/comparison_400")

    all_results = []

    if args.svg or args.all:
        print("\n" + "=" * 70)
        print("TESTING W3C SVG SAMPLES")
        print("=" * 70)
        results, errors, buckets = test_directory(
            samples_dir, output_base / "svg",
            limit=args.limit, save_all=args.save_all, size=args.size,
            num_workers=args.workers
        )
        print_summary(results, errors, buckets, "W3C SVG")
        all_results.extend(results)

    if args.flags or args.all:
        print("\n" + "=" * 70)
        print("TESTING FLAGS")
        print("=" * 70)
        results, errors, buckets = test_directory(
            noto_dir / "flags" / "svg", output_base / "flags",
            limit=args.limit, save_all=args.save_all, size=args.size,
            num_workers=args.workers
        )
        print_summary(results, errors, buckets, "FLAGS")
        all_results.extend(results)

    if args.emojis or args.all:
        print("\n" + "=" * 70)
        print("TESTING EMOJIS")
        print("=" * 70)
        results, errors, buckets = test_directory(
            noto_dir / "emojis" / "svg", output_base / "emojis",
            limit=args.limit, save_all=args.save_all, size=args.size,
            num_workers=args.workers
        )
        print_summary(results, errors, buckets, "EMOJIS")
        all_results.extend(results)

    # Overall summary
    if all_results and len(all_results) > 50:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        avg = np.mean([s for _, s, _, _ in all_results])
        above_99 = sum(1 for _, s, _, _ in all_results if s >= 0.99)
        above_95 = sum(1 for _, s, _, _ in all_results if s >= 0.95)
        print(f"Total files: {len(all_results)}")
        print(f"Average similarity: {avg:.1%}")
        print(f"Files >= 99%: {above_99} ({above_99/len(all_results)*100:.1f}%)")
        print(f"Files >= 95%: {above_95} ({above_95/len(all_results)*100:.1f}%)")

    if not (args.flags or args.emojis or args.svg or args.all):
        print("Usage: python compare_all.py [--flags] [--emojis] [--svg] [--all] [--limit N] [--size N] [-j WORKERS]")
        print(f"\nDefault workers: {min(cpu_count(), 16)} (detected {cpu_count()} CPUs)")


if __name__ == "__main__":
    main()
