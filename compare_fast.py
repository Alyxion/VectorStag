#!/usr/bin/env python3
"""Fast comparison using pre-rendered references."""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import argparse
from collections import defaultdict

from vectorstag import SVGRenderer


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity between two images."""
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


def process_svg(args):
    """Process a single SVG - compare VectorStag with pre-rendered references."""
    svg_path, cairo_ref_dir, resvg_ref_dir, size = args
    name = svg_path.stem

    try:
        # Render with VectorStag
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
        vs_img = renderer.render_file(str(svg_path), size, size)

        if vs_img is None:
            return {"name": name, "error": "VectorStag render failed"}

        # Load pre-rendered references
        cairo_path = cairo_ref_dir / f"{name}.png"
        resvg_path = resvg_ref_dir / f"{name}.png"

        cairo_img = Image.open(cairo_path).convert("RGBA") if cairo_path.exists() else None
        resvg_img = Image.open(resvg_path).convert("RGBA") if resvg_path.exists() else None

        # Compute similarities
        sim_cairo = compute_similarity(cairo_img, vs_img) if cairo_img else 0.0
        sim_resvg = compute_similarity(resvg_img, vs_img) if resvg_img else 0.0
        sim = max(sim_cairo, sim_resvg)

        return {
            "name": name,
            "sim": sim,
            "sim_cairo": sim_cairo,
            "sim_resvg": sim_resvg
        }
    except Exception as e:
        return {"name": name, "error": str(e)[:50]}


def test_directory(svg_dir: Path, ref_dir: Path, limit: int = None,
                   size: int = 400, num_workers: int = None):
    """Test all SVGs against pre-rendered references."""
    svg_files = sorted(svg_dir.glob("*.svg"))
    if limit:
        svg_files = svg_files[:limit]

    if not svg_files:
        print(f"No SVG files found in {svg_dir}")
        return [], [], {}

    cairo_ref_dir = ref_dir / "cairo"
    resvg_ref_dir = ref_dir / "resvg"

    if not cairo_ref_dir.exists():
        print(f"Reference directory not found: {cairo_ref_dir}")
        print("Run: python prerender_references.py --emojis first")
        return [], [], {}

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Comparing {len(svg_files)} SVGs against pre-rendered references")
    print(f"Resolution: {size}x{size}")
    print(f"Workers: {num_workers}\n")

    tasks = [(svg_path, cairo_ref_dir, resvg_ref_dir, size) for svg_path in svg_files]

    results = []
    errors = []
    buckets = defaultdict(list)

    start_time = time.time()

    with Pool(num_workers) as pool:
        completed = 0
        for result in pool.imap_unordered(process_svg, tasks):
            completed += 1

            if "error" in result:
                errors.append((result["name"], result["error"]))
            else:
                name = result["name"]
                sim = result["sim"]
                results.append((name, sim, result["sim_cairo"], result["sim_resvg"]))

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

            if completed % 500 == 0 or completed == len(svg_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                avg = np.mean([s for _, s, _, _ in results]) if results else 0
                print(f"  {completed}/{len(svg_files)} - Avg: {avg:.1%} - {rate:.1f} files/sec")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({len(svg_files)/elapsed:.1f} files/sec)")

    return results, errors, buckets


def print_summary(results, errors, buckets, title=""):
    """Print test summary."""
    print("\n" + "=" * 70)
    print(f"SUMMARY{' - ' + title if title else ''}")
    print("=" * 70)

    if results:
        avg = np.mean([s for _, s, _, _ in results])
        print(f"\nTotal: {len(results)}")
        print(f"Average: {avg:.1%}")
        print(f"\nDistribution:")
        for bucket in ["99-100%", "95-99%", "90-95%", "80-90%", "<80%"]:
            count = len(buckets[bucket])
            pct = count / len(results) * 100 if results else 0
            print(f"  {bucket}: {count} ({pct:.1f}%)")

        if buckets["<80%"]:
            print(f"\nWorst (<80%):")
            worst = sorted([(n, s) for n, s, _, _ in results if s < 0.80], key=lambda x: x[1])[:10]
            for name, sim in worst:
                print(f"  {name}: {sim:.1%}")

    if errors:
        print(f"\nErrors: {len(errors)}")


def main():
    parser = argparse.ArgumentParser(description="Fast comparison using pre-rendered references")
    parser.add_argument("--emojis", action="store_true", help="Test emojis")
    parser.add_argument("--flags", action="store_true", help="Test flags")
    parser.add_argument("--limit", type=int, help="Limit files")
    parser.add_argument("--size", type=int, default=400, help="Render size")
    parser.add_argument("-j", "--workers", type=int, help="Workers")
    args = parser.parse_args()

    noto_dir = Path("SciStagEssentialData/images/noto")
    ref_base = Path("samples/references_400")

    if args.emojis:
        print("=" * 70)
        print("COMPARING EMOJIS")
        print("=" * 70)
        results, errors, buckets = test_directory(
            noto_dir / "emojis" / "svg",
            ref_base / "emojis",
            limit=args.limit,
            size=args.size,
            num_workers=args.workers
        )
        print_summary(results, errors, buckets, "EMOJIS")

    if args.flags:
        print("=" * 70)
        print("COMPARING FLAGS")
        print("=" * 70)
        results, errors, buckets = test_directory(
            noto_dir / "flags" / "svg",
            ref_base / "flags",
            limit=args.limit,
            size=args.size,
            num_workers=args.workers
        )
        print_summary(results, errors, buckets, "FLAGS")

    if not (args.emojis or args.flags):
        print("Usage: python compare_fast.py [--emojis] [--flags] [--limit N] [-j WORKERS]")


if __name__ == "__main__":
    main()
