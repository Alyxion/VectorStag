#!/usr/bin/env python3
"""Compare VectorStag SVG renderer against CairoSVG for Noto SVGs."""

import os
import sys
from pathlib import Path
from PIL import Image
import cairosvg
import io
import numpy as np
from collections import defaultdict

from vectorstag import SVGRenderer


def render_with_cairo(svg_path: Path, width: int = 128, height: int = 128) -> Image.Image:
    """Render SVG using CairoSVG."""
    try:
        png_data = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
        return Image.open(io.BytesIO(png_data)).convert("RGBA")
    except Exception as e:
        return None


def render_with_vectorstag(svg_path: Path, width: int = 128, height: int = 128) -> Image.Image:
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


def test_directory(svg_dir: Path, limit: int = None, save_failures: bool = True):
    """Test all SVGs in a directory."""
    svg_files = sorted(svg_dir.glob("*.svg"))  # Only direct children, not subdirs
    if limit:
        svg_files = svg_files[:limit]

    if not svg_files:
        print(f"No SVG files found in {svg_dir}")
        return

    print(f"Testing {len(svg_files)} SVG files from {svg_dir}\n")

    results = []
    errors = []
    buckets = defaultdict(list)  # Group by similarity range

    for i, svg_path in enumerate(svg_files):
        name = svg_path.stem
        try:
            cairo_img = render_with_cairo(svg_path)
            vs_img = render_with_vectorstag(svg_path)

            if cairo_img is None or vs_img is None:
                errors.append((name, "Render failed"))
                continue

            sim = compute_similarity(cairo_img, vs_img)
            results.append((name, sim))

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

            # Progress
            if (i + 1) % 50 == 0:
                avg = np.mean([s for _, s in results])
                print(f"  Processed {i+1}/{len(svg_files)} - Current avg: {avg:.1%}")

        except Exception as e:
            errors.append((name, str(e)[:50]))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:
        avg_sim = np.mean([s for _, s in results])
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
            worst = sorted([(n, s) for n, s in results if s < 0.80], key=lambda x: x[1])[:10]
            for name, sim in worst:
                print(f"  {name}: {sim:.1%}")

        # Show files between 80-99%
        needs_work = [(n, s) for n, s in results if 0.80 <= s < 0.99]
        if needs_work:
            print(f"\nFiles needing improvement (80-99%): {len(needs_work)}")
            if len(needs_work) <= 20:
                for name, sim in sorted(needs_work, key=lambda x: x[1]):
                    print(f"  {name}: {sim:.1%}")

    if errors:
        print(f"\nErrors: {len(errors)}")
        for name, err in errors[:5]:
            print(f"  {name}: {err}")

    return results, errors, buckets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flags", action="store_true", help="Test flags")
    parser.add_argument("--emojis", action="store_true", help="Test emojis")
    parser.add_argument("--limit", type=int, help="Limit number of files")
    parser.add_argument("--all", action="store_true", help="Test all")
    args = parser.parse_args()

    base_dir = Path("SciStagEssentialData/images/noto")

    if args.flags or args.all:
        print("\n" + "=" * 60)
        print("TESTING FLAGS")
        print("=" * 60)
        test_directory(base_dir / "flags" / "svg", limit=args.limit)

    if args.emojis or args.all:
        print("\n" + "=" * 60)
        print("TESTING EMOJIS")
        print("=" * 60)
        test_directory(base_dir / "emojis" / "svg", limit=args.limit)

    if not (args.flags or args.emojis or args.all):
        print("Usage: python compare_noto.py [--flags] [--emojis] [--all] [--limit N]")


if __name__ == "__main__":
    main()
