#!/usr/bin/env python3
"""Fast comparison using pre-rendered references."""

from pathlib import Path
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import argparse
from collections import defaultdict

from vectorstag import SVGRenderer
from comparison_utils import compute_similarity, create_comparison_grid


def process_svg(args):
    """Process a single SVG - compare VectorStag with pre-rendered references."""
    svg_path, cairo_ref_dir, resvg_ref_dir, size, save_dir = args
    name = svg_path.stem

    try:
        # Read SVG to get dimensions and check preserveAspectRatio
        import re
        import xml.etree.ElementTree as ET
        with open(svg_path, 'r') as f:
            svg_content = f.read()

        par_match = re.search(r'preserveAspectRatio\s*=\s*["\']([^"\']+)["\']', svg_content)
        should_stretch = par_match and 'none' in par_match.group(1).lower()

        # Parse SVG dimensions
        root = ET.fromstring(svg_content)
        svg_w = root.get('width', '100')
        svg_h = root.get('height', '100')
        # Strip units
        svg_w = float(re.sub(r'[^0-9.]', '', svg_w) or '100')
        svg_h = float(re.sub(r'[^0-9.]', '', svg_h) or '100')

        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)

        if should_stretch:
            # Render directly at target size
            vs_img = renderer.render_file(str(svg_path), size, size)
        else:
            # Calculate optimal render size to match resvg workflow
            # Render at size that preserves aspect ratio, then center
            aspect = svg_w / svg_h if svg_h > 0 else 1
            if aspect >= 1:
                render_w = size
                render_h = int(size / aspect)
            else:
                render_h = size
                render_w = int(size * aspect)

            vs_img = renderer.render_file(str(svg_path), render_w, render_h)

            if vs_img is not None and vs_img.size != (size, size):
                # Center on canvas
                canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
                offset_x = (size - vs_img.width) // 2
                offset_y = (size - vs_img.height) // 2
                canvas.paste(vs_img, (offset_x, offset_y))
                vs_img = canvas

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

        # Save comparison image if requested
        if save_dir is not None:
            grid = create_comparison_grid(vs_img, resvg_img, cairo_img, size)
            grid.save(save_dir / f"{name}_comparison.png")

        return {
            "name": name,
            "sim": sim,
            "sim_cairo": sim_cairo,
            "sim_resvg": sim_resvg
        }
    except Exception as e:
        return {"name": name, "error": str(e)[:50]}


def test_directory(svg_dir: Path, ref_dir: Path, limit: int = None,
                   size: int = 400, num_workers: int = None, save_dir: Path = None):
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

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving comparison images to: {save_dir}")

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Comparing {len(svg_files)} SVGs against pre-rendered references")
    print(f"Resolution: {size}x{size}")
    print(f"Workers: {num_workers}\n")

    tasks = [(svg_path, cairo_ref_dir, resvg_ref_dir, size, save_dir) for svg_path in svg_files]

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
    parser.add_argument("--save", action="store_true", help="Save comparison images (VectorStag | resvg | diff)")
    args = parser.parse_args()

    noto_dir = Path("SciStagEssentialData/images/noto")
    ref_base = Path("samples/references_400")
    comparison_base = Path("samples/comparison_400")

    if args.emojis:
        print("=" * 70)
        print("COMPARING EMOJIS")
        print("=" * 70)
        save_dir = comparison_base / "emojis" if args.save else None
        results, errors, buckets = test_directory(
            noto_dir / "emojis" / "svg",
            ref_base / "emojis",
            limit=args.limit,
            size=args.size,
            num_workers=args.workers,
            save_dir=save_dir
        )
        print_summary(results, errors, buckets, "EMOJIS")

    if args.flags:
        print("=" * 70)
        print("COMPARING FLAGS")
        print("=" * 70)
        save_dir = comparison_base / "flags" if args.save else None
        results, errors, buckets = test_directory(
            noto_dir / "flags" / "svg",
            ref_base / "flags",
            limit=args.limit,
            size=args.size,
            num_workers=args.workers,
            save_dir=save_dir
        )
        print_summary(results, errors, buckets, "FLAGS")

    if not (args.emojis or args.flags):
        print("Usage: python compare_fast.py [--emojis] [--flags] [--limit N] [-j WORKERS] [--save]")


if __name__ == "__main__":
    main()
