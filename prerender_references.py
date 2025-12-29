#!/usr/bin/env python3
"""Pre-render all SVGs with Cairo and resvg to create reference images."""

import os
import sys
from pathlib import Path
from PIL import Image
import cairosvg
from resvg_python import svg_to_png
import io
from multiprocessing import Pool, cpu_count
import time
import argparse


def render_cairo(svg_path: Path, output_path: Path, size: int = 400):
    """Render SVG with CairoSVG."""
    try:
        png_data = cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        img.save(output_path)
        return True
    except Exception as e:
        return False


def render_resvg(svg_path: Path, output_path: Path, size: int = 400):
    """Render SVG with resvg, respecting preserveAspectRatio setting."""
    try:
        with open(svg_path, 'r') as f:
            svg_content = f.read()

        # Check if SVG has preserveAspectRatio="none" - should stretch
        import re
        par_match = re.search(r'preserveAspectRatio\s*=\s*["\']([^"\']+)["\']', svg_content)
        should_stretch = par_match and 'none' in par_match.group(1).lower()

        png_data = bytes(svg_to_png(svg_content))
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")

        if img.size != (size, size):
            if should_stretch:
                # Stretch to fill entire canvas (preserveAspectRatio="none")
                img = img.resize((size, size), Image.Resampling.LANCZOS)
            else:
                # Preserve aspect ratio and center on transparent background
                scale = min(size / img.width, size / img.height)
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Center on transparent canvas
                canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
                offset_x = (size - new_w) // 2
                offset_y = (size - new_h) // 2
                canvas.paste(img, (offset_x, offset_y))
                img = canvas

        img.save(output_path)
        return True
    except Exception as e:
        return False


def process_svg(args):
    """Process a single SVG - render with both Cairo and resvg."""
    svg_path, cairo_dir, resvg_dir, size = args
    name = svg_path.stem

    cairo_ok = render_cairo(svg_path, cairo_dir / f"{name}.png", size)
    resvg_ok = render_resvg(svg_path, resvg_dir / f"{name}.png", size)

    return name, cairo_ok, resvg_ok


def prerender_directory(svg_dir: Path, output_base: Path, size: int = 400, num_workers: int = None):
    """Pre-render all SVGs in a directory."""
    svg_files = sorted(svg_dir.glob("*.svg"))

    if not svg_files:
        print(f"No SVG files found in {svg_dir}")
        return

    cairo_dir = output_base / "cairo"
    resvg_dir = output_base / "resvg"
    cairo_dir.mkdir(parents=True, exist_ok=True)
    resvg_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Pre-rendering {len(svg_files)} SVGs from {svg_dir}")
    print(f"Output: {output_base}")
    print(f"Resolution: {size}x{size}")
    print(f"Workers: {num_workers}\n")

    tasks = [(svg_path, cairo_dir, resvg_dir, size) for svg_path in svg_files]

    start_time = time.time()
    cairo_ok = 0
    resvg_ok = 0

    with Pool(num_workers) as pool:
        completed = 0
        for name, c_ok, r_ok in pool.imap_unordered(process_svg, tasks):
            completed += 1
            if c_ok:
                cairo_ok += 1
            if r_ok:
                resvg_ok += 1

            if completed % 500 == 0 or completed == len(svg_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  {completed}/{len(svg_files)} - {rate:.1f} files/sec")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({len(svg_files)/elapsed:.1f} files/sec)")
    print(f"Cairo: {cairo_ok}/{len(svg_files)} successful")
    print(f"resvg: {resvg_ok}/{len(svg_files)} successful")


def main():
    parser = argparse.ArgumentParser(description="Pre-render SVGs with Cairo and resvg")
    parser.add_argument("--emojis", action="store_true", help="Render emojis")
    parser.add_argument("--flags", action="store_true", help="Render flags")
    parser.add_argument("--all", action="store_true", help="Render all")
    parser.add_argument("--size", type=int, default=400, help="Render size (default: 400)")
    parser.add_argument("--workers", "-j", type=int, default=None, help="Number of workers")
    args = parser.parse_args()

    noto_dir = Path("SciStagEssentialData/images/noto")
    output_base = Path("samples/references_400")

    if args.emojis or args.all:
        print("=" * 70)
        print("PRE-RENDERING EMOJIS")
        print("=" * 70)
        prerender_directory(
            noto_dir / "emojis" / "svg",
            output_base / "emojis",
            size=args.size,
            num_workers=args.workers
        )

    if args.flags or args.all:
        print("\n" + "=" * 70)
        print("PRE-RENDERING FLAGS")
        print("=" * 70)
        prerender_directory(
            noto_dir / "flags" / "svg",
            output_base / "flags",
            size=args.size,
            num_workers=args.workers
        )

    if not (args.emojis or args.flags or args.all):
        print("Usage: python prerender_references.py [--emojis] [--flags] [--all] [--size N] [-j WORKERS]")


if __name__ == "__main__":
    main()
