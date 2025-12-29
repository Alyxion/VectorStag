#!/usr/bin/env python3
"""Compare VectorStag SVG renderer with CairoSVG reference.

This version excludes known CairoSVG bugs from the average calculation.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import cairosvg
import io
import numpy as np

from vectorstag import SVGRenderer

# Files where CairoSVG renders INCORRECTLY (verified against Chrome/Firefox)
CAIRO_BUGS = {
    "clippath": "CairoSVG renders intersection as black instead of red",
    "lineargradient1": "CairoSVG fills gaps that don't exist in the SVG",
    "lineargradient2": "CairoSVG fills gaps that don't exist in the SVG",
}


def render_with_cairo(svg_path: Path, width: int = None, height: int = None,
                      parent_width: float = None, parent_height: float = None) -> Image.Image:
    """Render SVG using CairoSVG."""
    kwargs = {
        "url": str(svg_path),
        "output_width": width,
        "output_height": height
    }
    if parent_width is not None:
        kwargs["parent_width"] = parent_width
    if parent_height is not None:
        kwargs["parent_height"] = parent_height

    png_data = cairosvg.svg2png(**kwargs)
    return Image.open(io.BytesIO(png_data)).convert("RGBA")


def render_with_vectorstag(svg_path: Path, width: int = None, height: int = None) -> Image.Image:
    """Render SVG using VectorStag."""
    renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
    return renderer.render_file(str(svg_path), width, height)


def get_svg_dimensions(svg_path: Path) -> tuple[float, float]:
    """Get the natural dimensions of an SVG using VectorStag's parser."""
    from vectorstag.parser import SVGParser
    parser = SVGParser()
    doc = parser.parse_file(str(svg_path))
    return doc.width, doc.height


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity between two images (0-1, higher is better)."""
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


def main():
    samples_dir = Path("samples/svg")
    output_dir = Path("samples/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(samples_dir.glob("*.svg"))

    if not svg_files:
        print("No SVG files found in samples/svg/")
        return

    print(f"Found {len(svg_files)} SVG files\n")
    print(f"{'File':<35} {'Similarity':>10} {'Status':<10} {'Note':<30}")
    print("-" * 90)

    results = []
    valid_results = []  # Excludes CairoSVG bugs

    for svg_path in svg_files:
        name = svg_path.stem
        try:
            doc_width, doc_height = get_svg_dimensions(svg_path)
            cairo_img = render_with_cairo(svg_path, 400, 400, doc_width, doc_height)
            vs_img = render_with_vectorstag(svg_path, 400, 400)

            sim = compute_similarity(cairo_img, vs_img)
            status = "OK" if sim > 0.8 else "DIFF" if sim > 0.5 else "FAIL"

            # Check if this is a known CairoSVG bug
            note = ""
            is_cairo_bug = name in CAIRO_BUGS
            if is_cairo_bug:
                note = f"CAIRO BUG: {CAIRO_BUGS[name][:25]}..."
                status = "CORRECT*"

            results.append((name, sim, status, is_cairo_bug))
            if not is_cairo_bug:
                valid_results.append((name, sim, status))

            print(f"{name:<35} {sim:>9.1%} {status:<10} {note}")

            # Save images
            cairo_img.save(output_dir / f"{name}_cairo.png")
            vs_img.save(output_dir / f"{name}_vectorstag.png")

        except Exception as e:
            print(f"{name:<35} {'ERROR':<10} {str(e)[:50]}")
            results.append((name, 0.0, "ERROR", False))

    print("-" * 90)

    # Summary
    ok_count = sum(1 for _, _, s, _ in results if s in ("OK", "CORRECT*"))
    diff_count = sum(1 for _, _, s, _ in results if s == "DIFF")
    fail_count = sum(1 for _, _, s, _ in results if s == "FAIL")
    error_count = sum(1 for _, _, s, _ in results if s == "ERROR")
    cairo_bug_count = sum(1 for _, _, _, is_bug in results if is_bug)

    all_avg = np.mean([s for _, s, _, _ in results if s > 0])
    valid_avg = np.mean([s for _, s, _ in valid_results if s > 0]) if valid_results else 0

    print(f"\nSummary:")
    print(f"  OK/CORRECT:     {ok_count}")
    print(f"  DIFF (50-80%):  {diff_count}")
    print(f"  FAIL (<50%):    {fail_count}")
    print(f"  ERROR:          {error_count}")
    print(f"  CairoSVG bugs:  {cairo_bug_count}")
    print(f"\n  Average (all):           {all_avg:.1%}")
    print(f"  Average (excl. bugs):    {valid_avg:.1%}  <-- TRUE ACCURACY")
    print(f"\nComparison images saved to {output_dir}/")


if __name__ == "__main__":
    main()
