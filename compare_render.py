#!/usr/bin/env python3
"""Compare VectorStag SVG renderer with CairoSVG reference."""

import os
import sys
from pathlib import Path
from PIL import Image
import cairosvg
import io
import numpy as np

from vectorstag import SVGRenderer


def render_with_cairo(svg_path: Path, width: int = None, height: int = None) -> Image.Image:
    """Render SVG using CairoSVG."""
    png_data = cairosvg.svg2png(
        url=str(svg_path),
        output_width=width,
        output_height=height
    )
    return Image.open(io.BytesIO(png_data)).convert("RGBA")


def render_with_vectorstag(svg_path: Path, width: int = None, height: int = None) -> Image.Image:
    """Render SVG using VectorStag."""
    renderer = SVGRenderer(background=(0, 0, 0, 0))  # Transparent background
    return renderer.render_file(str(svg_path), width, height)


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity between two images (0-1, higher is better)."""
    # Resize to same size
    size = (max(img1.width, img2.width), max(img1.height, img2.height))
    img1 = img1.resize(size, Image.Resampling.LANCZOS)
    img2 = img2.resize(size, Image.Resampling.LANCZOS)

    # Composite both images onto white background for fair comparison
    white_bg = Image.new("RGBA", size, (255, 255, 255, 255))
    img1_comp = Image.alpha_composite(white_bg, img1)
    img2_comp = Image.alpha_composite(white_bg, img2)

    # Convert to numpy (RGB only, since alpha is now all 255)
    arr1 = np.array(img1_comp, dtype=np.float32)[:, :, :3] / 255.0
    arr2 = np.array(img2_comp, dtype=np.float32)[:, :, :3] / 255.0

    # Compute MSE
    mse = np.mean((arr1 - arr2) ** 2)

    # Convert to similarity (1 - normalized MSE)
    similarity = 1.0 - min(1.0, mse * 4)  # Scale for visibility
    return max(0.0, similarity)


def create_comparison_image(cairo_img: Image.Image, vs_img: Image.Image,
                            name: str) -> Image.Image:
    """Create a side-by-side comparison image."""
    # Ensure same size
    width = max(cairo_img.width, vs_img.width)
    height = max(cairo_img.height, vs_img.height)

    cairo_img = cairo_img.resize((width, height), Image.Resampling.LANCZOS)
    vs_img = vs_img.resize((width, height), Image.Resampling.LANCZOS)

    # Create comparison (Cairo | VectorStag | Diff)
    comp = Image.new("RGBA", (width * 3 + 20, height + 40), (240, 240, 240, 255))

    # Paste images
    comp.paste(cairo_img, (0, 30))
    comp.paste(vs_img, (width + 10, 30))

    # Compute difference
    arr1 = np.array(cairo_img, dtype=np.float32)
    arr2 = np.array(vs_img, dtype=np.float32)
    diff = np.abs(arr1 - arr2)
    diff_img = Image.fromarray(diff.astype(np.uint8))
    comp.paste(diff_img, (width * 2 + 20, 30))

    return comp


def main():
    samples_dir = Path("samples/svg")
    output_dir = Path("samples/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all SVG files
    svg_files = sorted(samples_dir.glob("*.svg"))

    if not svg_files:
        print("No SVG files found in samples/svg/")
        return

    print(f"Found {len(svg_files)} SVG files\n")
    print(f"{'File':<35} {'Similarity':>10} {'Status':<10}")
    print("-" * 60)

    results = []

    for svg_path in svg_files:
        name = svg_path.stem
        try:
            # Render with both engines
            cairo_img = render_with_cairo(svg_path, 400, 400)
            vs_img = render_with_vectorstag(svg_path, 400, 400)

            # Compute similarity
            sim = compute_similarity(cairo_img, vs_img)
            status = "OK" if sim > 0.8 else "DIFF" if sim > 0.5 else "FAIL"

            results.append((name, sim, status))
            print(f"{name:<35} {sim:>9.1%} {status:<10}")

            # Save comparison image
            comp = create_comparison_image(cairo_img, vs_img, name)
            comp.save(output_dir / f"{name}_comparison.png")

            # Save individual renders
            cairo_img.save(output_dir / f"{name}_cairo.png")
            vs_img.save(output_dir / f"{name}_vectorstag.png")

        except Exception as e:
            print(f"{name:<35} {'ERROR':<10} {str(e)[:30]}")
            results.append((name, 0.0, "ERROR"))

    print("-" * 60)

    # Summary
    ok_count = sum(1 for _, _, s in results if s == "OK")
    diff_count = sum(1 for _, _, s in results if s == "DIFF")
    fail_count = sum(1 for _, _, s in results if s == "FAIL")
    error_count = sum(1 for _, _, s in results if s == "ERROR")
    avg_sim = np.mean([s for _, s, _ in results if s > 0])

    print(f"\nSummary:")
    print(f"  OK (>80%):    {ok_count}")
    print(f"  DIFF (50-80%): {diff_count}")
    print(f"  FAIL (<50%):  {fail_count}")
    print(f"  ERROR:        {error_count}")
    print(f"  Average:      {avg_sim:.1%}")
    print(f"\nComparison images saved to {output_dir}/")


if __name__ == "__main__":
    main()
