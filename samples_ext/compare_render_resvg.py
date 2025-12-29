#!/usr/bin/env python3
"""Compare VectorStag SVG renderer with resvg reference (more accurate than CairoSVG)."""

import os
import sys
from pathlib import Path
from PIL import Image
from resvg_python import svg_to_png
import io
import numpy as np

from vectorstag import SVGRenderer


def render_with_resvg(svg_path: Path, width: int = None, height: int = None) -> Image.Image:
    """Render SVG using resvg."""
    with open(svg_path, 'r') as f:
        svg_content = f.read()

    png_data = bytes(svg_to_png(svg_content))
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")

    # Resize if dimensions specified
    if width and height:
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    return img


def render_with_vectorstag(svg_path: Path, width: int = None, height: int = None) -> Image.Image:
    """Render SVG using VectorStag."""
    renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
    return renderer.render_file(str(svg_path), width, height)


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


def create_comparison_image(ref_img: Image.Image, vs_img: Image.Image) -> Image.Image:
    """Create a side-by-side comparison image."""
    width = max(ref_img.width, vs_img.width)
    height = max(ref_img.height, vs_img.height)

    ref_img = ref_img.resize((width, height), Image.Resampling.LANCZOS)
    vs_img = vs_img.resize((width, height), Image.Resampling.LANCZOS)

    comp = Image.new("RGBA", (width * 3 + 20, height + 40), (240, 240, 240, 255))
    comp.paste(ref_img, (0, 30))
    comp.paste(vs_img, (width + 10, 30))

    arr1 = np.array(ref_img, dtype=np.float32)
    arr2 = np.array(vs_img, dtype=np.float32)
    diff = np.abs(arr1 - arr2)
    diff_img = Image.fromarray(diff.astype(np.uint8))
    comp.paste(diff_img, (width * 2 + 20, 30))

    return comp


def main():
    samples_dir = Path("samples/svg")
    output_dir = Path("samples/comparison_resvg")
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(samples_dir.glob("*.svg"))

    if not svg_files:
        print("No SVG files found in samples/svg/")
        return

    print(f"Comparing VectorStag vs resvg (accurate reference)")
    print(f"Found {len(svg_files)} SVG files\n")
    print(f"{'File':<35} {'Similarity':>10} {'Status':<10}")
    print("-" * 60)

    results = []

    for svg_path in svg_files:
        name = svg_path.stem
        try:
            resvg_img = render_with_resvg(svg_path, 400, 400)
            vs_img = render_with_vectorstag(svg_path, 400, 400)

            sim = compute_similarity(resvg_img, vs_img)
            status = "OK" if sim > 0.8 else "DIFF" if sim > 0.5 else "FAIL"

            results.append((name, sim, status))
            print(f"{name:<35} {sim:>9.1%} {status:<10}")

            # Save comparison
            comp = create_comparison_image(resvg_img, vs_img)
            comp.save(output_dir / f"{name}_comparison.png")
            resvg_img.save(output_dir / f"{name}_resvg.png")
            vs_img.save(output_dir / f"{name}_vectorstag.png")

        except Exception as e:
            print(f"{name:<35} {'ERROR':<10} {str(e)[:40]}")
            results.append((name, 0.0, "ERROR"))

    print("-" * 60)

    ok_count = sum(1 for _, _, s in results if s == "OK")
    diff_count = sum(1 for _, _, s in results if s == "DIFF")
    fail_count = sum(1 for _, _, s in results if s == "FAIL")
    error_count = sum(1 for _, _, s in results if s == "ERROR")
    avg_sim = np.mean([s for _, s, _ in results if s > 0])

    print(f"\nSummary (vs resvg reference):")
    print(f"  OK (>80%):    {ok_count}")
    print(f"  DIFF (50-80%): {diff_count}")
    print(f"  FAIL (<50%):  {fail_count}")
    print(f"  ERROR:        {error_count}")
    print(f"  Average:      {avg_sim:.1%}")
    print(f"\nComparison images saved to {output_dir}/")


if __name__ == "__main__":
    main()
