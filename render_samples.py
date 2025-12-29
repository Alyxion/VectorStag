#!/usr/bin/env python3
"""Render all flags and emojis with comparison images."""

import os
import shutil
from pathlib import Path
from PIL import Image
import cairosvg
import io
import numpy as np
from vectorstag import SVGRenderer


def render_comparison(svg_path: Path, output_dir: Path, size: int = 128):
    """Render SVG and create comparison image."""
    name = svg_path.stem

    try:
        # Copy SVG
        shutil.copy(svg_path, output_dir / f"{name}.svg")

        # Render with VectorStag
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
        vs_img = renderer.render_file(str(svg_path), size, size)

        # Render with CairoSVG
        try:
            png_data = cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)
            cairo_img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        except Exception:
            cairo_img = Image.new("RGBA", (size, size), (255, 0, 0, 128))  # Red for error

        # Create comparison image: [VectorStag | Cairo | Diff]
        white_bg = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        vs_comp = Image.alpha_composite(white_bg.copy(), vs_img)
        cairo_comp = Image.alpha_composite(white_bg.copy(), cairo_img)

        # Calculate difference
        arr1 = np.array(vs_comp, dtype=np.float32)[:, :, :3]
        arr2 = np.array(cairo_comp, dtype=np.float32)[:, :, :3]
        diff = np.abs(arr1 - arr2)
        diff_enhanced = np.clip(diff * 3, 0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_enhanced, mode="RGB")

        # Calculate similarity
        mse = np.mean((arr1 / 255.0 - arr2 / 255.0) ** 2)
        similarity = 1.0 - min(1.0, mse * 4)

        # Combine images
        combined = Image.new("RGB", (size * 3 + 4, size + 20), (240, 240, 240))
        combined.paste(vs_comp.convert("RGB"), (0, 0))
        combined.paste(cairo_comp.convert("RGB"), (size + 2, 0))
        combined.paste(diff_img, (size * 2 + 4, 0))

        # Add labels
        from PIL import ImageDraw
        draw = ImageDraw.Draw(combined)
        draw.text((5, size + 2), f"VS", fill=(0, 0, 0))
        draw.text((size + 7, size + 2), f"Cairo", fill=(0, 0, 0))
        draw.text((size * 2 + 9, size + 2), f"Diff {similarity:.0%}", fill=(0, 0, 0))

        combined.save(output_dir / f"{name}_compare.png")

        return name, similarity
    except Exception as e:
        return name, f"error: {e}"


def process_directory(svg_dir: Path, output_dir: Path, limit: int = None):
    """Process all SVGs in directory."""
    svg_files = sorted(svg_dir.glob("*.svg"))
    if limit:
        svg_files = svg_files[:limit]

    print(f"Processing {len(svg_files)} files from {svg_dir}...")

    results = []
    for i, svg_path in enumerate(svg_files):
        result = render_comparison(svg_path, output_dir)
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(svg_files)}")

    # Summary
    successes = [(n, s) for n, s in results if isinstance(s, float)]
    if successes:
        avg = np.mean([s for _, s in successes])
        above99 = sum(1 for _, s in successes if s >= 0.99)
        print(f"  Done: {len(successes)} files, avg {avg:.1%}, {above99} at >=99%")

    return results


def main():
    # Process flags
    print("\n=== RENDERING FLAGS ===")
    flags_dir = Path("SciStagEssentialData/images/noto/flags/svg")
    if flags_dir.exists():
        process_directory(flags_dir, Path("samples_ext/flags"))

    # Process emojis
    print("\n=== RENDERING EMOJIS ===")
    emojis_dir = Path("SciStagEssentialData/images/noto/emojis/svg")
    if emojis_dir.exists():
        process_directory(emojis_dir, Path("samples_ext/emojis"))

    print("\nDone! Check samples_ext/ for results.")


if __name__ == "__main__":
    main()
