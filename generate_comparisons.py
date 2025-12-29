#!/usr/bin/env python3
"""Generate comparison images for all SVGs using multiprocessing."""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import argparse


def create_diff_image(img1, img2, size, threshold=10):
    """Create a diff image highlighting differences in magenta."""
    white_bg = (255, 255, 255, 255)
    s = (size, size)

    if img1 is None or img2 is None:
        # Return gray if one is missing
        return Image.new("RGB", s, (128, 128, 128))

    # Resize if needed
    if img1.size != s:
        img1 = img1.resize(s, Image.Resampling.LANCZOS)
    if img2.size != s:
        img2 = img2.resize(s, Image.Resampling.LANCZOS)

    # Composite on white
    white = Image.new("RGBA", s, white_bg)
    img1_comp = Image.alpha_composite(white, img1)
    img2_comp = Image.alpha_composite(white, img2)

    # Convert to arrays
    arr1 = np.array(img1_comp, dtype=np.int16)[:, :, :3]
    arr2 = np.array(img2_comp, dtype=np.int16)[:, :, :3]

    # Compute difference
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff, axis=2)

    # Create output: show original image darkened, with differences in magenta
    base = np.array(img1_comp)[:, :, :3].astype(np.float32) * 0.3  # Darken original

    # Highlight differences in magenta (amplified)
    mask = max_diff > threshold
    diff_amplified = np.clip(max_diff * 3, 0, 255)  # Amplify differences

    result = base.copy()
    result[mask, 0] = np.clip(base[mask, 0] + diff_amplified[mask], 0, 255)  # Red
    result[mask, 1] = base[mask, 1] * 0.3  # Reduce green
    result[mask, 2] = np.clip(base[mask, 2] + diff_amplified[mask], 0, 255)  # Blue -> Magenta

    return Image.fromarray(result.astype(np.uint8), mode="RGB")


def process_svg(args):
    """Process a single SVG - render with VectorStag and create comparison grid."""
    svg_path, cairo_ref_dir, resvg_ref_dir, output_dir, size = args
    name = svg_path.stem

    # Import here for multiprocessing
    from vectorstag import SVGRenderer

    try:
        # Render with VectorStag
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
        vs_img = renderer.render_file(str(svg_path), size, size)

        if vs_img is None:
            return name, 0.0, "render_failed"

        # Load cached references
        cairo_path = cairo_ref_dir / f"{name}.png"
        resvg_path = resvg_ref_dir / f"{name}.png"

        cairo_img = Image.open(cairo_path).convert("RGBA") if cairo_path.exists() else None
        resvg_img = Image.open(resvg_path).convert("RGBA") if resvg_path.exists() else None

        # Resize references if needed
        if cairo_img and cairo_img.size != (size, size):
            cairo_img = cairo_img.resize((size, size), Image.Resampling.LANCZOS)
        if resvg_img and resvg_img.size != (size, size):
            resvg_img = resvg_img.resize((size, size), Image.Resampling.LANCZOS)

        white_bg = (255, 255, 255, 255)

        # Create 2x3 grid:
        # Row 1: VectorStag | Cairo | resvg
        # Row 2: VS-Cairo diff | Cairo-resvg diff | VS-resvg diff
        grid_width = size * 3
        grid_height = size * 2
        combined = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        # Row 1: Original renders
        vs_composite = Image.alpha_composite(Image.new("RGBA", (size, size), white_bg), vs_img)
        combined.paste(vs_composite.convert("RGB"), (0, 0))

        if cairo_img:
            cairo_composite = Image.alpha_composite(Image.new("RGBA", (size, size), white_bg), cairo_img)
            combined.paste(cairo_composite.convert("RGB"), (size, 0))

        if resvg_img:
            resvg_composite = Image.alpha_composite(Image.new("RGBA", (size, size), white_bg), resvg_img)
            combined.paste(resvg_composite.convert("RGB"), (size * 2, 0))

        # Row 2: Diff images
        diff_vs_cairo = create_diff_image(vs_img, cairo_img, size)
        diff_cairo_resvg = create_diff_image(cairo_img, resvg_img, size)
        diff_vs_resvg = create_diff_image(vs_img, resvg_img, size)

        combined.paste(diff_vs_cairo, (0, size))
        combined.paste(diff_cairo_resvg, (size, size))
        combined.paste(diff_vs_resvg, (size * 2, size))

        # Save comparison
        combined.save(output_dir / f"{name}_comparison.png")

        # Compute similarity
        def compute_similarity(img1, img2):
            if img1 is None or img2 is None:
                return 0.0
            s = (size, size)
            if img1.size != s:
                img1 = img1.resize(s, Image.Resampling.LANCZOS)
            if img2.size != s:
                img2 = img2.resize(s, Image.Resampling.LANCZOS)
            white = Image.new("RGBA", s, white_bg)
            img1_comp = Image.alpha_composite(white, img1)
            img2_comp = Image.alpha_composite(white, img2)
            arr1 = np.array(img1_comp, dtype=np.float32)[:, :, :3] / 255.0
            arr2 = np.array(img2_comp, dtype=np.float32)[:, :, :3] / 255.0
            mse = np.mean((arr1 - arr2) ** 2)
            return max(0.0, 1.0 - min(1.0, mse * 4))

        sim_cairo = compute_similarity(cairo_img, vs_img) if cairo_img else 0.0
        sim_resvg = compute_similarity(resvg_img, vs_img) if resvg_img else 0.0
        sim = max(sim_cairo, sim_resvg)

        return name, sim, "ok"
    except Exception as e:
        return name, 0.0, str(e)[:50]


def generate_comparisons(svg_dir: Path, ref_dir: Path, output_dir: Path,
                         size: int = 400, num_workers: int = None):
    """Generate comparison images for all SVGs in a directory."""
    svg_files = sorted(svg_dir.glob("*.svg"))

    if not svg_files:
        print(f"No SVG files found in {svg_dir}")
        return

    cairo_ref_dir = ref_dir / "cairo"
    resvg_ref_dir = ref_dir / "resvg"
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Generating {len(svg_files)} comparison images")
    print(f"Output: {output_dir}")
    print(f"Resolution: {size}x{size}")
    print(f"Workers: {num_workers}\n")

    tasks = [(svg_path, cairo_ref_dir, resvg_ref_dir, output_dir, size)
             for svg_path in svg_files]

    start_time = time.time()
    completed = 0
    errors = 0
    total_sim = 0.0

    with Pool(num_workers) as pool:
        for name, sim, status in pool.imap_unordered(process_svg, tasks):
            completed += 1
            if status == "ok":
                total_sim += sim
            else:
                errors += 1

            if completed % 50 == 0 or completed == len(svg_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                avg_sim = total_sim / (completed - errors) if (completed - errors) > 0 else 0
                timestamp = time.strftime("%H:%M:%S")
                print(f"  [{timestamp}] {completed}/{len(svg_files)} - Avg: {avg_sim:.1%} - {rate:.1f} files/sec - Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({len(svg_files)/elapsed:.1f} files/sec)")
    print(f"Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison images for all SVGs")
    parser.add_argument("--emojis", action="store_true", help="Generate emoji comparisons")
    parser.add_argument("--flags", action="store_true", help="Generate flag comparisons")
    parser.add_argument("--all", action="store_true", help="Generate all comparisons")
    parser.add_argument("--size", type=int, default=400, help="Image size (default: 400)")
    parser.add_argument("-j", "--workers", type=int, help="Number of workers")
    args = parser.parse_args()

    noto_dir = Path("SciStagEssentialData/images/noto")
    ref_base = Path("samples/references_400")
    output_base = Path("samples/comparison_400")

    if args.emojis or args.all:
        print("=" * 70)
        print("GENERATING EMOJI COMPARISONS")
        print("=" * 70)
        generate_comparisons(
            noto_dir / "emojis" / "svg",
            ref_base / "emojis",
            output_base / "emojis",
            size=args.size,
            num_workers=args.workers
        )

    if args.flags or args.all:
        print("\n" + "=" * 70)
        print("GENERATING FLAG COMPARISONS")
        print("=" * 70)
        generate_comparisons(
            noto_dir / "flags" / "svg",
            ref_base / "flags",
            output_base / "flags",
            size=args.size,
            num_workers=args.workers
        )

    if not (args.emojis or args.flags or args.all):
        print("Usage: python generate_comparisons.py [--emojis] [--flags] [--all] [-j WORKERS]")


if __name__ == "__main__":
    main()
