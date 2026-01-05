#!/usr/bin/env python3
"""
Generate comparison masks between VectorStag and reference renderers.

For each SVG in selected collections, renders VectorStag and compares it to
each available reference PNG (resvg, Cairo, Chrome), producing a mask image
highlighting pixel differences.

Outputs grayscale or binary masks indicating per-pixel max-channel differences
after compositing on a white background and fitting to a square canvas.

Usage examples:
  # Save binary masks (threshold 10, default) for emojis to comparisons/masks
  python scripts/generate_comparison_masks.py --emojis --output comparisons/masks -j 8

  # Save grayscale masks for all collections with custom threshold and limit
  python scripts/generate_comparison_masks.py --all --mode grayscale --threshold 6 --limit 200 -j 8 
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import svg_compare as sc


def compute_mask(img1: Image.Image, img2: Image.Image, size: int, threshold: int, mode: str) -> Image.Image:
    # Composite images on white
    white = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    i1 = Image.alpha_composite(white, sc.fit_to_canvas(img1, size))
    i2 = Image.alpha_composite(white, sc.fit_to_canvas(img2, size))

    a1 = np.array(i1, dtype=np.int16)[:, :, :3]
    a2 = np.array(i2, dtype=np.int16)[:, :, :3]
    d = np.abs(a1 - a2)
    maxd = np.max(d, axis=2)

    if mode == "binary":
        mask = (maxd > threshold).astype(np.uint8) * 255
    else:
        mask = np.clip(maxd, 0, 255).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def load_ref(ref_dir: Path, backend: str, name: str) -> Optional[Image.Image]:
    p = ref_dir / backend / f"{name}.png"
    if p.exists():
        try:
            return Image.open(p).convert("RGBA")
        except Exception:
            return None
    return None


def worker(args):
    svg_path, base_dir, ref_dir, out_dir, size, threshold, mode = args
    name = sc.get_unique_name(svg_path, base_dir)
    try:
        vs = sc.render_vectorstag_for_comparison(svg_path, size)
        if vs is None:
            return (name, 0)

        done = 0
        for backend in ("resvg", "cairo", "chrome"):
            ref = load_ref(ref_dir, backend, name)
            if ref is None:
                continue
            m = compute_mask(vs, ref, size, threshold, mode)
            out_dir.mkdir(parents=True, exist_ok=True)
            m.save(out_dir / f"{name}_vs_{backend}_mask.png")
            done += 1
        return (name, done)
    except Exception:
        return (name, 0)


def main():
    parser = argparse.ArgumentParser(description="Generate VectorStag vs reference comparison masks")
    parser.add_argument("--emojis", action="store_true")
    parser.add_argument("--flags", action="store_true")
    parser.add_argument("--material", action="store_true")
    parser.add_argument("--fontawesome", action="store_true")
    parser.add_argument("--lucide", action="store_true")
    parser.add_argument("--w3c", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("-j", "--workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=str, default="comparisons/masks")
    parser.add_argument("--mode", choices=["binary", "grayscale"], default="binary")
    parser.add_argument("--threshold", type=int, default=10)

    args = parser.parse_args()

    collections = sc.get_collections()
    selected = []
    if args.all:
        selected = list(collections.values())
    else:
        for name in ['emojis', 'flags', 'material', 'fontawesome', 'lucide', 'w3c']:
            if getattr(args, name, False):
                if name in collections:
                    selected.append(collections[name])

    if not selected:
        print("No collections selected. Use --emojis, --flags, --material, etc. or --all")
        return

    workers = args.workers or min(cpu_count(), 16)
    out_root = Path(args.output)

    for col in selected:
        if not col.svg_dir.exists():
            print(f"Skipping {col.name}: {col.svg_dir} not found")
            continue

        svg_files = sorted(col.svg_dir.glob("**/*.svg"))
        if args.limit:
            svg_files = svg_files[: args.limit]
        if not svg_files:
            print(f"No SVG files found in {col.svg_dir}")
            continue

        print("\n" + "=" * 70)
        print(f"MASKS: {col.name.upper()}  mode={args.mode}  thr={args.threshold}")
        print("=" * 70)
        out_dir = out_root / col.name
        tasks = [(p, col.svg_dir, col.ref_dir, out_dir, col.size, args.threshold, args.mode) for p in svg_files]

        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    _, n = fut.result(timeout=30)
                    done += n
                except Exception:
                    pass
                if i % 500 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} processed - masks saved: {done}")

        print(f"Completed. Total masks saved: {done}")


if __name__ == "__main__":
    main()

