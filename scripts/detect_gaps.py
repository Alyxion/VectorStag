#!/usr/bin/env python3
"""
Gap detection tool for VectorStag SVG rendering.

Detects interior gaps in rendered SVGs - pixels that should be filled but aren't.
These are NOT antialiasing artifacts (which occur at edges) but actual rendering bugs.

Usage:
    python scripts/detect_gaps.py --categories lucide material
    python scripts/detect_gaps.py --all --limit 100
    python scripts/detect_gaps.py --file path/to/file.svg
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstag import SVGRenderer
from PIL import Image
import numpy as np


COLLECTIONS = {
    "emojis": Path("SciStagEssentialData/images/noto/emojis/svg"),
    "flags": Path("SciStagEssentialData/images/noto/flags/svg"),
    "material": Path("advanced_svg/material"),
    "fontawesome": Path("advanced_svg/fontawesome"),
    "lucide": Path("advanced_svg/lucide"),
    "w3c": Path("samples/svg"),
}


def detect_interior_gaps(arr: np.ndarray, threshold: int = 60) -> list:
    """
    Detect interior gaps: light pixels surrounded by dark pixels.

    These are NOT antialiasing artifacts (at edges) but actual holes
    in what should be solid filled regions.

    Returns list of (x, y, pixel_value, num_dark_neighbors)
    """
    gaps = []
    h, w = arr.shape[:2]

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            # Get grayscale value (use red channel for simplicity)
            px = int(arr[y, x, 0])

            # Only look for light/white pixels
            if px < 200:
                continue

            # Count dark neighbors in all 8 directions
            dark_count = 0
            neighbor_vals = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    nval = int(arr[y + dy, x + dx, 0])
                    neighbor_vals.append(nval)
                    if nval < px - threshold:
                        dark_count += 1

            # Interior gap: white pixel surrounded by mostly dark pixels
            # (at least 7 of 8 neighbors are significantly darker)
            # Use 7 to avoid false positives at corners and edges
            if dark_count >= 7:
                gaps.append((x, y, px, dark_count, neighbor_vals))

    return gaps


def detect_line_gaps(arr: np.ndarray) -> list:
    """
    Detect gaps in what should be continuous strokes.

    Look for patterns like: dark-white-dark in a line that suggests
    a broken stroke.
    """
    gaps = []
    h, w = arr.shape[:2]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            px = int(arr[y, x, 0])

            # Only look for white/light pixels
            if px < 220:
                continue

            # Check horizontal pattern: dark-light-dark
            left = int(arr[y, x - 1, 0])
            right = int(arr[y, x + 1, 0])
            if left < 100 and right < 100:
                # Also check vertical neighbors to confirm it's not an edge
                top = int(arr[y - 1, x, 0])
                bottom = int(arr[y + 1, x, 0])
                if top < 150 or bottom < 150:
                    gaps.append((x, y, 'horizontal', px))
                    continue

            # Check vertical pattern
            top = int(arr[y - 1, x, 0])
            bottom = int(arr[y + 1, x, 0])
            if top < 100 and bottom < 100:
                left = int(arr[y, x - 1, 0])
                right = int(arr[y, x + 1, 0])
                if left < 150 or right < 150:
                    gaps.append((x, y, 'vertical', px))

    return gaps


def analyze_svg(svg_path: Path, size: int = 256) -> dict:
    """Analyze a single SVG for gaps."""
    try:
        renderer = SVGRenderer(background=(255, 255, 255, 255), antialias=4)
        img = renderer.render_file(str(svg_path), width=size, height=size)
        if img is None:
            return {"path": str(svg_path), "error": "render returned None"}

        arr = np.array(img)

        interior_gaps = detect_interior_gaps(arr)
        line_gaps = detect_line_gaps(arr)

        return {
            "path": str(svg_path),
            "name": svg_path.name,
            "interior_gaps": len(interior_gaps),
            "line_gaps": len(line_gaps),
            "gap_details": interior_gaps[:5] if interior_gaps else [],
            "line_details": line_gaps[:5] if line_gaps else [],
        }
    except Exception as e:
        return {"path": str(svg_path), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Detect rendering gaps in SVGs")
    parser.add_argument("--categories", "-c", nargs="+",
                        choices=list(COLLECTIONS.keys()) + ["all"],
                        default=["lucide"])
    parser.add_argument("--file", "-f", type=str, help="Analyze single file")
    parser.add_argument("--size", "-s", type=int, default=256,
                        help="Render size (default: 256)")
    parser.add_argument("--limit", "-n", type=int, default=50,
                        help="Max files per category (default: 50)")
    parser.add_argument("--workers", "-j", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--threshold", "-t", type=int, default=0,
                        help="Min gaps to report (default: 0 = all)")
    parser.add_argument("--save-bad", action="store_true",
                        help="Save problematic renders to /tmp/gaps/")
    args = parser.parse_args()

    if args.file:
        # Single file mode
        result = analyze_svg(Path(args.file), args.size)
        print(f"\n{result['name']}:")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Interior gaps: {result['interior_gaps']}")
            print(f"  Line gaps: {result['line_gaps']}")
            if result['gap_details']:
                print("  Gap locations:")
                for x, y, px, dc, _ in result['gap_details']:
                    print(f"    ({x}, {y}): val={px}, dark_neighbors={dc}")
        return

    # Multi-file mode
    if "all" in args.categories:
        categories = list(COLLECTIONS.keys())
    else:
        categories = args.categories

    print(f"\n{'='*60}")
    print("VectorStag Gap Detection")
    print(f"{'='*60}")

    if args.save_bad:
        bad_dir = Path("/tmp/gaps")
        bad_dir.mkdir(exist_ok=True)
        print(f"Saving problematic renders to: {bad_dir}")

    all_results = []

    for cat_name in categories:
        cat_dir = COLLECTIONS.get(cat_name)
        if not cat_dir or not cat_dir.exists():
            print(f"\n{cat_name}: directory not found")
            continue

        svg_files = list(cat_dir.glob("*.svg"))[:args.limit]
        if not svg_files:
            svg_files = list(cat_dir.rglob("*.svg"))[:args.limit]

        if not svg_files:
            print(f"\n{cat_name}: no SVG files found")
            continue

        print(f"\n{cat_name} ({len(svg_files)} files)")
        print("-" * 40)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(analyze_svg, f, args.size): f
                      for f in svg_files}

            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Report files with gaps
                if "error" not in result:
                    total_gaps = result["interior_gaps"] + result["line_gaps"]
                    if total_gaps > args.threshold:
                        print(f"  {result['name']}: {result['interior_gaps']} interior, {result['line_gaps']} line gaps")

                        if args.save_bad and total_gaps > 0:
                            # Save the rendered image
                            renderer = SVGRenderer(background=(255, 255, 255, 255), antialias=4)
                            img = renderer.render_file(result['path'], width=args.size, height=args.size)
                            if img:
                                save_path = bad_dir / f"{cat_name}_{Path(result['path']).stem}.png"
                                img.save(save_path)

        # Summary for category
        error_count = sum(1 for r in results if "error" in r)
        gap_files = [r for r in results if "error" not in r and
                     (r["interior_gaps"] > 0 or r["line_gaps"] > 0)]

        print(f"  Summary: {len(gap_files)}/{len(results)} files have gaps, {error_count} errors")
        all_results.extend(results)

    # Overall summary
    print(f"\n{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")

    total_files = len(all_results)
    error_files = sum(1 for r in all_results if "error" in r)
    files_with_gaps = [r for r in all_results if "error" not in r and
                       (r["interior_gaps"] > 0 or r["line_gaps"] > 0)]

    print(f"Total files analyzed: {total_files}")
    print(f"Files with gaps: {len(files_with_gaps)}")
    print(f"Files with errors: {error_files}")

    if files_with_gaps:
        print(f"\nTop 10 files with most gaps:")
        sorted_gaps = sorted(files_with_gaps,
                            key=lambda r: r["interior_gaps"] + r["line_gaps"],
                            reverse=True)
        for r in sorted_gaps[:10]:
            print(f"  {r['name']}: {r['interior_gaps']} interior, {r['line_gaps']} line")


if __name__ == "__main__":
    main()
