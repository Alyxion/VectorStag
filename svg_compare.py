#!/usr/bin/env python3
"""
Unified SVG comparison tool for VectorStag.

Features:
- Pre-render references with Cairo and resvg
- Fast comparison against pre-rendered references
- Generate comparison grid PNGs (VectorStag | resvg | diff)
- Support for multiple SVG collections

Usage:
    # Pre-render references (run once per collection)
    python svg_compare.py prerender --emojis --flags --material --w3c -j 16

    # Fast comparison (no PNG output)
    python svg_compare.py compare --emojis --flags -j 16

    # Comparison with PNG grid output
    python svg_compare.py compare --emojis --save -j 16

    # List available collections
    python svg_compare.py list
"""

import argparse
import io
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image

# Worker timeout in seconds (prevent infinite loops)
WORKER_TIMEOUT = 30

# Optional imports - fail gracefully
try:
    import cairosvg
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False

try:
    from resvg_python import svg_to_png
    HAS_RESVG = True
except ImportError:
    HAS_RESVG = False

from vectorstag import SVGRenderer


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Collection:
    """SVG collection configuration."""
    name: str
    svg_dir: Path
    ref_dir: Path
    output_dir: Path
    size: int = 400
    description: str = ""


def get_collections(base_ref_dir: Path = None, base_output_dir: Path = None) -> Dict[str, Collection]:
    """Get all available SVG collections."""
    if base_ref_dir is None:
        base_ref_dir = Path("references")
    if base_output_dir is None:
        base_output_dir = Path("comparisons")

    noto_dir = Path("SciStagEssentialData/images/noto")

    collections = {
        "emojis": Collection(
            name="emojis",
            svg_dir=noto_dir / "emojis" / "svg",
            ref_dir=base_ref_dir / "emojis",
            output_dir=base_output_dir / "emojis",
            size=400,
            description="Noto Color Emojis (3427 files)"
        ),
        "flags": Collection(
            name="flags",
            svg_dir=noto_dir / "flags" / "svg",
            ref_dir=base_ref_dir / "flags",
            output_dir=base_output_dir / "flags",
            size=400,
            description="Noto Flags (358 files)"
        ),
        "material": Collection(
            name="material",
            svg_dir=Path("advanced_svg/material"),
            ref_dir=base_ref_dir / "material",
            output_dir=base_output_dir / "material",
            size=256,
            description="Material Design Icons (336 files)"
        ),
        "fontawesome": Collection(
            name="fontawesome",
            svg_dir=Path("advanced_svg/fontawesome"),
            ref_dir=base_ref_dir / "fontawesome",
            output_dir=base_output_dir / "fontawesome",
            size=128,
            description="FontAwesome Icons"
        ),
        "lucide": Collection(
            name="lucide",
            svg_dir=Path("advanced_svg/lucide"),
            ref_dir=base_ref_dir / "lucide",
            output_dir=base_output_dir / "lucide",
            size=128,
            description="Lucide Icons"
        ),
        "w3c": Collection(
            name="w3c",
            svg_dir=Path("samples/svg"),
            ref_dir=base_ref_dir / "w3c",
            output_dir=base_output_dir / "w3c",
            size=400,
            description="W3C SVG Test Suite samples"
        ),
    }

    return collections


# =============================================================================
# Image Utilities
# =============================================================================

def get_svg_dimensions(svg_path: Path) -> Tuple[float, float]:
    """Parse SVG to get dimensions."""
    try:
        with open(svg_path, 'r') as f:
            content = f.read()

        root = ET.fromstring(content)

        # Try width/height attributes first
        width_str = root.get('width', '')
        height_str = root.get('height', '')

        # Strip units and convert
        width = float(re.sub(r'[^0-9.]', '', width_str) or '0')
        height = float(re.sub(r'[^0-9.]', '', height_str) or '0')

        # Fall back to viewBox if either dimension is missing
        if width <= 0 or height <= 0:
            viewbox = root.get('viewBox', '')
            vb_parts = viewbox.split()
            if len(vb_parts) == 4:
                if width <= 0:
                    width = float(vb_parts[2])
                if height <= 0:
                    height = float(vb_parts[3])

        # If still missing dimensions, use VectorStag parser (calculates bounding box)
        if width <= 0 or height <= 0:
            from vectorstag.parser import SVGParser
            parser = SVGParser()
            doc = parser.parse_file(str(svg_path))
            if width <= 0:
                width = doc.width if doc.width > 0 else 100
            if height <= 0:
                height = doc.height if doc.height > 0 else 100

        return width, height

    except Exception:
        return 100, 100


def should_stretch(svg_path: Path) -> bool:
    """Check if SVG has preserveAspectRatio='none'."""
    try:
        with open(svg_path, 'r') as f:
            content = f.read()
        match = re.search(r'preserveAspectRatio\s*=\s*["\']([^"\']+)["\']', content)
        return match and 'none' in match.group(1).lower()
    except Exception:
        return False


def calculate_render_size(svg_w: float, svg_h: float, target_size: int) -> Tuple[int, int]:
    """Calculate render dimensions preserving aspect ratio."""
    aspect = svg_w / svg_h if svg_h > 0 else 1
    if aspect >= 1:
        render_w = target_size
        raw_h = target_size / aspect
        render_h = round(raw_h) if abs(raw_h - round(raw_h)) < 0.001 else int(raw_h)
    else:
        render_h = target_size
        raw_w = target_size * aspect
        render_w = round(raw_w) if abs(raw_w - round(raw_w)) < 0.001 else int(raw_w)

    return max(1, render_w), max(1, render_h)


def fit_to_canvas(img: Image.Image, size: int) -> Image.Image:
    """Fit image to canvas, centered with transparent background."""
    if img.size == (size, size):
        return img

    # Scale to fit
    scale = min(size / img.width, size / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)

    if new_w != img.width or new_h != img.height:
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on canvas
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    canvas.paste(img, (offset_x, offset_y))

    return canvas


def create_diff_image(img1: Image.Image, img2: Image.Image, size: int) -> Image.Image:
    """Create diff image highlighting differences in magenta."""
    if img1 is None or img2 is None:
        return Image.new("RGB", (size, size), (128, 128, 128))

    # Composite on white
    white = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    img1_comp = Image.alpha_composite(white, fit_to_canvas(img1, size))
    img2_comp = Image.alpha_composite(white, fit_to_canvas(img2, size))

    # Convert to arrays
    arr1 = np.array(img1_comp, dtype=np.int16)[:, :, :3]
    arr2 = np.array(img2_comp, dtype=np.int16)[:, :, :3]

    # Compute difference
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff, axis=2)

    # Create output: darkened original with magenta highlights
    base = np.array(img1_comp)[:, :, :3].astype(np.float32) * 0.3
    mask = max_diff > 10
    diff_amplified = np.clip(max_diff * 3, 0, 255)

    result = base.copy()
    result[mask, 0] = np.clip(base[mask, 0] + diff_amplified[mask], 0, 255)
    result[mask, 1] = base[mask, 1] * 0.3
    result[mask, 2] = np.clip(base[mask, 2] + diff_amplified[mask], 0, 255)

    return Image.fromarray(result.astype(np.uint8), mode="RGB")


def create_comparison_grid(vs_img: Image.Image, resvg_img: Image.Image, size: int) -> Image.Image:
    """Create comparison grid: VectorStag | resvg | diff."""
    white = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    grid = Image.new("RGB", (size * 3, size), (255, 255, 255))

    vs_fitted = fit_to_canvas(vs_img, size) if vs_img else Image.new("RGBA", (size, size), (200, 200, 200, 255))
    resvg_fitted = fit_to_canvas(resvg_img, size) if resvg_img else Image.new("RGBA", (size, size), (200, 200, 200, 255))

    # Composite on white and place
    grid.paste(Image.alpha_composite(white, vs_fitted).convert("RGB"), (0, 0))
    grid.paste(Image.alpha_composite(white, resvg_fitted).convert("RGB"), (size, 0))
    grid.paste(create_diff_image(vs_fitted, resvg_fitted, size), (size * 2, 0))

    return grid


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity between two images (0.0 - 1.0)."""
    if img1 is None or img2 is None:
        return 0.0

    # Resize to same size
    size = (max(img1.width, img2.width), max(img1.height, img2.height))
    if img1.size != size:
        img1 = img1.resize(size, Image.Resampling.LANCZOS)
    if img2.size != size:
        img2 = img2.resize(size, Image.Resampling.LANCZOS)

    # Composite on white
    white = Image.new("RGBA", size, (255, 255, 255, 255))
    img1_comp = Image.alpha_composite(white, img1)
    img2_comp = Image.alpha_composite(white, img2)

    # Compute MSE-based similarity
    arr1 = np.array(img1_comp, dtype=np.float32)[:, :, :3] / 255.0
    arr2 = np.array(img2_comp, dtype=np.float32)[:, :, :3] / 255.0

    mse = np.mean((arr1 - arr2) ** 2)
    return max(0.0, 1.0 - min(1.0, mse * 4))


# =============================================================================
# File naming helpers
# =============================================================================

def get_unique_name(svg_path: Path, base_dir: Path) -> str:
    """Get unique name for an SVG file, handling subdirectory structure.

    For files in subdirectories, use path relative to base_dir with underscores.
    e.g., 'brands/twitter.svg' -> 'brands_twitter'
    """
    try:
        rel_path = svg_path.relative_to(base_dir)
        # Replace path separators with underscores, remove .svg extension
        parts = list(rel_path.parts)
        parts[-1] = parts[-1].replace('.svg', '')
        return '_'.join(parts)
    except ValueError:
        return svg_path.stem


# =============================================================================
# Pre-rendering
# =============================================================================

def render_with_cairo(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Render SVG with CairoSVG."""
    if not HAS_CAIRO:
        return None

    try:
        png_data = cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)
        return Image.open(io.BytesIO(png_data)).convert("RGBA")
    except Exception:
        return None


def render_with_resvg(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Render SVG with resvg."""
    if not HAS_RESVG:
        return None

    try:
        with open(svg_path, 'r') as f:
            content = f.read()

        png_data = bytes(svg_to_png(content))
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")

        # Fit to size, preserving aspect ratio
        stretch = should_stretch(svg_path)
        if img.size != (size, size):
            if stretch:
                img = img.resize((size, size), Image.Resampling.LANCZOS)
            else:
                img = fit_to_canvas(img, size)

        return img
    except Exception:
        return None


def render_with_vectorstag(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Render SVG with VectorStag."""
    try:
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)

        svg_w, svg_h = get_svg_dimensions(svg_path)
        stretch = should_stretch(svg_path)

        if stretch:
            img = renderer.render_file(str(svg_path), size, size)
        else:
            render_w, render_h = calculate_render_size(svg_w, svg_h, size)
            img = renderer.render_file(str(svg_path), render_w, render_h)

            if img is not None and img.size != (size, size):
                img = fit_to_canvas(img, size)

        return img
    except Exception:
        return None


def prerender_worker(args):
    """Worker function for pre-rendering."""
    svg_path, base_dir, cairo_dir, resvg_dir, size = args
    name = get_unique_name(svg_path, base_dir)

    cairo_ok = False
    resvg_ok = False

    try:
        if cairo_dir:
            img = render_with_cairo(svg_path, size)
            if img:
                img.save(cairo_dir / f"{name}.png")
                cairo_ok = True

        if resvg_dir:
            img = render_with_resvg(svg_path, size)
            if img:
                img.save(resvg_dir / f"{name}.png")
                resvg_ok = True
    except Exception:
        pass

    return name, cairo_ok, resvg_ok


def prerender_collection(collection: Collection, num_workers: int = None,
                         render_cairo: bool = True, render_resvg: bool = True):
    """Pre-render all SVGs in a collection."""
    svg_files = sorted(collection.svg_dir.glob("**/*.svg"))

    if not svg_files:
        print(f"No SVG files found in {collection.svg_dir}")
        return

    cairo_dir = collection.ref_dir / "cairo" if render_cairo and HAS_CAIRO else None
    resvg_dir = collection.ref_dir / "resvg" if render_resvg and HAS_RESVG else None

    if cairo_dir:
        cairo_dir.mkdir(parents=True, exist_ok=True)
    if resvg_dir:
        resvg_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Pre-rendering {len(svg_files)} SVGs from {collection.svg_dir}")
    print(f"Output: {collection.ref_dir}")
    print(f"Size: {collection.size}x{collection.size}")
    print(f"Workers: {num_workers}")
    if cairo_dir:
        print(f"Cairo: enabled")
    if resvg_dir:
        print(f"resvg: enabled")
    print()

    tasks = [(svg_path, collection.svg_dir, cairo_dir, resvg_dir, collection.size) for svg_path in svg_files]

    start_time = time.time()
    cairo_ok = resvg_ok = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(prerender_worker, task): task for task in tasks}

        completed = 0
        for future in as_completed(future_to_task):
            completed += 1

            try:
                name, c_ok, r_ok = future.result(timeout=WORKER_TIMEOUT)
                cairo_ok += c_ok
                resvg_ok += r_ok
            except FuturesTimeoutError:
                pass  # Timeout, skip this file
            except Exception:
                pass  # Error, skip this file

            if completed % 500 == 0 or completed == len(svg_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  {completed}/{len(svg_files)} - {rate:.1f} files/sec")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({len(svg_files)/elapsed:.1f} files/sec)")
    if cairo_dir:
        print(f"Cairo: {cairo_ok}/{len(svg_files)} successful")
    if resvg_dir:
        print(f"resvg: {resvg_ok}/{len(svg_files)} successful")


# =============================================================================
# Comparison
# =============================================================================

def get_resvg_native_aspect(svg_path: Path) -> float:
    """Get resvg's native render aspect ratio for an SVG."""
    if not HAS_RESVG:
        return 1.0
    try:
        with open(svg_path, 'r') as f:
            content = f.read()
        png_data = bytes(svg_to_png(content))
        img = Image.open(io.BytesIO(png_data))
        return img.width / img.height if img.height > 0 else 1.0
    except Exception:
        return 1.0


def render_vectorstag_for_comparison(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Render VectorStag at appropriate dimensions and fit to canvas."""
    try:
        renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)

        # Check if SVG should be stretched (preserveAspectRatio="none")
        stretch = should_stretch(svg_path)

        if stretch:
            # Render at target size (stretched to fill)
            img = renderer.render_file(str(svg_path), size, size)
        else:
            # Render at native dimensions to preserve aspect ratio
            img = renderer.render_file(str(svg_path))

        if img is None:
            return None
        return fit_to_canvas(img, size)
    except Exception:
        return None


def compare_worker(args):
    """Worker function for comparison."""
    svg_path, base_dir, resvg_ref_dir, cairo_ref_dir, size, save_dir = args
    name = get_unique_name(svg_path, base_dir)

    try:
        # Load references first (to get aspect ratio for VectorStag rendering)
        resvg_path = resvg_ref_dir / f"{name}.png"
        cairo_path = cairo_ref_dir / f"{name}.png" if cairo_ref_dir else None

        resvg_img = Image.open(resvg_path).convert("RGBA") if resvg_path.exists() else None
        cairo_img = Image.open(cairo_path).convert("RGBA") if cairo_path and cairo_path.exists() else None

        # Render VectorStag for comparison
        vs_img = render_vectorstag_for_comparison(svg_path, size)

        if vs_img is None:
            return {"name": name, "error": "VectorStag render failed"}

        # Compute similarities
        sim_resvg = compute_similarity(vs_img, resvg_img) if resvg_img else 0.0
        sim_cairo = compute_similarity(vs_img, cairo_img) if cairo_img else 0.0
        sim = max(sim_resvg, sim_cairo)

        # Save comparison grid if requested
        if save_dir is not None:
            grid = create_comparison_grid(vs_img, resvg_img, size)
            grid.save(save_dir / f"{name}.png")

        return {
            "name": name,
            "sim": sim,
            "sim_resvg": sim_resvg,
            "sim_cairo": sim_cairo
        }

    except Exception as e:
        return {"name": name, "error": str(e)[:50]}


def compare_collection(collection: Collection, num_workers: int = None,
                       save_grids: bool = False, limit: int = None):
    """Compare VectorStag against pre-rendered references."""
    svg_files = sorted(collection.svg_dir.glob("**/*.svg"))
    if limit:
        svg_files = svg_files[:limit]

    if not svg_files:
        print(f"No SVG files found in {collection.svg_dir}")
        return [], [], {}

    resvg_ref_dir = collection.ref_dir / "resvg"
    cairo_ref_dir = collection.ref_dir / "cairo"

    if not resvg_ref_dir.exists():
        print(f"Reference directory not found: {resvg_ref_dir}")
        print(f"Run: python svg_compare.py prerender --{collection.name} first")
        return [], [], {}

    save_dir = None
    if save_grids:
        save_dir = collection.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving comparison grids to: {save_dir}")

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Comparing {len(svg_files)} SVGs from {collection.svg_dir}")
    print(f"References: {collection.ref_dir}")
    print(f"Size: {collection.size}x{collection.size}")
    print(f"Workers: {num_workers}")
    print()

    cairo_ref_dir_arg = cairo_ref_dir if cairo_ref_dir.exists() else None
    tasks = [(svg_path, collection.svg_dir, resvg_ref_dir, cairo_ref_dir_arg, collection.size, save_dir)
             for svg_path in svg_files]

    results = []
    errors = []
    buckets = defaultdict(list)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(compare_worker, task): task for task in tasks}

        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            task = future_to_task[future]
            name = get_unique_name(task[0], task[1])

            try:
                result = future.result(timeout=WORKER_TIMEOUT)

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

            except FuturesTimeoutError:
                errors.append((name, "timeout"))
            except Exception as e:
                errors.append((name, str(e)[:50]))

            if completed % 500 == 0 or completed == len(svg_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                avg = np.mean([s for _, s, _, _ in results]) if results else 0
                print(f"  {completed}/{len(svg_files)} - Avg: {avg:.1%} - {rate:.1f} files/sec")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({len(svg_files)/elapsed:.1f} files/sec)")

    return results, errors, buckets


def print_summary(results, errors, buckets, title=""):
    """Print comparison summary."""
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
        for name, err in errors[:5]:
            print(f"  {name}: {err}")


# =============================================================================
# CLI
# =============================================================================

def cmd_prerender(args):
    """Handle prerender command."""
    collections = get_collections()

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

    for collection in selected:
        if not collection.svg_dir.exists():
            print(f"Skipping {collection.name}: {collection.svg_dir} not found")
            continue

        print("\n" + "=" * 70)
        print(f"PRE-RENDERING: {collection.name.upper()}")
        print("=" * 70)
        prerender_collection(
            collection,
            num_workers=args.workers,
            render_cairo=not args.no_cairo,
            render_resvg=not args.no_resvg
        )


def cmd_compare(args):
    """Handle compare command."""
    collections = get_collections()

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

    for collection in selected:
        if not collection.svg_dir.exists():
            print(f"Skipping {collection.name}: {collection.svg_dir} not found")
            continue

        print("\n" + "=" * 70)
        print(f"COMPARING: {collection.name.upper()}")
        print("=" * 70)
        results, errors, buckets = compare_collection(
            collection,
            num_workers=args.workers,
            save_grids=args.save,
            limit=args.limit
        )
        print_summary(results, errors, buckets, collection.name.upper())


def cmd_list(args):
    """Handle list command."""
    collections = get_collections()

    print("\nAvailable collections:")
    print("-" * 70)
    for name, col in collections.items():
        exists = col.svg_dir.exists()
        count = len(list(col.svg_dir.glob("**/*.svg"))) if exists else 0
        ref_exists = (col.ref_dir / "resvg").exists()

        status = "ready" if exists and ref_exists else "no refs" if exists else "not found"
        print(f"  {name:15} {count:5} files  [{status}]  {col.description}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified SVG comparison tool for VectorStag",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prerender command
    pre_parser = subparsers.add_parser("prerender", help="Pre-render references with Cairo and resvg")
    pre_parser.add_argument("--emojis", action="store_true", help="Noto emojis")
    pre_parser.add_argument("--flags", action="store_true", help="Noto flags")
    pre_parser.add_argument("--material", action="store_true", help="Material icons")
    pre_parser.add_argument("--fontawesome", action="store_true", help="FontAwesome icons")
    pre_parser.add_argument("--lucide", action="store_true", help="Lucide icons")
    pre_parser.add_argument("--w3c", action="store_true", help="W3C samples")
    pre_parser.add_argument("--all", action="store_true", help="All collections")
    pre_parser.add_argument("-j", "--workers", type=int, help="Number of workers")
    pre_parser.add_argument("--no-cairo", action="store_true", help="Skip Cairo rendering")
    pre_parser.add_argument("--no-resvg", action="store_true", help="Skip resvg rendering")

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare VectorStag against references")
    cmp_parser.add_argument("--emojis", action="store_true", help="Noto emojis")
    cmp_parser.add_argument("--flags", action="store_true", help="Noto flags")
    cmp_parser.add_argument("--material", action="store_true", help="Material icons")
    cmp_parser.add_argument("--fontawesome", action="store_true", help="FontAwesome icons")
    cmp_parser.add_argument("--lucide", action="store_true", help="Lucide icons")
    cmp_parser.add_argument("--w3c", action="store_true", help="W3C samples")
    cmp_parser.add_argument("--all", action="store_true", help="All collections")
    cmp_parser.add_argument("-j", "--workers", type=int, help="Number of workers")
    cmp_parser.add_argument("--save", action="store_true", help="Save comparison grid PNGs")
    cmp_parser.add_argument("--limit", type=int, help="Limit number of files")

    # List command
    subparsers.add_parser("list", help="List available collections")

    args = parser.parse_args()

    if args.command == "prerender":
        cmd_prerender(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
