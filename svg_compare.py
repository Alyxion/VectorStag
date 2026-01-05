#!/usr/bin/env python3
"""
Unified SVG comparison tool for VectorStag.

Features:
- Pre-render references with Cairo, resvg, and Chrome (if available)
- Fast comparison against pre-rendered references
- Generate comparison grid PNGs (VectorStag | resvg | diff)
- Multi-renderer similarity matrix across VectorStag, resvg, Cairo, Chrome
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

    # Multi-renderer similarity matrix (VectorStag, resvg, Cairo, Chrome)
    python svg_compare.py matrix --emojis --limit 200 -j 8
"""

import argparse
import os
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
from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile
import shutil

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

# Optional: Playwright for robust Chromium rendering
HAS_PLAYWRIGHT = False
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except Exception:
    HAS_PLAYWRIGHT = False

# Cached Playwright context (single process reuse)
_PW = None
_PW_BROWSER = None
_PW_CONTEXT = None
_PW_PAGE = None

# Chrome backend selection: 'auto' (prefer playwright), 'playwright', or 'cli'
CHROME_BACKEND = 'auto'


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
        "resvgtests": Collection(
            name="resvgtests",
            svg_dir=Path("resvg-test-suite/tests"),
            ref_dir=base_ref_dir / "resvgtests",
            output_dir=base_output_dir / "resvgtests",
            size=400,
            description="resvg test suite (1679 tests)"
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
    # Prefer Python binding if available, else CLI fallback
    if HAS_RESVG:
        try:
            with open(svg_path, 'r') as f:
                content = f.read()

            png_data = bytes(svg_to_png(content))
            img = Image.open(io.BytesIO(png_data)).convert("RGBA")

            # Composite over white background for consistent comparison
            white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white_bg, img)

            # Fit to size, preserving aspect ratio
            stretch = should_stretch(svg_path)
            if img.size != (size, size):
                if stretch:
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                else:
                    img = fit_to_canvas(img, size)

            return img
        except Exception:
            pass

    # CLI fallback
    return render_with_resvg_cli(svg_path, size)


def find_resvg_executable() -> Optional[str]:
    """Locate a `resvg` CLI executable via env and PATH."""
    for key in ("RESVG_BIN", "RESVG_PATH"):
        val = os.environ.get(key)
        if val and Path(val).exists():
            return val
    p = shutil.which("resvg")
    return p


def render_with_resvg_cli(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Render SVG with resvg CLI and fit to canvas.

    Requires `resvg` available in PATH or via RESVG_BIN/RESVG_PATH.
    """
    exe = find_resvg_executable()
    if not exe:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            out_png = tmp / "out.png"
            # Basic invocation: resvg input.svg output.png
            cmd = [exe, str(svg_path), str(out_png)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            if not out_png.exists():
                return None
            img = Image.open(out_png).convert("RGBA")
            # Fit to size, preserving aspect ratio unless stretch
            stretch = should_stretch(svg_path)
            if img.size != (size, size):
                if stretch:
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                else:
                    img = fit_to_canvas(img, size)
            return img
    except Exception:
        return None


def find_chrome_executable() -> Optional[str]:
    """Locate a Chrome/Chromium executable on common paths.

    Checks env vars CHROME_BIN/CHROME_PATH, common macOS and Linux locations, and PATH.
    """
    # Env overrides
    for key in ("CHROME_BIN", "CHROME_PATH"):
        val = os.environ.get(key)
        if val and Path(val).exists():
            return val

    # Common macOS locations
    mac_candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for c in mac_candidates:
        if Path(c).exists():
            return c

    # PATH-based executables
    for exe in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(exe)
        if p:
            return p

    return None


def render_with_chrome(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Render SVG with headless Chrome by screenshotting an HTML wrapper.

    Requires a Chrome/Chromium binary. Attempts to keep transparent background and
    preserve aspect via CSS so comparisons align with other renderers.
    """
    # Respect backend selection and prefer Playwright if available
    if (CHROME_BACKEND in ('playwright', 'auto')) and HAS_PLAYWRIGHT:
        try:
            img = render_with_chrome_playwright(svg_path, size)
            if img is not None:
                return img
        except Exception:
            pass
    if CHROME_BACKEND == 'playwright':
        # Explicitly requested Playwright but it failed or is missing
        return None
    chrome = find_chrome_executable()
    if not chrome:
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            out_png = tmp / "out.png"
            # HTML wrapper with transparent background and centered SVG using <img>
            # object-fit: contain preserves aspect; size is enforced by viewport/window-size
            stretch = should_stretch(svg_path)
            fit_mode = "fill" if stretch else "contain"
            html = f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <style>
      html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background: rgba(0,0,0,0); }}
      .wrap {{ width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0); }}
      img {{ width: 100%; height: 100%; object-fit: {fit_mode}; image-rendering: auto; background: rgba(0,0,0,0); }}
    </style>
  </head>
  <body>
    <div class=\"wrap\"><img src=\"file://{svg_path.absolute().as_posix()}\" /></div>
  </body>
  </html>
"""
            html_path = tmp / "index.html"
            html_path.write_text(html, encoding="utf-8")

            # Run Chrome headless to capture a screenshot of the fixed-size viewport
            url = f"file://{html_path.absolute().as_posix()}"
            cmd = [
                chrome,
                "--headless=new",
                f"--screenshot={out_png}",
                f"--window-size={size},{size}",
                "--force-device-scale-factor=1",
                "--disable-gpu",
                "--hide-scrollbars",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--force-color-profile=srgb",
                "--default-background-color=00000000",
                "--virtual-time-budget=2000",
                url,
            ]
            # Fallback for older versions not supporting headless=new
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            except subprocess.CalledProcessError:
                cmd[1] = "--headless"
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)

            if not out_png.exists():
                return None
            img = Image.open(out_png).convert("RGBA")
            # Heuristic: crop 1px uniform border if present (Chrome sometimes adds it)
            img = maybe_crop_uniform_border(img)
            # Ensure exact size
            if img.size != (size, size):
                img = img.resize((size, size), Image.Resampling.LANCZOS)
            return img
    except Exception:
        return None


def maybe_crop_uniform_border(img: Image.Image) -> Image.Image:
    """Detect and crop a uniform 1px border around the image, if present.

    Checks if the top/bottom rows and left/right cols are identical and equal to each other.
    If so, crops by 1px on all sides. Otherwise returns the original image.
    """
    if img.width < 3 or img.height < 3:
        return img
    arr = np.array(img)
    top = arr[0, :, :]
    bottom = arr[-1, :, :]
    left = arr[:, 0, :]
    right = arr[:, -1, :]
    # Compare uniformity
    def is_uniform(edge):
        first = edge[0]
        return np.all(edge == first)
    if is_uniform(top) and is_uniform(bottom) and is_uniform(left) and is_uniform(right):
        # Also ensure all four borders match each other at corners
        if np.array_equal(top[0], left[0]) and np.array_equal(top[-1], right[0]):
            return img.crop((1, 1, img.width - 1, img.height - 1))
    return img


def render_with_chrome_playwright(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Render using Playwright/Chromium with a reusable browser.

    Renders SVG at target size directly using viewport sizing.
    """
    global _PW, _PW_BROWSER, _PW_CONTEXT, _PW_PAGE
    if not HAS_PLAYWRIGHT:
        return None

    # Lazy-init browser (once per process)
    try:
        if _PW is None:
            _PW = sync_playwright().start()
        if _PW_BROWSER is None:
            executable = os.environ.get("CHROME_BIN") or os.environ.get("CHROME_PATH")
            launch_kwargs = {"headless": True, "args": ["--force-color-profile=srgb", "--disable-gpu", "--disable-web-security"]}
            if executable:
                launch_kwargs["executable_path"] = executable
            _PW_BROWSER = _PW.chromium.launch(**launch_kwargs)
    except Exception:
        return None

    # Create fresh context/page for each render to avoid state issues
    context = None
    page = None
    try:
        context = _PW_BROWSER.new_context(viewport={"width": size, "height": size}, device_scale_factor=1)
        page = context.new_page()

        # Navigate directly to SVG file - Chrome renders it natively
        page.goto(f"file://{svg_path.absolute()}", wait_until="load", timeout=5000)

        # Take full page screenshot
        data = page.screenshot(omit_background=True, full_page=False)
        img = Image.open(io.BytesIO(data)).convert("RGBA")

        # Should be exactly size x size from viewport
        if img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.LANCZOS)

        return img
    except Exception:
        return None
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass
        if context:
            try:
                context.close()
            except Exception:
                pass


def render_with_vectorstag(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Render SVG with VectorStag Rust renderer for best quality."""
    try:
        # Use Rust renderer directly for best quality
        from vectorstag_rust import VectorStagRenderer
        rust_renderer = VectorStagRenderer()

        with open(svg_path, 'r') as f:
            svg_content = f.read()

        svg_w, svg_h = get_svg_dimensions(svg_path)
        stretch = should_stretch(svg_path)

        if stretch:
            arr = rust_renderer.render(svg_content, width=size, height=size,
                                       background=(255, 255, 255, 255), antialias=4)
            img = Image.fromarray(arr, 'RGBA')
        else:
            render_w, render_h = calculate_render_size(svg_w, svg_h, size)
            arr = rust_renderer.render(svg_content, width=render_w, height=render_h,
                                       background=(255, 255, 255, 255), antialias=4)
            img = Image.fromarray(arr, 'RGBA')

            if img is not None and img.size != (size, size):
                img = fit_to_canvas(img, size)

        return img
    except Exception as e:
        # Fallback to Python renderer
        try:
            renderer = SVGRenderer(background=(255, 255, 255, 255), antialias=4)
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
    svg_path, base_dir, cairo_dir, resvg_dir, chrome_dir, size, force = args
    name = get_unique_name(svg_path, base_dir)

    cairo_ok = False
    resvg_ok = False
    chrome_ok = False

    def save_valid_image(img, path):
        """Save image only if it's valid (has content and proper size)."""
        if img is None:
            return False
        # Check image has reasonable dimensions
        if img.width < 1 or img.height < 1:
            return False
        # Check image isn't completely transparent/empty
        try:
            extrema = img.getextrema()
            # For RGBA, check if alpha channel has any non-zero pixels
            if len(extrema) >= 4 and extrema[3] == (0, 0):
                return False  # Completely transparent
            img.save(path, "PNG")
            return True
        except Exception:
            return False

    try:
        if cairo_dir:
            out_path = cairo_dir / f"{name}.png"
            if force or not out_path.exists():
                img = render_with_cairo(svg_path, size)
                cairo_ok = save_valid_image(img, out_path)
            else:
                cairo_ok = True  # Already exists

        if resvg_dir:
            out_path = resvg_dir / f"{name}.png"
            if force or not out_path.exists():
                img = render_with_resvg(svg_path, size)
                resvg_ok = save_valid_image(img, out_path)
            else:
                resvg_ok = True  # Already exists

        if chrome_dir:
            out_path = chrome_dir / f"{name}.png"
            if force or not out_path.exists():
                img = render_with_chrome(svg_path, size)
                chrome_ok = save_valid_image(img, out_path)
            else:
                chrome_ok = True  # Already exists
    except Exception:
        pass

    return name, cairo_ok, resvg_ok, chrome_ok


def prerender_collection(collection: Collection, num_workers: int = None,
                         render_cairo: bool = True, render_resvg: bool = True,
                         render_chrome: bool = True, override_size: Optional[int] = None,
                         chrome_serial: bool = False, force: bool = False):
    """Pre-render all SVGs in a collection."""
    svg_files = sorted(collection.svg_dir.glob("**/*.svg"))

    if not svg_files:
        print(f"No SVG files found in {collection.svg_dir}")
        return

    cairo_dir = collection.ref_dir / "cairo" if render_cairo and HAS_CAIRO else None
    enable_resvg = False
    if render_resvg:
        try:
            enable_resvg = HAS_RESVG or (find_resvg_executable() is not None)
        except Exception:
            enable_resvg = HAS_RESVG
    resvg_dir = collection.ref_dir / "resvg" if enable_resvg else None
    enable_chrome = False
    chosen_backend = None
    if render_chrome:
        if CHROME_BACKEND == 'playwright':
            enable_chrome = HAS_PLAYWRIGHT
            chosen_backend = 'playwright'
        elif CHROME_BACKEND == 'cli':
            enable_chrome = find_chrome_executable() is not None
            chosen_backend = 'cli'
        else:  # auto
            if HAS_PLAYWRIGHT:
                enable_chrome = True
                chosen_backend = 'playwright'
            else:
                enable_chrome = find_chrome_executable() is not None
                chosen_backend = 'cli' if enable_chrome else None
    chrome_dir = collection.ref_dir / "chrome" if enable_chrome else None

    if cairo_dir:
        cairo_dir.mkdir(parents=True, exist_ok=True)
    if resvg_dir:
        resvg_dir.mkdir(parents=True, exist_ok=True)
    if chrome_dir:
        chrome_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 16)
    # Optionally serialize Chrome to avoid UI bouncing on macOS
    exec_workers = 1 if (chrome_dir and chrome_serial) else num_workers

    size = override_size or collection.size
    print(f"Pre-rendering {len(svg_files)} SVGs from {collection.svg_dir}")
    print(f"Output: {collection.ref_dir}")
    print(f"Size: {size}x{size}")
    if exec_workers != num_workers and chrome_dir:
        print(f"Workers: {num_workers} (chrome serialized -> {exec_workers})")
    else:
        print(f"Workers: {num_workers}")
    if cairo_dir:
        print(f"Cairo: enabled")
    else:
        print(f"Cairo: disabled (cairosvg not installed)")
    if resvg_dir:
        print(f"resvg: enabled")
    else:
        print(f"resvg: disabled (binding/CLI not found)")
    if chrome_dir:
        print(f"Chrome: enabled via {chosen_backend}")
    else:
        reason = 'playwright not installed' if (CHROME_BACKEND in ('playwright','auto') and not HAS_PLAYWRIGHT) else 'chrome/Chromium not found'
        print(f"Chrome: disabled ({reason})")
    print()

    tasks = [(svg_path, collection.svg_dir, cairo_dir, resvg_dir, chrome_dir, size, force) for svg_path in svg_files]

    start_time = time.time()
    cairo_ok = resvg_ok = chrome_ok = 0

    with ProcessPoolExecutor(max_workers=exec_workers) as executor:
        future_to_task = {executor.submit(prerender_worker, task): task for task in tasks}

        completed = 0
        for future in as_completed(future_to_task):
            completed += 1

            try:
                name, c_ok, r_ok, ch_ok = future.result(timeout=WORKER_TIMEOUT)
                cairo_ok += c_ok
                resvg_ok += r_ok
                chrome_ok += ch_ok
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
    if chrome_dir:
        print(f"Chrome: {chrome_ok}/{len(svg_files)} successful")


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
        # Use Python renderer for correctness (handles all SVG features)
        renderer = SVGRenderer(background=(255, 255, 255, 255), antialias=4)

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
# Multi-renderer matrix comparison
# =============================================================================

def matrix_worker(args):
    svg_path, base_dir, ref_dir, size = args
    name = get_unique_name(svg_path, base_dir)
    try:
        # Load available references
        imgs = {}
        # Pre-rendered backends
        for backend in ("resvg", "cairo", "chrome"):
            p = (ref_dir / backend / f"{name}.png")
            if p.exists():
                try:
                    imgs[backend] = Image.open(p).convert("RGBA")
                except Exception:
                    pass

        # VectorStag on-the-fly
        vs = render_vectorstag_for_comparison(svg_path, size)
        if vs is not None:
            imgs["vectorstag"] = vs

        # Compute pairwise similarities
        keys = sorted(imgs.keys())
        sims = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                sims[(a, b)] = compute_similarity(imgs[a], imgs[b])

        # Compute a per-file summary: minimal similarity across available pairs
        worst = 1.0
        for v in sims.values():
            if v < worst:
                worst = v
        return {"name": name, "sims": sims, "present": set(keys), "worst": worst}
    except Exception as e:
        return {"name": name, "error": str(e)[:80]}


def matrix_collection(collection: Collection, num_workers: int = None, limit: int = None):
    svg_files = sorted(collection.svg_dir.glob("**/*.svg"))
    if limit:
        svg_files = svg_files[:limit]

    if not svg_files:
        print(f"No SVG files found in {collection.svg_dir}")
        return {}

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print(f"Matrix compare on {len(svg_files)} SVGs from {collection.svg_dir}")
    print(f"References: {collection.ref_dir} (expect resvg/cairo/chrome subdirs)")
    print(f"Size: {collection.size}x{collection.size}")
    print(f"Workers: {num_workers}")
    print()

    tasks = [(svg_path, collection.svg_dir, collection.ref_dir, collection.size) for svg_path in svg_files]

    # Accumulators
    sums: Dict[tuple, float] = defaultdict(float)
    counts: Dict[tuple, int] = defaultdict(int)
    errors = []
    per_file = []

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(matrix_worker, t): t for t in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            t = future_to_task[future]
            name = get_unique_name(t[0], t[1])
            try:
                res = future.result(timeout=WORKER_TIMEOUT)
                if "error" in res:
                    errors.append((name, res["error"]))
                else:
                    per_file.append({"name": res["name"], "sims": res["sims"], "present": res["present"], "worst": res.get("worst", 1.0), "svg_path": t[0]})
                    for k, v in res["sims"].items():
                        sums[k] += v
                        counts[k] += 1
            except FuturesTimeoutError:
                errors.append((name, "timeout"))
            except Exception as e:
                errors.append((name, str(e)[:80]))

            if completed % 500 == 0 or completed == len(svg_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  {completed}/{len(svg_files)} - {rate:.1f} files/sec")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({len(svg_files)/elapsed:.1f} files/sec)")

    return {"sums": sums, "counts": counts, "errors": errors, "per_file": per_file}


def print_matrix(matrix_res, title: str = ""):
    sums: Dict[tuple, float] = matrix_res["sums"]
    counts: Dict[tuple, int] = matrix_res["counts"]
    errors = matrix_res.get("errors", [])

    print("\n" + "=" * 99)
    print(f"PAIRWISE SIMILARITY MATRIX{' - ' + title if title else ''}")
    print("=" * 99)
    pairs = [
        ("vectorstag", "resvg"),
        ("vectorstag", "cairo"),
        ("vectorstag", "chrome"),
        ("resvg", "cairo"),
        ("resvg", "chrome"),
        ("cairo", "chrome"),
    ]

    def get_pair_data(p):
        """Get count and sum for a pair, checking both orderings."""
        # Keys are sorted alphabetically in matrix_worker, so check canonical order
        canonical = tuple(sorted(p))
        if canonical in counts and counts[canonical] > 0:
            return counts[canonical], sums[canonical]
        return 0, 0.0

    # Header
    print("\nPair                     |  Avg   |  Count")
    print("-" * 99)
    for p in pairs:
        count, total = get_pair_data(p)
        if count > 0:
            avg = total / count
            print(f"{p[0]:>10} vs {p[1]:<10} | {avg:6.1%} | {count:6}")
        else:
            print(f"{p[0]:>10} vs {p[1]:<10} |   n/a  |      0")

    if errors:
        print(f"\nErrors: {len(errors)} (showing up to 5)")
        for n, e in errors[:5]:
            print(f"  {n}: {e}")


def create_big4_grid(vs_img: Optional[Image.Image], resvg_img: Optional[Image.Image],
                     cairo_img: Optional[Image.Image], chrome_img: Optional[Image.Image], size: int,
                     show_labels: bool = True) -> Image.Image:
    """Create a 3x4 grid:
    Row1: VectorStag | resvg | Cairo | Chrome (composited on white)
    Row2: VS vs resvg | VS vs Cairo | VS vs Chrome | label tile
    Row3: resvg vs VS | resvg vs Cairo | resvg vs Chrome | label tile
    """
    cols = 4
    rows = 3
    cell_w = size
    cell_h = size

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    white = Image.new("RGBA", (size, size), (255, 255, 255, 255))

    def fit(img):
        return fit_to_canvas(img, size) if img is not None else Image.new("RGBA", (size, size), (220, 220, 220, 255))

    # Row 1: originals
    tiles = [vs_img, resvg_img, cairo_img, chrome_img]
    for i, t in enumerate(tiles):
        comp = Image.alpha_composite(white, fit(t)).convert("RGB")
        grid.paste(comp, (i * cell_w, 0))
    # Labels row 1
    if show_labels:
        labels_r1 = ["VectorStag", "resvg", "Cairo", "Chrome"]
        _label_grid_row(grid, 0, labels_r1, cell_w, cell_h)

    # Row 2: VS diffs
    diffs_vs = [create_diff_image(fit(vs_img), fit(resvg_img), size),
                create_diff_image(fit(vs_img), fit(cairo_img), size),
                create_diff_image(fit(vs_img), fit(chrome_img), size)]
    for i, d in enumerate(diffs_vs):
        grid.paste(d, (i * cell_w, 1 * cell_h))
    # Label tile for column 4
    grid.paste(Image.new("RGB", (cell_w, cell_h), (245, 245, 245)), (3 * cell_w, 1 * cell_h))
    if show_labels:
        labels_r2 = ["VS vs resvg", "VS vs Cairo", "VS vs Chrome", "VS diffs"]
        _label_grid_row(grid, 1, labels_r2, cell_w, cell_h)

    # Row 3: resvg diffs
    diffs_rs = [create_diff_image(fit(resvg_img), fit(vs_img), size),
                create_diff_image(fit(resvg_img), fit(cairo_img), size),
                create_diff_image(fit(resvg_img), fit(chrome_img), size)]
    for i, d in enumerate(diffs_rs):
        grid.paste(d, (i * cell_w, 2 * cell_h))
    # Label tile for column 4
    grid.paste(Image.new("RGB", (cell_w, cell_h), (245, 245, 245)), (3 * cell_w, 2 * cell_h))
    if show_labels:
        labels_r3 = ["resvg vs VS", "resvg vs Cairo", "resvg vs Chrome", "resvg diffs"]
        _label_grid_row(grid, 2, labels_r3, cell_w, cell_h)

    # Similarity percentages on diff tiles (only when labels are on)
    if show_labels:
        # Prepare fitted images for similarity computation
        fvs, frs, fca, fch = fit(vs_img), fit(resvg_img), fit(cairo_img), fit(chrome_img)

        def fmt_sim(a: Optional[Image.Image], b: Optional[Image.Image]) -> str:
            if a is None or b is None:
                return "n/a"
            try:
                sim = compute_similarity(a, b)
                return f"{sim*100:.1f}%"
            except Exception:
                return "n/a"

        # Row 2
        sims_r2 = [fmt_sim(fvs, frs), fmt_sim(fvs, fca), fmt_sim(fvs, fch)]
        for i, s in enumerate(sims_r2):
            _label_cell(grid, 1, i, s, cell_w, cell_h, anchor="br")
        # Row 3
        sims_r3 = [fmt_sim(frs, fvs), fmt_sim(frs, fca), fmt_sim(frs, fch)]
        for i, s in enumerate(sims_r3):
            _label_cell(grid, 2, i, s, cell_w, cell_h, anchor="br")

    return grid


def _label_grid_row(grid: Image.Image, row: int, labels: List[str], cell_w: int, cell_h: int) -> None:
    """Draw simple labels in the top-left of each cell in a row."""
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 4
    for col, text in enumerate(labels):
        if not text:
            continue
        x = col * cell_w + pad
        y = row * cell_h + pad
        # background box sized to text
        try:
            bbox = draw.textbbox((x, y), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (8 * len(text), 10)
        box = [x - 2, y - 2, x + tw + 2, y + th + 2]
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((x, y), text, fill=(0, 0, 0), font=font)


def _label_cell(grid: Image.Image, row: int, col: int, text: str, cell_w: int, cell_h: int, anchor: str = "tl") -> None:
    """Draw a label inside a single cell. anchor: 'tl' top-left, 'br' bottom-right."""
    if not text:
        return
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 4
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = (8 * len(text), 10)
    if anchor == "br":
        x = (col + 1) * cell_w - tw - pad
        y = (row + 1) * cell_h - th - pad
    else:  # default top-left
        x = col * cell_w + pad
        y = row * cell_h + pad
    box = [x - 2, y - 2, x + tw + 2, y + th + 2]
    draw.rectangle(box, fill=(255, 255, 255))
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def build_and_save_grid(svg_path: Path, base_dir: Path, ref_dir: Path, size: int, out_path: Path,
                        show_labels: bool = True):
    name = get_unique_name(svg_path, base_dir)
    resvg_img = None
    cairo_img = None
    chrome_img = None
    p = ref_dir / "resvg" / f"{name}.png"
    if p.exists():
        try:
            resvg_img = Image.open(p).convert("RGBA")
        except Exception:
            pass
    p = ref_dir / "cairo" / f"{name}.png"
    if p.exists():
        try:
            cairo_img = Image.open(p).convert("RGBA")
        except Exception:
            pass
    p = ref_dir / "chrome" / f"{name}.png"
    if p.exists():
        try:
            chrome_img = Image.open(p).convert("RGBA")
        except Exception:
            pass
    vs_img = render_vectorstag_for_comparison(svg_path, size)
    grid = create_big4_grid(vs_img, resvg_img, cairo_img, chrome_img, size, show_labels=show_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


# =============================================================================
# CLI
# =============================================================================

def cmd_prerender(args):
    """Handle prerender command."""
    collections = get_collections()

    # Apply Chrome backend selection globally
    global CHROME_BACKEND
    CHROME_BACKEND = getattr(args, 'chrome_backend', 'playwright')

    # High-level backend info
    print("\nChrome rendering backend:")
    print(f"  requested: {CHROME_BACKEND}")
    print(f"  playwright available: {'yes' if HAS_PLAYWRIGHT else 'no'}")

    selected = []
    if args.all:
        selected = list(collections.values())
    else:
        for name in ['emojis', 'flags', 'material', 'fontawesome', 'lucide', 'w3c', 'resvgtests']:
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
            render_resvg=not args.no_resvg,
            render_chrome=not args.no_chrome,
            override_size=args.size,
            chrome_serial=args.chrome_serial,
            force=args.force
        )


def cmd_compare(args):
    """Handle compare command."""
    collections = get_collections()

    selected = []
    if args.all:
        selected = list(collections.values())
    else:
        for name in ['emojis', 'flags', 'material', 'fontawesome', 'lucide', 'w3c', 'resvgtests']:
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


def cmd_matrix(args):
    """Handle matrix command."""
    collections = get_collections()

    selected = []
    if args.all:
        selected = list(collections.values())
    else:
        for name in ['emojis', 'flags', 'material', 'fontawesome', 'lucide', 'w3c', 'resvgtests']:
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
        print(f"MATRIX: {collection.name.upper()}")
        print("=" * 70)
        matrix = matrix_collection(collection, num_workers=args.workers, limit=args.limit)
        print_matrix(matrix, collection.name.upper())

        # Optional saving of grids
        if args.save or args.save_all or args.save_top:
            per_file = matrix.get("per_file", [])
            out_dir = Path(args.save_dir) if args.save_dir else (collection.output_dir / "matrix")

            selected = per_file
            if args.save_all:
                selected = per_file
            elif args.save_top is not None:
                selected = sorted(per_file, key=lambda x: x.get("worst", 1.0))[: max(0, args.save_top)]
            elif args.save:
                # Default to top-4 worst if only --save is given
                selected = sorted(per_file, key=lambda x: x.get("worst", 1.0))[:4]

            print(f"\nSaving {len(selected)} grids to: {out_dir}")
            start = time.time()
            saved = 0
            for item in selected:
                name = item["name"]
                svg_path = item["svg_path"]
                out_path = out_dir / f"{name}.png"
                try:
                    build_and_save_grid(
                        svg_path,
                        collection.svg_dir,
                        collection.ref_dir,
                        collection.size,
                        out_path,
                        show_labels=(not args.labels_off),
                    )
                    saved += 1
                except Exception:
                    pass
            print(f"Saved {saved}/{len(selected)} grids in {time.time()-start:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Unified SVG comparison tool for VectorStag",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prerender command
    pre_parser = subparsers.add_parser("prerender", help="Pre-render references with Cairo, resvg, and Chrome")
    pre_parser.add_argument("--emojis", action="store_true", help="Noto emojis")
    pre_parser.add_argument("--flags", action="store_true", help="Noto flags")
    pre_parser.add_argument("--material", action="store_true", help="Material icons")
    pre_parser.add_argument("--fontawesome", action="store_true", help="FontAwesome icons")
    pre_parser.add_argument("--lucide", action="store_true", help="Lucide icons")
    pre_parser.add_argument("--w3c", action="store_true", help="W3C samples")
    pre_parser.add_argument("--resvgtests", action="store_true", help="resvg test suite")
    pre_parser.add_argument("--all", action="store_true", help="All collections")
    pre_parser.add_argument("-j", "--workers", type=int, help="Number of workers")
    pre_parser.add_argument("--no-cairo", action="store_true", help="Skip Cairo rendering")
    pre_parser.add_argument("--no-resvg", action="store_true", help="Skip resvg rendering")
    pre_parser.add_argument("--no-chrome", action="store_true", help="Skip Chrome rendering")
    pre_parser.add_argument("--chrome-backend", choices=["auto", "playwright", "cli"], default="playwright",
                            help="Chrome rendering backend (default: playwright)")
    pre_parser.add_argument("--size", type=int, help="Override render size for all references (e.g., 400)")
    pre_parser.add_argument("--chrome-serial", action="store_true", help="Run Chrome prerendering with a single worker to prevent UI popups")
    pre_parser.add_argument("--force", action="store_true", help="Re-render even if output file exists")

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare VectorStag against references")
    cmp_parser.add_argument("--emojis", action="store_true", help="Noto emojis")
    cmp_parser.add_argument("--flags", action="store_true", help="Noto flags")
    cmp_parser.add_argument("--material", action="store_true", help="Material icons")
    cmp_parser.add_argument("--fontawesome", action="store_true", help="FontAwesome icons")
    cmp_parser.add_argument("--lucide", action="store_true", help="Lucide icons")
    cmp_parser.add_argument("--w3c", action="store_true", help="W3C samples")
    cmp_parser.add_argument("--resvgtests", action="store_true", help="resvg test suite")
    cmp_parser.add_argument("--all", action="store_true", help="All collections")
    cmp_parser.add_argument("-j", "--workers", type=int, help="Number of workers")
    cmp_parser.add_argument("--save", action="store_true", help="Save comparison grid PNGs")
    cmp_parser.add_argument("--limit", type=int, help="Limit number of files")

    # List command
    subparsers.add_parser("list", help="List available collections")

    # Matrix command
    mat_parser = subparsers.add_parser("matrix", help="Pairwise similarity across VectorStag/resvg/Cairo/Chrome")
    mat_parser.add_argument("--emojis", action="store_true", help="Noto emojis")
    mat_parser.add_argument("--flags", action="store_true", help="Noto flags")
    mat_parser.add_argument("--material", action="store_true", help="Material icons")
    mat_parser.add_argument("--fontawesome", action="store_true", help="FontAwesome icons")
    mat_parser.add_argument("--lucide", action="store_true", help="Lucide icons")
    mat_parser.add_argument("--w3c", action="store_true", help="W3C samples")
    mat_parser.add_argument("--resvgtests", action="store_true", help="resvg test suite")
    mat_parser.add_argument("--all", action="store_true", help="All collections")
    mat_parser.add_argument("-j", "--workers", type=int, help="Number of workers")
    mat_parser.add_argument("--limit", type=int, help="Limit number of files")
    mat_parser.add_argument("--save", action="store_true", help="Save 3x4 grids for visual review")
    mat_parser.add_argument("--save-top", type=int, help="Save top-N worst cases (by min pairwise similarity)")
    mat_parser.add_argument("--save-all", action="store_true", help="Save grids for all files (careful: large)")
    mat_parser.add_argument("--save-dir", type=str, help="Custom output directory for saved grids")
    mat_parser.add_argument("--labels-off", action="store_true", help="Disable labels/percentages on grids")

    args = parser.parse_args()

    if args.command == "prerender":
        cmd_prerender(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "matrix":
        cmd_matrix(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
