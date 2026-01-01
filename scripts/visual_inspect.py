#!/usr/bin/env python3
"""
Visual inspection tool for VectorStag SVG rendering.

Renders sample SVGs from all categories to a temp directory for visual review.

Usage:
    python scripts/visual_inspect.py
    python scripts/visual_inspect.py --count 5
    python scripts/visual_inspect.py --categories emojis flags
"""

import argparse
import random
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstag import SVGRenderer
from PIL import Image


# Collection configurations
COLLECTIONS = {
    "emojis": {
        "dir": Path("SciStagEssentialData/images/noto/emojis/svg"),
        "size": 400,
        "description": "Noto Color Emojis",
    },
    "flags": {
        "dir": Path("SciStagEssentialData/images/noto/flags/svg"),
        "size": 400,
        "description": "Noto Flags",
    },
    "material": {
        "dir": Path("advanced_svg/material"),
        "size": 256,
        "description": "Material Design Icons",
    },
    "fontawesome": {
        "dir": Path("advanced_svg/fontawesome"),
        "size": 256,
        "description": "FontAwesome Icons",
    },
    "lucide": {
        "dir": Path("advanced_svg/lucide"),
        "size": 256,
        "description": "Lucide Icons",
    },
    "w3c": {
        "dir": Path("samples/svg"),
        "size": 400,
        "description": "W3C SVG Test Suite",
    },
}


def get_svg_files(collection_dir: Path, count: int = 3) -> list:
    """Get random sample of SVG files from a collection."""
    if not collection_dir.exists():
        return []

    svg_files = list(collection_dir.glob("*.svg"))
    if not svg_files:
        # Try recursive search
        svg_files = list(collection_dir.rglob("*.svg"))

    if len(svg_files) <= count:
        return sorted(svg_files)

    # Pick random samples, but use consistent seed for reproducibility
    random.seed(42)
    return sorted(random.sample(svg_files, count))


def render_svg(svg_path: Path, output_path: Path, size: int) -> bool:
    """Render an SVG file to PNG."""
    try:
        # SVGRenderer takes background in constructor
        renderer = SVGRenderer(background=(255, 255, 255, 255), antialias=4)
        img = renderer.render_file(str(svg_path), width=size, height=size)

        if img is None:
            print(f"  ERROR rendering {svg_path.name}: render returned None")
            return False

        img.save(output_path)
        return True
    except Exception as e:
        print(f"  ERROR rendering {svg_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Visual inspection of VectorStag SVG rendering")
    parser.add_argument("--count", "-n", type=int, default=3,
                        help="Number of samples per category (default: 3)")
    parser.add_argument("--categories", "-c", nargs="+",
                        choices=list(COLLECTIONS.keys()) + ["all"],
                        default=["all"],
                        help="Categories to render (default: all)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: temp directory)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="vectorstag_inspect_"))

    print(f"\n{'='*60}")
    print(f"VectorStag Visual Inspection")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per category: {args.count}")
    print()

    # Determine which categories to process
    if "all" in args.categories:
        categories = list(COLLECTIONS.keys())
    else:
        categories = args.categories

    random.seed(args.seed)
    total_rendered = 0
    total_failed = 0

    for cat_name in categories:
        config = COLLECTIONS[cat_name]
        cat_dir = config["dir"]
        size = config["size"]

        print(f"\n{config['description']} ({cat_name})")
        print("-" * 40)

        if not cat_dir.exists():
            print(f"  Directory not found: {cat_dir}")
            continue

        svg_files = get_svg_files(cat_dir, args.count)
        if not svg_files:
            print(f"  No SVG files found in {cat_dir}")
            continue

        # Create category subdirectory
        cat_output_dir = output_dir / cat_name
        cat_output_dir.mkdir(exist_ok=True)

        for svg_path in svg_files:
            output_path = cat_output_dir / f"{svg_path.stem}.png"
            success = render_svg(svg_path, output_path, size)

            if success:
                print(f"  {svg_path.name} -> {output_path.name}")
                total_rendered += 1
            else:
                total_failed += 1

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Rendered: {total_rendered} files")
    print(f"Failed: {total_failed} files")
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("To view the rendered images:")
    print(f"  ls -la {output_dir}/*/")
    print()

    # Also create an index HTML for easy viewing
    create_index_html(output_dir, categories)
    print(f"Open in browser: file://{output_dir}/index.html")
    print()

    return 0 if total_failed == 0 else 1


def create_index_html(output_dir: Path, categories: list):
    """Create an HTML index for easy viewing."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>VectorStag Visual Inspection</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .gallery { display: flex; flex-wrap: wrap; gap: 20px; }
        .item { background: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
        .item img { max-width: 300px; max-height: 300px; border: 1px solid #eee;
                    background: repeating-conic-gradient(#f0f0f0 0% 25%, white 0% 50%)
                    50% / 20px 20px; }
        .item p { margin: 10px 0 0; font-size: 12px; color: #666; word-break: break-all; }
    </style>
</head>
<body>
    <h1>VectorStag Visual Inspection</h1>
"""

    for cat_name in categories:
        cat_dir = output_dir / cat_name
        if not cat_dir.exists():
            continue

        png_files = sorted(cat_dir.glob("*.png"))
        if not png_files:
            continue

        config = COLLECTIONS.get(cat_name, {"description": cat_name})
        html += f'    <h2>{config.get("description", cat_name)}</h2>\n'
        html += '    <div class="gallery">\n'

        for png_path in png_files:
            rel_path = f"{cat_name}/{png_path.name}"
            html += f'''        <div class="item">
            <img src="{rel_path}" alt="{png_path.stem}">
            <p>{png_path.stem}</p>
        </div>
'''

        html += '    </div>\n'

    html += """</body>
</html>
"""

    with open(output_dir / "index.html", 'w') as f:
        f.write(html)


if __name__ == "__main__":
    sys.exit(main())
