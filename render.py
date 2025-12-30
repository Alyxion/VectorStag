#!/usr/bin/env python3
"""
Simple SVG rendering utility for VectorStag.

Usage:
    # Render to PNG
    python render.py input.svg output.png

    # Render at specific size
    python render.py input.svg output.png --width 800 --height 600

    # Render with options
    python render.py input.svg output.png --antialias 8 --background white

    # Render to stdout (for piping)
    python render.py input.svg - | display
"""

import argparse
import sys
from pathlib import Path

from vectorstag import SVGRenderer


def parse_color(color_str: str) -> tuple:
    """Parse color string to RGBA tuple."""
    color_str = color_str.lower().strip()

    # Named colors
    named = {
        'transparent': (0, 0, 0, 0),
        'white': (255, 255, 255, 255),
        'black': (0, 0, 0, 255),
        'red': (255, 0, 0, 255),
        'green': (0, 255, 0, 255),
        'blue': (0, 0, 255, 255),
    }

    if color_str in named:
        return named[color_str]

    # Hex color
    if color_str.startswith('#'):
        hex_str = color_str[1:]
        if len(hex_str) == 3:
            hex_str = ''.join(c * 2 for c in hex_str)
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return (r, g, b, 255)
        if len(hex_str) == 8:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            a = int(hex_str[6:8], 16)
            return (r, g, b, a)

    # Default to transparent
    return (0, 0, 0, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Render SVG to PNG using VectorStag",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input", help="Input SVG file")
    parser.add_argument("output", help="Output PNG file (use '-' for stdout)")

    parser.add_argument("-w", "--width", type=int, help="Output width")
    parser.add_argument("-h", "--height", type=int, dest="height", help="Output height")
    parser.add_argument("-s", "--size", type=int, help="Output size (square)")

    parser.add_argument("-a", "--antialias", type=int, default=4,
                        help="Antialiasing level (default: 4)")
    parser.add_argument("-b", "--background", type=str, default="transparent",
                        help="Background color (default: transparent)")

    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output messages")

    args = parser.parse_args()

    # Parse input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine output dimensions
    width = args.width
    height = args.height

    if args.size:
        width = args.size
        height = args.size

    # Parse background color
    background = parse_color(args.background)

    # Render
    try:
        renderer = SVGRenderer(background=background, antialias=args.antialias)

        if width and height:
            img = renderer.render_file(str(input_path), width, height)
        elif width:
            img = renderer.render_file(str(input_path), width=width)
        elif height:
            img = renderer.render_file(str(input_path), height=height)
        else:
            img = renderer.render_file(str(input_path))

        if img is None:
            print(f"Error: Failed to render {args.input}", file=sys.stderr)
            sys.exit(1)

        # Save output
        if args.output == '-':
            img.save(sys.stdout.buffer, format='PNG')
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)

            if not args.quiet:
                print(f"Rendered {input_path.name} -> {output_path} ({img.width}x{img.height})")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
