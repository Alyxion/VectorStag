"""High-performance SVG renderer using the full Rust pipeline.

This module provides a drop-in replacement for SVGRenderer that uses
resvg for rendering entirely in Rust, eliminating Python/Rust boundary
crossing overhead.
"""

from PIL import Image
import numpy as np
import vectorstag_rust


class RustSVGRenderer:
    """High-performance SVG renderer using resvg in Rust.

    This renderer processes SVGs entirely in Rust, providing significant
    speedups compared to the Python-based SVGRenderer.
    """

    def __init__(
        self,
        scale: float = 1.0,
        background: tuple = (255, 255, 255, 255),
        antialias: int = 4,
    ):
        """Initialize the Rust SVG renderer.

        Args:
            scale: Scale factor for rendering
            background: Background color as (r, g, b, a) tuple
            antialias: Antialiasing factor (default 4x supersampling)
        """
        self._renderer = vectorstag_rust.SvgRenderer()
        self.scale = scale
        self.background = background
        self.antialias = antialias

    def render(
        self,
        svg_content: str,
        width: int = None,
        height: int = None,
    ) -> Image.Image:
        """Render SVG content to a PIL Image.

        Args:
            svg_content: SVG string content
            width: Output width (optional)
            height: Output height (optional)

        Returns:
            PIL Image in RGBA mode
        """
        arr = self._renderer.render(
            svg_content,
            width=width,
            height=height,
            scale=self.scale,
            background=self.background,
            antialias=self.antialias,
        )
        return Image.fromarray(arr, "RGBA")

    def render_file(
        self,
        file_path: str,
        width: int = None,
        height: int = None,
    ) -> Image.Image:
        """Render SVG file to a PIL Image.

        Args:
            file_path: Path to SVG file
            width: Output width (optional)
            height: Output height (optional)

        Returns:
            PIL Image in RGBA mode
        """
        with open(file_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        return self.render(svg_content, width, height)

    def render_to_array(
        self,
        svg_content: str,
        width: int = None,
        height: int = None,
    ) -> np.ndarray:
        """Render SVG content to a numpy array.

        Args:
            svg_content: SVG string content
            width: Output width (optional)
            height: Output height (optional)

        Returns:
            RGBA numpy array of shape (height, width, 4)
        """
        return self._renderer.render(
            svg_content,
            width=width,
            height=height,
            scale=self.scale,
            background=self.background,
            antialias=self.antialias,
        )

    def render_file_to_array(
        self,
        file_path: str,
        width: int = None,
        height: int = None,
    ) -> np.ndarray:
        """Render SVG file to a numpy array.

        Args:
            file_path: Path to SVG file
            width: Output width (optional)
            height: Output height (optional)

        Returns:
            RGBA numpy array of shape (height, width, 4)
        """
        return self._renderer.render_file(
            file_path,
            width=width,
            height=height,
            scale=self.scale,
            background=self.background,
            antialias=self.antialias,
        )
