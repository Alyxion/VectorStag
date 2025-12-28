"""VectorStag - A pure Python SVG renderer using Pillow."""

__version__ = "0.1.0"

from .renderer import SVGRenderer
from .parser import SVGParser

__all__ = ["SVGRenderer", "SVGParser"]
