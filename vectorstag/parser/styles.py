"""SVG style definitions and parsing."""

from dataclasses import dataclass
from typing import Optional, Union


# Sentinel value to distinguish "fill not set" from "fill=none"
FILL_NOT_SET = object()


@dataclass
class Style:
    """Style attributes for an SVG element."""
    fill: Optional[Union[str, tuple[int, int, int, int], object]] = FILL_NOT_SET
    fill_opacity: float = 1.0
    stroke: Optional[Union[str, tuple[int, int, int, int]]] = None
    stroke_width: float = 1.0
    stroke_opacity: float = 1.0
    stroke_linecap: str = "butt"  # butt, round, square
    stroke_linejoin: str = "miter"  # miter, round, bevel
    stroke_miterlimit: float = 4.0
    stroke_dasharray: Optional[list[float]] = None  # Dash pattern [dash, gap, ...]
    opacity: float = 1.0
    fill_rule: str = "nonzero"  # nonzero, evenodd
    filter_id: Optional[str] = None  # Filter reference
    display: str = "inline"  # none, inline, block, etc.
    visibility: str = "visible"  # visible, hidden, collapse
