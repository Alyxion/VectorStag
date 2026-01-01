"""SVG element dataclasses."""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .styles import Style
    from ..core.transforms import Transform


@dataclass
class SVGElement:
    """Base class for SVG elements."""
    tag: str
    style: "Style"
    transform: "Transform"
    children: list["SVGElement"] = field(default_factory=list)
    clip_path_id: Optional[str] = None
    mask_id: Optional[str] = None
    filter_id: Optional[str] = None


@dataclass
class RectElement(SVGElement):
    """Rectangle element."""
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    rx: float = 0
    ry: float = 0


@dataclass
class CircleElement(SVGElement):
    """Circle element."""
    cx: float = 0
    cy: float = 0
    r: float = 0


@dataclass
class EllipseElement(SVGElement):
    """Ellipse element."""
    cx: float = 0
    cy: float = 0
    rx: float = 0
    ry: float = 0


@dataclass
class LineElement(SVGElement):
    """Line element."""
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0


@dataclass
class PolylineElement(SVGElement):
    """Polyline element."""
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PolygonElement(SVGElement):
    """Polygon element."""
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PathElement(SVGElement):
    """Path element with parsed commands."""
    commands: list[tuple] = field(default_factory=list)


@dataclass
class GroupElement(SVGElement):
    """Group element (g)."""
    pass


@dataclass
class TextElement(SVGElement):
    """Text element."""
    x: float = 0
    y: float = 0
    text: str = ""
    font_family: str = "Arial"
    font_size: float = 16
    text_anchor: str = "start"  # start, middle, end
    # textPath support
    text_path_href: str = None  # Reference to path element for textPath
    text_path_start_offset: float = 0  # Starting offset along path (0-1 for percentage)
    text_path_data: str = None  # Direct path data (SVG 2 path attribute)


@dataclass
class ImageElement(SVGElement):
    """Image element for embedding raster images."""
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    href: str = ""  # Data URL, external reference, or memory:name
    preserveAspectRatio: str = "xMidYMid meet"
    base_path: str = ""  # Base path for resolving relative URLs


@dataclass
class ClipPath:
    """Clip path definition."""
    id: str
    elements: list[SVGElement] = field(default_factory=list)
    clip_path_id: Optional[str] = None  # For nested clip paths (intersection)
    units: str = "userSpaceOnUse"  # clipPathUnits: userSpaceOnUse or objectBoundingBox


@dataclass
class Mask:
    """Mask definition."""
    id: str
    elements: list[SVGElement] = field(default_factory=list)
    x: float = -0.1  # Default mask region (in objectBoundingBox units)
    y: float = -0.1
    width: float = 1.2
    height: float = 1.2
    mask_units: str = "objectBoundingBox"  # or "userSpaceOnUse"
    mask_content_units: str = "userSpaceOnUse"  # or "objectBoundingBox"


@dataclass
class SVGDocument:
    """Parsed SVG document."""
    width: float
    height: float
    viewBox: Optional[tuple[float, float, float, float]] = None
    elements: list[SVGElement] = field(default_factory=list)
    gradients: dict = field(default_factory=dict)
    patterns: dict = field(default_factory=dict)
    clip_paths: dict[str, ClipPath] = field(default_factory=dict)
    masks: dict[str, Mask] = field(default_factory=dict)
    filters: dict = field(default_factory=dict)
    elements_by_id: dict[str, SVGElement] = field(default_factory=dict)
    path_data_by_id: dict[str, str] = field(default_factory=dict)
    preserve_aspect_ratio: str = "xMidYMid"
