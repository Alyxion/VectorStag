"""SVG Parser - Parse SVG documents into a renderable structure."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Union
import math


def _safe_float(val: str, default: float = 0.0) -> float:
    """Parse a float value, handling percentages and units."""
    if not val:
        return default
    val = val.strip()
    try:
        if val.endswith('%'):
            return float(val[:-1]) / 100.0
        # Strip px or other units
        for unit in ['px', 'em', 'pt', 'mm', 'cm', 'in']:
            if val.endswith(unit):
                val = val[:-len(unit)]
                break
        return float(val)
    except (ValueError, TypeError):
        return default


@dataclass
class Transform:
    """Represents a 2D affine transformation matrix."""
    a: float = 1.0  # scale x
    b: float = 0.0  # skew y
    c: float = 0.0  # skew x
    d: float = 1.0  # scale y
    e: float = 0.0  # translate x
    f: float = 0.0  # translate y

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """Apply transformation to a point."""
        return (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f
        )

    def multiply(self, other: "Transform") -> "Transform":
        """Multiply this transform by another (self * other)."""
        return Transform(
            a=self.a * other.a + self.c * other.b,
            b=self.b * other.a + self.d * other.b,
            c=self.a * other.c + self.c * other.d,
            d=self.b * other.c + self.d * other.d,
            e=self.a * other.e + self.c * other.f + self.e,
            f=self.b * other.e + self.d * other.f + self.f
        )

    @classmethod
    def identity(cls) -> "Transform":
        return cls()

    @classmethod
    def translate(cls, tx: float, ty: float = 0) -> "Transform":
        return cls(e=tx, f=ty)

    @classmethod
    def scale(cls, sx: float, sy: Optional[float] = None) -> "Transform":
        if sy is None:
            sy = sx
        return cls(a=sx, d=sy)

    @classmethod
    def rotate(cls, angle: float, cx: float = 0, cy: float = 0) -> "Transform":
        """Rotate by angle (degrees) around point (cx, cy)."""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        if cx == 0 and cy == 0:
            return cls(a=cos_a, b=sin_a, c=-sin_a, d=cos_a)
        # Translate to origin, rotate, translate back
        t1 = cls.translate(-cx, -cy)
        r = cls(a=cos_a, b=sin_a, c=-sin_a, d=cos_a)
        t2 = cls.translate(cx, cy)
        return t2.multiply(r.multiply(t1))

    @classmethod
    def skewX(cls, angle: float) -> "Transform":
        return cls(c=math.tan(math.radians(angle)))

    @classmethod
    def skewY(cls, angle: float) -> "Transform":
        return cls(b=math.tan(math.radians(angle)))

    @classmethod
    def matrix(cls, a: float, b: float, c: float, d: float, e: float, f: float) -> "Transform":
        return cls(a=a, b=b, c=c, d=d, e=e, f=f)


@dataclass
class GradientStop:
    """A stop in a gradient."""
    offset: float  # 0.0 to 1.0
    color: tuple[int, int, int, int]  # RGBA


@dataclass
class LinearGradient:
    """Linear gradient definition."""
    id: str
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 0.0
    stops: list[GradientStop] = field(default_factory=list)
    units: str = "objectBoundingBox"
    transform: Optional[Transform] = None
    href: Optional[str] = None  # Reference to another gradient
    spread_method: str = "pad"  # pad, reflect, repeat


@dataclass
class RadialGradient:
    """Radial gradient definition."""
    id: str
    cx: float = 0.5
    cy: float = 0.5
    r: float = 0.5
    fx: Optional[float] = None
    fy: Optional[float] = None
    fr: float = 0.0  # focal radius - gradient starts at fr distance from focal point
    stops: list[GradientStop] = field(default_factory=list)
    units: str = "objectBoundingBox"
    transform: Optional[Transform] = None
    href: Optional[str] = None
    spread_method: str = "pad"  # pad, reflect, repeat


@dataclass
class Pattern:
    """Pattern definition for tiled fills."""
    id: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    pattern_units: str = "objectBoundingBox"  # objectBoundingBox or userSpaceOnUse
    pattern_content_units: str = "userSpaceOnUse"  # objectBoundingBox or userSpaceOnUse
    transform: Optional[Transform] = None
    href: Optional[str] = None  # Reference to another pattern
    viewbox: Optional[tuple[float, float, float, float]] = None
    elements: list = field(default_factory=list)  # Child elements to render


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
    stroke_dasharray: Optional[list[float]] = None  # Dash pattern [dash, gap, dash, gap, ...]
    opacity: float = 1.0
    fill_rule: str = "nonzero"  # nonzero, evenodd
    filter_id: Optional[str] = None  # Filter reference
    display: str = "inline"  # none, inline, block, etc.
    visibility: str = "visible"  # visible, hidden, collapse


@dataclass
class SVGElement:
    """Base class for SVG elements."""
    tag: str
    style: Style
    transform: Transform
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
    href: str = ""  # Data URL or external reference
    preserveAspectRatio: str = "xMidYMid meet"
    base_path: str = ""  # Base path for resolving relative URLs


@dataclass
class ClipPath:
    """Clip path definition."""
    id: str
    elements: list[SVGElement] = field(default_factory=list)
    clip_path_id: Optional[str] = None  # For nested clip paths (intersection)


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
class FilterPrimitive:
    """Base class for filter primitives."""
    input1: Optional[str] = None  # 'in' attribute
    input2: Optional[str] = None  # 'in2' attribute
    result: Optional[str] = None  # 'result' attribute
    # Subregion (x, y, width, height can be set per-primitive)
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass
class FeGaussianBlur(FilterPrimitive):
    """Gaussian blur filter primitive."""
    std_deviation_x: float = 0.0
    std_deviation_y: float = 0.0


@dataclass
class FeOffset(FilterPrimitive):
    """Offset filter primitive."""
    dx: float = 0.0
    dy: float = 0.0


@dataclass
class FeFlood(FilterPrimitive):
    """Flood fill filter primitive."""
    flood_color: tuple[int, int, int, int] = (0, 0, 0, 255)  # RGBA
    flood_opacity: float = 1.0


@dataclass
class FeBlend(FilterPrimitive):
    """Blend filter primitive."""
    mode: str = "normal"  # normal, multiply, screen, darken, lighten, etc.


@dataclass
class FeComposite(FilterPrimitive):
    """Composite filter primitive."""
    operator: str = "over"  # over, in, out, atop, xor, arithmetic
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0


@dataclass
class FeMergeNode:
    """A node in a merge operation."""
    input1: str = "SourceGraphic"


@dataclass
class FeMerge(FilterPrimitive):
    """Merge filter primitive."""
    nodes: list[FeMergeNode] = field(default_factory=list)


@dataclass
class FeColorMatrix(FilterPrimitive):
    """Color matrix filter primitive."""
    type: str = "matrix"  # matrix, saturate, hueRotate, luminanceToAlpha
    values: list[float] = field(default_factory=list)


@dataclass
class FeComponentTransferFunc:
    """Transfer function for a single channel."""
    type: str = "identity"  # identity, table, discrete, linear, gamma
    table_values: list[float] = field(default_factory=list)
    slope: float = 1.0
    intercept: float = 0.0
    amplitude: float = 1.0
    exponent: float = 1.0
    offset: float = 0.0


@dataclass
class FeComponentTransfer(FilterPrimitive):
    """Component transfer filter primitive."""
    func_r: FeComponentTransferFunc = field(default_factory=FeComponentTransferFunc)
    func_g: FeComponentTransferFunc = field(default_factory=FeComponentTransferFunc)
    func_b: FeComponentTransferFunc = field(default_factory=FeComponentTransferFunc)
    func_a: FeComponentTransferFunc = field(default_factory=FeComponentTransferFunc)


@dataclass
class FeMorphology(FilterPrimitive):
    """Morphology filter primitive (erode/dilate)."""
    operator: str = "erode"  # erode, dilate
    radius_x: float = 0.0
    radius_y: float = 0.0


@dataclass
class FeConvolveMatrix(FilterPrimitive):
    """Convolution matrix filter primitive."""
    order_x: int = 3
    order_y: int = 3
    kernel_matrix: list[float] = field(default_factory=list)
    divisor: Optional[float] = None  # Auto-calculated if not specified
    bias: float = 0.0
    target_x: Optional[int] = None  # Default: floor(orderX / 2)
    target_y: Optional[int] = None  # Default: floor(orderY / 2)
    edge_mode: str = "duplicate"  # duplicate, wrap, none
    preserve_alpha: bool = False


@dataclass
class FeTurbulence(FilterPrimitive):
    """Turbulence filter primitive (Perlin noise)."""
    type: str = "turbulence"  # turbulence, fractalNoise
    base_frequency_x: float = 0.0
    base_frequency_y: float = 0.0
    num_octaves: int = 1
    seed: int = 0
    stitch_tiles: str = "noStitch"  # noStitch, stitch


@dataclass
class FeDisplacementMap(FilterPrimitive):
    """Displacement map filter primitive."""
    scale: float = 0.0
    x_channel_selector: str = "A"  # R, G, B, A
    y_channel_selector: str = "A"  # R, G, B, A


@dataclass
class FeImage(FilterPrimitive):
    """Image filter primitive."""
    href: str = ""
    preserveAspectRatio: str = "xMidYMid meet"


@dataclass
class FeTile(FilterPrimitive):
    """Tile filter primitive."""
    pass  # Uses inherited attributes


@dataclass
class LightSource:
    """Base class for light sources."""
    pass


@dataclass
class FeDistantLight(LightSource):
    """Distant light source."""
    azimuth: float = 0.0
    elevation: float = 0.0


@dataclass
class FePointLight(LightSource):
    """Point light source."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class FeSpotLight(LightSource):
    """Spotlight source."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    points_at_x: float = 0.0
    points_at_y: float = 0.0
    points_at_z: float = 0.0
    specular_exponent: float = 1.0
    limiting_cone_angle: Optional[float] = None


@dataclass
class FeDiffuseLighting(FilterPrimitive):
    """Diffuse lighting filter primitive."""
    surface_scale: float = 1.0
    diffuse_constant: float = 1.0
    lighting_color: tuple[int, int, int] = (255, 255, 255)
    light_source: Optional[LightSource] = None


@dataclass
class FeSpecularLighting(FilterPrimitive):
    """Specular lighting filter primitive."""
    surface_scale: float = 1.0
    specular_constant: float = 1.0
    specular_exponent: float = 1.0
    lighting_color: tuple[int, int, int] = (255, 255, 255)
    light_source: Optional[LightSource] = None


@dataclass
class FeDropShadow(FilterPrimitive):
    """Drop shadow filter primitive (SVG2)."""
    dx: float = 2.0
    dy: float = 2.0
    std_deviation_x: float = 0.0
    std_deviation_y: float = 0.0
    flood_color: tuple[int, int, int, int] = (0, 0, 0, 255)
    flood_opacity: float = 1.0


@dataclass
class Filter:
    """Complete filter definition with chain of primitives."""
    id: str
    primitives: list[FilterPrimitive] = field(default_factory=list)
    # Filter region
    x: float = -0.1  # Default: -10% of element bbox
    y: float = -0.1
    width: float = 1.2  # Default: 120% of element bbox
    height: float = 1.2
    filter_units: str = "objectBoundingBox"  # or "userSpaceOnUse"
    primitive_units: str = "userSpaceOnUse"  # or "objectBoundingBox"
    color_interpolation_filters: str = "linearRGB"  # linearRGB or sRGB


@dataclass
class SVGDocument:
    """Parsed SVG document."""
    width: float
    height: float
    viewBox: Optional[tuple[float, float, float, float]] = None
    elements: list[SVGElement] = field(default_factory=list)
    gradients: dict[str, Union[LinearGradient, RadialGradient]] = field(default_factory=dict)
    patterns: dict[str, Pattern] = field(default_factory=dict)
    clip_paths: dict[str, ClipPath] = field(default_factory=dict)
    masks: dict[str, Mask] = field(default_factory=dict)
    filters: dict[str, Filter] = field(default_factory=dict)
    elements_by_id: dict[str, SVGElement] = field(default_factory=dict)  # For feImage references
    path_data_by_id: dict[str, str] = field(default_factory=dict)  # For textPath references
    preserve_aspect_ratio: str = "xMidYMid"  # SVG default


class SVGParser:
    """Parse SVG documents."""

    # Namespace handling
    SVG_NS = "{http://www.w3.org/2000/svg}"
    XLINK_NS = "{http://www.w3.org/1999/xlink}"

    # Color name to RGB mapping (basic colors)
    COLORS = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "brown": (165, 42, 42),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "lime": (0, 255, 0),
        "navy": (0, 0, 128),
        "teal": (0, 128, 128),
        "olive": (128, 128, 0),
        "maroon": (128, 0, 0),
        "silver": (192, 192, 192),
        "aqua": (0, 255, 255),
        "fuchsia": (255, 0, 255),
        "none": None,
        "transparent": (0, 0, 0, 0),
        # Extended SVG named colors
        "firebrick": (178, 34, 34),
        "darkred": (139, 0, 0),
        "crimson": (220, 20, 60),
        "indianred": (205, 92, 92),
        "lightcoral": (240, 128, 128),
        "salmon": (250, 128, 114),
        "darksalmon": (233, 150, 122),
        "lightsalmon": (255, 160, 122),
        "orangered": (255, 69, 0),
        "tomato": (255, 99, 71),
        "coral": (255, 127, 80),
        "darkorange": (255, 140, 0),
        "gold": (255, 215, 0),
        "khaki": (240, 230, 140),
        "darkkhaki": (189, 183, 107),
        "lavender": (230, 230, 250),
        "violet": (238, 130, 238),
        "plum": (221, 160, 221),
        "orchid": (218, 112, 214),
        "mediumorchid": (186, 85, 211),
        "darkorchid": (153, 50, 204),
        "darkviolet": (148, 0, 211),
        "blueviolet": (138, 43, 226),
        "indigo": (75, 0, 130),
        "darkslateblue": (72, 61, 139),
        "slateblue": (106, 90, 205),
        "mediumslateblue": (123, 104, 238),
        "greenyellow": (173, 255, 47),
        "chartreuse": (127, 255, 0),
        "lawngreen": (124, 252, 0),
        "limegreen": (50, 205, 50),
        "palegreen": (152, 251, 152),
        "lightgreen": (144, 238, 144),
        "mediumspringgreen": (0, 250, 154),
        "springgreen": (0, 255, 127),
        "mediumseagreen": (60, 179, 113),
        "seagreen": (46, 139, 87),
        "forestgreen": (34, 139, 34),
        "darkgreen": (0, 100, 0),
        "yellowgreen": (154, 205, 50),
        "olivedrab": (107, 142, 35),
        "darkolivegreen": (85, 107, 47),
        "mediumaquamarine": (102, 205, 170),
        "darkseagreen": (143, 188, 143),
        "lightseagreen": (32, 178, 170),
        "darkcyan": (0, 139, 139),
        "lightcyan": (224, 255, 255),
        "paleturquoise": (175, 238, 238),
        "aquamarine": (127, 255, 212),
        "turquoise": (64, 224, 208),
        "mediumturquoise": (72, 209, 204),
        "darkturquoise": (0, 206, 209),
        "cadetblue": (95, 158, 160),
        "steelblue": (70, 130, 180),
        "lightsteelblue": (176, 196, 222),
        "powderblue": (176, 224, 230),
        "lightblue": (173, 216, 230),
        "skyblue": (135, 206, 235),
        "lightskyblue": (135, 206, 250),
        "deepskyblue": (0, 191, 255),
        "dodgerblue": (30, 144, 255),
        "cornflowerblue": (100, 149, 237),
        "royalblue": (65, 105, 225),
        "mediumblue": (0, 0, 205),
        "darkblue": (0, 0, 139),
        "midnightblue": (25, 25, 112),
        "cornsilk": (255, 248, 220),
        "blanchedalmond": (255, 235, 205),
        "bisque": (255, 228, 196),
        "navajowhite": (255, 222, 173),
        "wheat": (245, 222, 179),
        "burlywood": (222, 184, 135),
        "tan": (210, 180, 140),
        "rosybrown": (188, 143, 143),
        "sandybrown": (244, 164, 96),
        "goldenrod": (218, 165, 32),
        "darkgoldenrod": (184, 134, 11),
        "peru": (205, 133, 63),
        "chocolate": (210, 105, 30),
        "saddlebrown": (139, 69, 19),
        "sienna": (160, 82, 45),
        "snow": (255, 250, 250),
        "honeydew": (240, 255, 240),
        "mintcream": (245, 255, 250),
        "azure": (240, 255, 255),
        "aliceblue": (240, 248, 255),
        "ghostwhite": (248, 248, 255),
        "whitesmoke": (245, 245, 245),
        "seashell": (255, 245, 238),
        "beige": (245, 245, 220),
        "oldlace": (253, 245, 230),
        "floralwhite": (255, 250, 240),
        "ivory": (255, 255, 240),
        "antiquewhite": (250, 235, 215),
        "linen": (250, 240, 230),
        "lavenderblush": (255, 240, 245),
        "mistyrose": (255, 228, 225),
        "gainsboro": (220, 220, 220),
        "lightgray": (211, 211, 211),
        "lightgrey": (211, 211, 211),
        "darkgray": (169, 169, 169),
        "darkgrey": (169, 169, 169),
        "dimgray": (105, 105, 105),
        "dimgrey": (105, 105, 105),
        "lightslategray": (119, 136, 153),
        "lightslategrey": (119, 136, 153),
        "slategray": (112, 128, 144),
        "slategrey": (112, 128, 144),
        "darkslategray": (47, 79, 79),
        "darkslategrey": (47, 79, 79),
        "hotpink": (255, 105, 180),
        "deeppink": (255, 20, 147),
        "mediumvioletred": (199, 21, 133),
        "palevioletred": (219, 112, 147),
    }

    # Maximum recursion depth for parsing (prevents infinite loops)
    MAX_PARSE_DEPTH = 100

    def __init__(self):
        self.gradients: dict[str, Union[LinearGradient, RadialGradient]] = {}
        self.clip_paths: dict[str, ClipPath] = {}
        self.masks: dict[str, Mask] = {}
        self.defs: dict[str, ET.Element] = {}
        self.default_width = 300
        self.default_height = 150
        # ViewBox dimensions for percentage resolution
        self.viewbox_width = 0
        self.viewbox_height = 0
        # CSS classes from <style> blocks
        self.css_classes: dict[str, dict[str, str]] = {}
        # Track <use> references being parsed to detect circular references
        self._use_stack: set[str] = set()

    def parse(self, svg_content: str) -> SVGDocument:
        """Parse SVG content string into SVGDocument."""
        # Remove any XML declaration issues
        svg_content = svg_content.strip()

        # Parse XML
        root = ET.fromstring(svg_content)

        # Parse viewBox first (needed for percentage dimensions)
        viewBox = None
        viewbox_str = root.get("viewBox")
        if viewbox_str:
            parts = re.split(r"[\s,]+", viewbox_str.strip())
            if len(parts) == 4:
                viewBox = tuple(float(p) for p in parts)
                # Store viewBox dimensions for percentage resolution
                self.viewbox_width = viewBox[2]
                self.viewbox_height = viewBox[3]

        # Get dimensions - use viewBox for percentage reference
        width_str = root.get("width", str(self.default_width))
        height_str = root.get("height", str(self.default_height))

        # Handle percentage dimensions (e.g., "100%")
        if "%" in width_str and viewBox:
            width = self._parse_length(width_str, viewBox[2])
        else:
            width = self._parse_length(width_str)
            # If width not specified but viewBox exists, use viewBox width
            if not root.get("width") and viewBox:
                width = viewBox[2]

        if "%" in height_str and viewBox:
            height = self._parse_length(height_str, viewBox[3])
        else:
            height = self._parse_length(height_str)
            # If height not specified but viewBox exists, use viewBox height
            if not root.get("height") and viewBox:
                height = viewBox[3]

        # Set viewbox dimensions for percentage resolution if not already set
        if not viewBox:
            self.viewbox_width = width
            self.viewbox_height = height

        # Parse preserveAspectRatio attribute
        preserve_aspect_ratio = root.get("preserveAspectRatio", "xMidYMid")
        # Strip optional "meet" or "slice" suffix
        preserve_aspect_ratio = preserve_aspect_ratio.split()[0] if preserve_aspect_ratio else "xMidYMid"

        # Reset state
        self.gradients = {}
        self.patterns = {}
        self.clip_paths = {}
        self.masks = {}
        self.filters = {}
        self.defs = {}
        self.css_classes = {}
        self.css_ids = {}  # ID selector CSS rules
        self.css_types = {}  # Type selector CSS rules
        self.css_attr_selectors = {}  # Attribute selector CSS rules [attr] or [attr=value]
        self.css_rules = []  # Complex CSS rules: (selector, properties, specificity, order)
        self.css_rule_order = 0  # For stable ordering of rules with equal specificity
        self.elements_by_id = {}  # For feImage element references
        self.path_data_by_id = {}  # For textPath references

        # Parse CSS from <style> blocks
        self._parse_css_styles(root)

        # First pass: collect defs
        self._collect_defs(root)

        # Parse clip paths and masks
        self._parse_clip_paths(root)
        self._parse_masks(root)

        # Resolve gradient and pattern references
        self._resolve_gradient_refs()
        self._resolve_pattern_refs()

        # Parse root element style (for inherited properties like stroke-width)
        root_style = self._parse_style(root, Style())

        # Parse elements
        elements = self._parse_children(root, Transform.identity(), root_style)

        # If no viewBox, compute dimensions from content bounding box
        # This matches resvg behavior - explicit width/height without viewBox
        # are treated as hints but actual dimensions come from content
        has_explicit_width = root.get("width") is not None
        has_explicit_height = root.get("height") is not None

        # If no viewBox and missing explicit dimensions, compute from content
        # Match resvg behavior: size to max coordinates, keep content at original position
        if not viewBox and (not has_explicit_width or not has_explicit_height):
            # Compute bounding box from elements
            bbox = self._compute_elements_bbox(elements)
            if bbox:
                min_x, min_y, max_x, max_y = bbox

                # resvg ignores explicit dimensions when there's no viewBox
                # Size document to max coordinates of content
                width = max_x
                height = max_y

                # NO viewBox - keep coordinate system at (0,0)
                # Content at (min_x, min_y) renders at those pixel coordinates

        return SVGDocument(
            width=width,
            height=height,
            viewBox=viewBox,
            elements=elements,
            gradients=self.gradients,
            patterns=self.patterns,
            clip_paths=self.clip_paths,
            masks=self.masks,
            filters=self.filters,
            elements_by_id=self.elements_by_id,
            path_data_by_id=self.path_data_by_id,
            preserve_aspect_ratio=preserve_aspect_ratio
        )

    def parse_file(self, filepath: str) -> SVGDocument:
        """Parse SVG file."""
        import os
        # Store base path for resolving relative URLs
        self.base_path = os.path.dirname(os.path.abspath(filepath))
        with open(filepath, "r", encoding="utf-8") as f:
            return self.parse(f.read())

    def _strip_ns(self, tag: str) -> str:
        """Remove namespace from tag."""
        if tag.startswith("{"):
            return tag.split("}", 1)[1]
        return tag

    def _parse_css_styles(self, root: ET.Element):
        """Parse CSS from <style> blocks."""
        for elem in root.iter():
            tag = self._strip_ns(elem.tag)
            if tag == "style":
                css_text = elem.text or ""
                # Handle CDATA
                css_text = css_text.strip()
                self._parse_css_text(css_text)

    def _parse_css_text(self, css_text: str):
        """Parse CSS text and extract class, ID, type, and attribute rules."""
        # Remove comments
        css_text = re.sub(r'/\*.*?\*/', '', css_text, flags=re.DOTALL)

        # Find all rule blocks: selector { properties }
        # Handle complex selectors including descendant combinators
        pattern = r'([^{]+)\{([^}]*)\}'
        for match in re.finditer(pattern, css_text):
            selector = match.group(1).strip()
            properties_str = match.group(2)

            if not selector:
                continue

            # Parse properties
            properties = {}
            for prop in properties_str.split(';'):
                prop = prop.strip()
                if ':' in prop:
                    key, value = prop.split(':', 1)
                    properties[key.strip()] = value.strip()

            if not properties:
                continue

            # Calculate specificity: (ids, classes+attrs, elements)
            specificity = self._calculate_specificity(selector)

            # Store rule with specificity and order
            self.css_rules.append((selector, properties, specificity, self.css_rule_order))
            self.css_rule_order += 1

            # Also store simple selectors for fast lookup
            if ' ' not in selector and '>' not in selector and '+' not in selector and '~' not in selector:
                # Simple selector
                if selector.startswith('[') and selector.endswith(']'):
                    attr_content = selector[1:-1]
                    self.css_attr_selectors[attr_content] = properties
                elif selector.startswith('.'):
                    self.css_classes[selector[1:]] = properties
                elif selector.startswith('#'):
                    self.css_ids[selector[1:]] = properties
                elif selector.isalpha() or selector == '*':
                    self.css_types[selector] = properties

    def _calculate_specificity(self, selector: str) -> tuple:
        """Calculate CSS specificity as (ids, classes+attrs, elements)."""
        # Split selector into parts
        parts = selector.replace('>', ' ').replace('+', ' ').replace('~', ' ').split()

        ids = 0
        classes_attrs = 0
        elements = 0

        for part in parts:
            # Count IDs (#name)
            ids += part.count('#')
            # Count classes (.name)
            classes_attrs += part.count('.')
            # Count attribute selectors [...]
            classes_attrs += part.count('[')
            # Count element types (letters only, excluding *)
            # Remove #id, .class, [attr] parts
            clean = re.sub(r'#[a-zA-Z0-9_-]+', '', part)
            clean = re.sub(r'\.[a-zA-Z0-9_-]+', '', clean)
            clean = re.sub(r'\[[^\]]+\]', '', clean)
            if clean and clean != '*' and clean[0].isalpha():
                elements += 1

        return (ids, classes_attrs, elements)

    def _selector_matches(self, selector: str, elem: ET.Element, ancestors: list) -> bool:
        """Check if a CSS selector matches an element."""
        # Split selector into parts (descendant combinator)
        parts = selector.split()
        if not parts:
            return False

        # The last part must match the current element
        if not self._simple_selector_matches(parts[-1], elem):
            return False

        if len(parts) == 1:
            return True

        # Check ancestor chain for remaining parts (right to left)
        remaining_parts = parts[:-1]
        ancestor_idx = 0

        for part in reversed(remaining_parts):
            matched = False
            while ancestor_idx < len(ancestors):
                if self._simple_selector_matches(part, ancestors[ancestor_idx]):
                    matched = True
                    ancestor_idx += 1
                    break
                ancestor_idx += 1

            if not matched:
                return False

        return True

    def _simple_selector_matches(self, selector: str, elem: ET.Element) -> bool:
        """Check if a simple selector (no combinators) matches an element."""
        if selector == '*':
            return True

        tag = self._strip_ns(elem.tag)
        elem_id = elem.get('id', '')
        elem_classes = elem.get('class', '').split()

        # Parse selector parts
        # e.g., "g#id.class" -> tag=g, id=id, class=class
        remaining = selector

        # Extract ID
        id_match = re.search(r'#([a-zA-Z0-9_-]+)', remaining)
        required_id = id_match.group(1) if id_match else None
        remaining = re.sub(r'#[a-zA-Z0-9_-]+', '', remaining)

        # Extract classes
        required_classes = re.findall(r'\.([a-zA-Z0-9_-]+)', remaining)
        remaining = re.sub(r'\.[a-zA-Z0-9_-]+', '', remaining)

        # Extract attribute selectors
        required_attrs = re.findall(r'\[([^\]]+)\]', remaining)
        remaining = re.sub(r'\[[^\]]+\]', '', remaining)

        # Remaining should be the tag name
        required_tag = remaining.strip() if remaining.strip() else None

        # Check tag
        if required_tag and required_tag != '*' and required_tag != tag:
            return False

        # Check ID
        if required_id and required_id != elem_id:
            return False

        # Check classes
        for req_class in required_classes:
            if req_class not in elem_classes:
                return False

        # Check attributes
        for attr_sel in required_attrs:
            if '=' in attr_sel:
                attr_name, attr_val = attr_sel.split('=', 1)
                attr_val = attr_val.strip('"\'')
                if elem.get(attr_name) != attr_val:
                    return False
            else:
                if elem.get(attr_sel) is None:
                    return False

        return True

    def _get_css_transform(self, elem: ET.Element) -> Optional[str]:
        """Get CSS transform property for an element.

        Checks style attribute, ID selector, class selector, and type selector.
        """
        # Check style attribute first (highest priority)
        style_str = elem.get("style", "")
        if "transform" in style_str:
            style_dict = self._parse_style_string(style_str)
            if "transform" in style_dict:
                return style_dict["transform"]

        # Check CSS ID selector
        elem_id = elem.get("id", "")
        if elem_id and elem_id in self.css_ids:
            css_props = self.css_ids[elem_id]
            if "transform" in css_props:
                return css_props["transform"]

        # Check CSS class selector
        class_attr = elem.get("class", "")
        if class_attr:
            for class_name in class_attr.split():
                if class_name in self.css_classes:
                    css_props = self.css_classes[class_name]
                    if "transform" in css_props:
                        return css_props["transform"]

        # Check CSS type selector
        tag_name = self._strip_ns(elem.tag)
        if tag_name in self.css_types:
            css_props = self.css_types[tag_name]
            if "transform" in css_props:
                return css_props["transform"]

        return None

    def _collect_defs(self, root: ET.Element):
        """Collect all definitions (gradients, filters, etc.)."""
        # Visual element tags that can be referenced by feImage
        visual_tags = {"rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "g", "text", "image", "use"}

        for elem in root.iter():
            tag = self._strip_ns(elem.tag)
            elem_id = elem.get("id")

            if elem_id:
                self.defs[elem_id] = elem
                # Parse visual elements inside defs for feImage references
                if tag in visual_tags:
                    parsed = self._parse_element(elem, Transform.identity(), Style(), depth=1)
                    if parsed:
                        self.elements_by_id[elem_id] = parsed
                # Store path data for textPath references
                if tag == "path":
                    d = elem.get("d", "")
                    if d:
                        self.path_data_by_id[elem_id] = d

            if tag == "linearGradient":
                self._parse_linear_gradient(elem)
            elif tag == "radialGradient":
                self._parse_radial_gradient(elem)
            elif tag == "pattern":
                self._parse_pattern(elem)
            elif tag == "filter":
                self._parse_filter(elem)

    def _parse_clip_paths(self, root: ET.Element):
        """Parse all clipPath elements."""
        for elem in root.iter():
            tag = self._strip_ns(elem.tag)
            if tag == "clipPath":
                clip_id = elem.get("id")
                if clip_id:
                    # Parse child elements as clip path shapes
                    clip_elements = []
                    for child in elem:
                        child_tag = self._strip_ns(child.tag)
                        parsed = self._parse_element(
                            child, Transform.identity(), Style(), depth=1
                        )
                        if parsed:
                            clip_elements.append(parsed)

                    # Check for nested clip-path attribute (for intersection)
                    nested_clip = None
                    clip_path_attr = elem.get("clip-path")
                    if clip_path_attr and clip_path_attr.startswith("url(#"):
                        nested_clip = clip_path_attr[5:-1]

                    self.clip_paths[clip_id] = ClipPath(
                        id=clip_id,
                        elements=clip_elements,
                        clip_path_id=nested_clip
                    )

    def _parse_masks(self, root: ET.Element):
        """Parse all mask elements."""
        for elem in root.iter():
            tag = self._strip_ns(elem.tag)
            if tag == "mask":
                mask_id = elem.get("id")
                if mask_id:
                    # Parse child elements as mask content
                    mask_elements = []
                    for child in elem:
                        child_tag = self._strip_ns(child.tag)
                        parsed = self._parse_element(
                            child, Transform.identity(), Style(), depth=1
                        )
                        if parsed:
                            mask_elements.append(parsed)

                    # Parse mask attributes
                    def parse_mask_val(val: str, default: float) -> float:
                        if val is None:
                            return default
                        val = val.strip()
                        if val.endswith('%'):
                            return float(val[:-1]) / 100.0
                        try:
                            return float(val)
                        except ValueError:
                            return default

                    self.masks[mask_id] = Mask(
                        id=mask_id,
                        elements=mask_elements,
                        x=parse_mask_val(elem.get("x"), -0.1),
                        y=parse_mask_val(elem.get("y"), -0.1),
                        width=parse_mask_val(elem.get("width"), 1.2),
                        height=parse_mask_val(elem.get("height"), 1.2),
                        mask_units=elem.get("maskUnits", "objectBoundingBox"),
                        mask_content_units=elem.get("maskContentUnits", "userSpaceOnUse")
                    )

    def _parse_filter(self, elem: ET.Element):
        """Parse a filter element with all filter primitives."""
        filter_id = elem.get("id")
        if not filter_id:
            return

        # Check for xlink:href to inherit from another filter
        href = elem.get(f"{self.XLINK_NS}href") or elem.get("href")
        base_filter = None
        if href and href.startswith("#"):
            base_id = href[1:]
            base_filter = self.filters.get(base_id)

        # Parse filter region attributes (can be percentages like "-10%")
        def parse_filter_val(val: str, default: float) -> float:
            if val is None:
                return default
            val = val.strip()
            if val.endswith('%'):
                return float(val[:-1]) / 100.0
            try:
                return float(val)
            except ValueError:
                return default

        # Use base filter defaults if inheriting
        default_x = base_filter.x if base_filter else -0.1
        default_y = base_filter.y if base_filter else -0.1
        default_w = base_filter.width if base_filter else 1.2
        default_h = base_filter.height if base_filter else 1.2

        filter_x = parse_filter_val(elem.get("x"), default_x)
        filter_y = parse_filter_val(elem.get("y"), default_y)
        filter_w = parse_filter_val(elem.get("width"), default_w)
        filter_h = parse_filter_val(elem.get("height"), default_h)
        filter_units_raw = elem.get("filterUnits") or (base_filter.filter_units if base_filter else "objectBoundingBox")
        # Validate filterUnits - only objectBoundingBox and userSpaceOnUse are valid
        filter_units = filter_units_raw if filter_units_raw in ("objectBoundingBox", "userSpaceOnUse") else "objectBoundingBox"
        primitive_units_raw = elem.get("primitiveUnits") or (base_filter.primitive_units if base_filter else "userSpaceOnUse")
        primitive_units = primitive_units_raw if primitive_units_raw in ("objectBoundingBox", "userSpaceOnUse") else "userSpaceOnUse"
        color_interp = elem.get("color-interpolation-filters") or (base_filter.color_interpolation_filters if base_filter else "linearRGB")

        primitives = []

        # Parse all filter primitives
        for child in elem:
            tag = self._strip_ns(child.tag)
            prim = self._parse_filter_primitive(child, tag)
            if prim:
                primitives.append(prim)

        # If no primitives and we have a base filter, inherit its primitives
        if not primitives and base_filter:
            primitives = base_filter.primitives[:]

        self.filters[filter_id] = Filter(
            id=filter_id,
            primitives=primitives,
            x=filter_x,
            y=filter_y,
            width=filter_w,
            height=filter_h,
            filter_units=filter_units,
            primitive_units=primitive_units,
            color_interpolation_filters=color_interp
        )

    def _parse_filter_primitive(self, elem: ET.Element, tag: str) -> Optional[FilterPrimitive]:
        """Parse a single filter primitive element."""
        # Common attributes
        input1 = elem.get("in")
        input2 = elem.get("in2")
        result = elem.get("result")

        # Subregion
        def parse_opt_float(val: str) -> Optional[float]:
            if val is None:
                return None
            try:
                if val.endswith('%'):
                    return float(val[:-1]) / 100.0
                return float(val)
            except ValueError:
                return None

        x = parse_opt_float(elem.get("x"))
        y = parse_opt_float(elem.get("y"))
        width = parse_opt_float(elem.get("width"))
        height = parse_opt_float(elem.get("height"))

        if tag == "feGaussianBlur":
            std_dev = elem.get("stdDeviation", "0")
            parts = std_dev.split()
            try:
                std_x = float(parts[0]) if parts else 0.0
                std_y = float(parts[1]) if len(parts) > 1 else std_x
            except ValueError:
                std_x = std_y = 0.0
            return FeGaussianBlur(input1=input1, result=result, x=x, y=y, width=width, height=height,
                                   std_deviation_x=std_x, std_deviation_y=std_y)

        elif tag == "feOffset":
            dx = _safe_float(elem.get("dx", "0"))
            dy = _safe_float(elem.get("dy", "0"))
            return FeOffset(input1=input1, result=result, x=x, y=y, width=width, height=height,
                           dx=dx, dy=dy)

        elif tag == "feFlood":
            flood_color = elem.get("flood-color", "black")
            flood_opacity = _safe_float(elem.get("flood-opacity", "1"), 1.0)
            color = self._parse_color(flood_color)
            if color is None:
                color = (0, 0, 0)
            rgba = (color[0], color[1], color[2], int(flood_opacity * 255))
            return FeFlood(input1=input1, result=result, x=x, y=y, width=width, height=height,
                          flood_color=rgba, flood_opacity=flood_opacity)

        elif tag == "feBlend":
            mode = elem.get("mode", "normal")
            return FeBlend(input1=input1, input2=input2, result=result, x=x, y=y, width=width, height=height,
                          mode=mode)

        elif tag == "feComposite":
            operator = elem.get("operator", "over")
            k1 = float(elem.get("k1", "0"))
            k2 = float(elem.get("k2", "0"))
            k3 = float(elem.get("k3", "0"))
            k4 = float(elem.get("k4", "0"))
            return FeComposite(input1=input1, input2=input2, result=result, x=x, y=y, width=width, height=height,
                              operator=operator, k1=k1, k2=k2, k3=k3, k4=k4)

        elif tag == "feMerge":
            nodes = []
            for node_elem in elem:
                node_tag = self._strip_ns(node_elem.tag)
                if node_tag == "feMergeNode":
                    node_in = node_elem.get("in", "SourceGraphic")
                    nodes.append(FeMergeNode(input1=node_in))
            return FeMerge(result=result, x=x, y=y, width=width, height=height, nodes=nodes)

        elif tag == "feColorMatrix":
            matrix_type = elem.get("type", "matrix")
            values_str = elem.get("values", "")
            try:
                values = [float(v) for v in values_str.split()]
            except ValueError:
                values = []
            return FeColorMatrix(input1=input1, result=result, x=x, y=y, width=width, height=height,
                                type=matrix_type, values=values)

        elif tag == "feComponentTransfer":
            func_r = self._parse_transfer_func(elem, "feFuncR")
            func_g = self._parse_transfer_func(elem, "feFuncG")
            func_b = self._parse_transfer_func(elem, "feFuncB")
            func_a = self._parse_transfer_func(elem, "feFuncA")
            return FeComponentTransfer(input1=input1, result=result, x=x, y=y, width=width, height=height,
                                       func_r=func_r, func_g=func_g, func_b=func_b, func_a=func_a)

        elif tag == "feMorphology":
            operator = elem.get("operator", "erode")
            radius = elem.get("radius", "0")
            parts = radius.split()
            try:
                rx = float(parts[0]) if parts else 0.0
                ry = float(parts[1]) if len(parts) > 1 else rx
            except ValueError:
                rx = ry = 0.0
            return FeMorphology(input1=input1, result=result, x=x, y=y, width=width, height=height,
                               operator=operator, radius_x=rx, radius_y=ry)

        elif tag == "feConvolveMatrix":
            order = elem.get("order", "3")
            parts = order.split()
            try:
                ox = max(1, int(parts[0])) if parts else 3
                oy = max(1, int(parts[1])) if len(parts) > 1 else ox
            except ValueError:
                ox = oy = 3
            kernel_str = elem.get("kernelMatrix", "")
            try:
                kernel = [float(v) for v in kernel_str.split()]
            except ValueError:
                kernel = []
            divisor_str = elem.get("divisor")
            divisor = float(divisor_str) if divisor_str else None
            bias = float(elem.get("bias", "0"))
            target_x_str = elem.get("targetX")
            target_y_str = elem.get("targetY")
            target_x = int(target_x_str) if target_x_str else None
            target_y = int(target_y_str) if target_y_str else None
            edge_mode = elem.get("edgeMode", "duplicate")
            preserve_alpha = elem.get("preserveAlpha", "false").lower() == "true"
            return FeConvolveMatrix(input1=input1, result=result, x=x, y=y, width=width, height=height,
                                    order_x=ox, order_y=oy, kernel_matrix=kernel, divisor=divisor,
                                    bias=bias, target_x=target_x, target_y=target_y,
                                    edge_mode=edge_mode, preserve_alpha=preserve_alpha)

        elif tag == "feTurbulence":
            turb_type = elem.get("type", "turbulence")
            base_freq = elem.get("baseFrequency", "0")
            parts = base_freq.split()
            try:
                bfx = float(parts[0]) if parts else 0.0
                bfy = float(parts[1]) if len(parts) > 1 else bfx
            except ValueError:
                bfx = bfy = 0.0
            num_octaves = max(1, int(elem.get("numOctaves", "1")))
            seed = int(float(elem.get("seed", "0")))
            stitch = elem.get("stitchTiles", "noStitch")
            return FeTurbulence(result=result, x=x, y=y, width=width, height=height,
                               type=turb_type, base_frequency_x=bfx, base_frequency_y=bfy,
                               num_octaves=num_octaves, seed=seed, stitch_tiles=stitch)

        elif tag == "feDisplacementMap":
            scale = float(elem.get("scale", "0"))
            x_channel = elem.get("xChannelSelector", "A")
            y_channel = elem.get("yChannelSelector", "A")
            return FeDisplacementMap(input1=input1, input2=input2, result=result, x=x, y=y, width=width, height=height,
                                     scale=scale, x_channel_selector=x_channel, y_channel_selector=y_channel)

        elif tag == "feImage":
            href = elem.get(f"{self.XLINK_NS}href") or elem.get("href", "")
            par = elem.get("preserveAspectRatio", "xMidYMid meet")
            return FeImage(result=result, x=x, y=y, width=width, height=height,
                          href=href, preserveAspectRatio=par)

        elif tag == "feTile":
            return FeTile(input1=input1, result=result, x=x, y=y, width=width, height=height)

        elif tag == "feDiffuseLighting":
            surface_scale = float(elem.get("surfaceScale", "1"))
            diffuse_const = float(elem.get("diffuseConstant", "1"))
            light_color = self._parse_color(elem.get("lighting-color", "white"))
            if light_color is None:
                light_color = (255, 255, 255)
            light_source = self._parse_light_source(elem)
            return FeDiffuseLighting(input1=input1, result=result, x=x, y=y, width=width, height=height,
                                     surface_scale=surface_scale, diffuse_constant=diffuse_const,
                                     lighting_color=light_color, light_source=light_source)

        elif tag == "feSpecularLighting":
            surface_scale = float(elem.get("surfaceScale", "1"))
            spec_const = float(elem.get("specularConstant", "1"))
            spec_exp = float(elem.get("specularExponent", "1"))
            light_color = self._parse_color(elem.get("lighting-color", "white"))
            if light_color is None:
                light_color = (255, 255, 255)
            light_source = self._parse_light_source(elem)
            return FeSpecularLighting(input1=input1, result=result, x=x, y=y, width=width, height=height,
                                      surface_scale=surface_scale, specular_constant=spec_const,
                                      specular_exponent=spec_exp, lighting_color=light_color,
                                      light_source=light_source)

        elif tag == "feDropShadow":
            dx = _safe_float(elem.get("dx", "2"), 2.0)
            dy = _safe_float(elem.get("dy", "2"), 2.0)
            std_dev = elem.get("stdDeviation", "0")
            parts = std_dev.split()
            try:
                std_x = _safe_float(parts[0], 0.0) if parts else 0.0
                std_y = _safe_float(parts[1], std_x) if len(parts) > 1 else std_x
            except (ValueError, IndexError):
                std_x = std_y = 0.0
            flood_color = elem.get("flood-color", "black")
            flood_opacity = _safe_float(elem.get("flood-opacity", "1"), 1.0)
            color = self._parse_color(flood_color)
            if color is None:
                color = (0, 0, 0)
            rgba = (color[0], color[1], color[2], int(flood_opacity * 255))
            return FeDropShadow(input1=input1, result=result, x=x, y=y, width=width, height=height,
                               dx=dx, dy=dy, std_deviation_x=std_x, std_deviation_y=std_y,
                               flood_color=rgba, flood_opacity=flood_opacity)

        return None

    def _parse_transfer_func(self, parent: ET.Element, func_tag: str) -> FeComponentTransferFunc:
        """Parse a transfer function element (feFuncR, feFuncG, etc.)."""
        def safe_float(val: str, default: float) -> float:
            if not val:
                return default
            val = val.strip()
            try:
                if val.endswith('%'):
                    return float(val[:-1]) / 100.0
                # Strip px or other units
                for unit in ['px', 'em', 'pt']:
                    if val.endswith(unit):
                        val = val[:-len(unit)]
                        break
                return float(val)
            except ValueError:
                return default

        for child in parent:
            if self._strip_ns(child.tag) == func_tag:
                func_type = child.get("type", "identity")
                table_str = child.get("tableValues", "")
                try:
                    table = [float(v) for v in table_str.split()] if table_str else []
                except ValueError:
                    table = []
                slope = safe_float(child.get("slope", "1"), 1.0)
                intercept = safe_float(child.get("intercept", "0"), 0.0)
                amplitude = safe_float(child.get("amplitude", "1"), 1.0)
                exponent = safe_float(child.get("exponent", "1"), 1.0)
                offset = safe_float(child.get("offset", "0"), 0.0)
                return FeComponentTransferFunc(type=func_type, table_values=table,
                                               slope=slope, intercept=intercept,
                                               amplitude=amplitude, exponent=exponent, offset=offset)
        return FeComponentTransferFunc()

    def _parse_light_source(self, parent: ET.Element) -> Optional[LightSource]:
        """Parse light source child element."""
        for child in parent:
            tag = self._strip_ns(child.tag)
            if tag == "feDistantLight":
                azimuth = float(child.get("azimuth", "0"))
                elevation = float(child.get("elevation", "0"))
                return FeDistantLight(azimuth=azimuth, elevation=elevation)
            elif tag == "fePointLight":
                x = float(child.get("x", "0"))
                y = float(child.get("y", "0"))
                z = float(child.get("z", "0"))
                return FePointLight(x=x, y=y, z=z)
            elif tag == "feSpotLight":
                x = float(child.get("x", "0"))
                y = float(child.get("y", "0"))
                z = float(child.get("z", "0"))
                px = float(child.get("pointsAtX", "0"))
                py = float(child.get("pointsAtY", "0"))
                pz = float(child.get("pointsAtZ", "0"))
                spec_exp = float(child.get("specularExponent", "1"))
                cone_str = child.get("limitingConeAngle")
                cone = float(cone_str) if cone_str else None
                return FeSpotLight(x=x, y=y, z=z, points_at_x=px, points_at_y=py, points_at_z=pz,
                                  specular_exponent=spec_exp, limiting_cone_angle=cone)
        return None

    def _parse_linear_gradient(self, elem: ET.Element):
        """Parse a linearGradient element."""
        grad_id = elem.get("id")
        if not grad_id:
            return

        # Get href for inheritance
        href = elem.get(f"{self.XLINK_NS}href") or elem.get("href")
        if href and href.startswith("#"):
            href = href[1:]

        # Validate gradientUnits - mark as invalid if not valid value
        units_raw = elem.get("gradientUnits")
        if units_raw is not None and units_raw not in ("objectBoundingBox", "userSpaceOnUse"):
            units = "invalid"  # Mark as invalid - renderer should not use this gradient
        else:
            units = units_raw or "objectBoundingBox"

        # Parse spreadMethod - default is "pad"
        spread_method = elem.get("spreadMethod", "pad")
        if spread_method not in ("pad", "reflect", "repeat"):
            spread_method = "pad"

        grad = LinearGradient(
            id=grad_id,
            x1=self._parse_gradient_coord(elem.get("x1", "0%")),
            y1=self._parse_gradient_coord(elem.get("y1", "0%")),
            x2=self._parse_gradient_coord(elem.get("x2", "100%")),
            y2=self._parse_gradient_coord(elem.get("y2", "0%")),
            units=units,
            href=href,
            spread_method=spread_method
        )

        # Parse transform
        transform_str = elem.get("gradientTransform")
        if transform_str:
            grad.transform = self._parse_transform(transform_str)

        # Parse stops
        grad.stops = self._parse_gradient_stops(elem)

        self.gradients[grad_id] = grad

    def _parse_radial_gradient(self, elem: ET.Element):
        """Parse a radialGradient element."""
        grad_id = elem.get("id")
        if not grad_id:
            return

        href = elem.get(f"{self.XLINK_NS}href") or elem.get("href")
        if href and href.startswith("#"):
            href = href[1:]

        # Validate gradientUnits - mark as invalid if not valid value
        units_raw = elem.get("gradientUnits")
        if units_raw is not None and units_raw not in ("objectBoundingBox", "userSpaceOnUse"):
            units = "invalid"  # Mark as invalid - renderer should not use this gradient
        else:
            units = units_raw or "objectBoundingBox"

        # Parse spreadMethod - default is "pad"
        spread_method = elem.get("spreadMethod", "pad")
        if spread_method not in ("pad", "reflect", "repeat"):
            spread_method = "pad"

        grad = RadialGradient(
            id=grad_id,
            cx=self._parse_gradient_coord(elem.get("cx", "50%")),
            cy=self._parse_gradient_coord(elem.get("cy", "50%")),
            r=self._parse_gradient_coord(elem.get("r", "50%")),
            units=units,
            href=href,
            spread_method=spread_method
        )

        fx = elem.get("fx")
        fy = elem.get("fy")
        fr = elem.get("fr")
        if fx:
            grad.fx = self._parse_gradient_coord(fx)
        if fy:
            grad.fy = self._parse_gradient_coord(fy)
        if fr:
            grad.fr = self._parse_gradient_coord(fr)

        transform_str = elem.get("gradientTransform")
        if transform_str:
            grad.transform = self._parse_transform(transform_str)

        grad.stops = self._parse_gradient_stops(elem)

        self.gradients[grad_id] = grad

    def _parse_gradient_coord(self, value: str) -> float:
        """Parse gradient coordinate (can be percentage or absolute)."""
        value = value.strip()
        if value.endswith("%"):
            return float(value[:-1]) / 100.0
        # Try parsing with units
        try:
            return self._parse_length(value)
        except (ValueError, AttributeError):
            try:
                return float(value)
            except ValueError:
                return 0.0

    def _parse_gradient_stops(self, elem: ET.Element) -> list[GradientStop]:
        """Parse gradient stop elements."""
        stops = []
        for child in elem:
            tag = self._strip_ns(child.tag)
            if tag == "stop":
                offset = child.get("offset", "0")
                if offset.endswith("%"):
                    try:
                        offset = float(offset[:-1]) / 100.0
                    except ValueError:
                        offset = 0.0
                else:
                    try:
                        offset = float(offset)
                    except ValueError:
                        # Invalid offset (like "5mm") - treat as 0
                        offset = 0.0

                # Get color from style or attributes
                style_str = child.get("style", "")
                style_dict = self._parse_style_string(style_str)

                color_str = style_dict.get("stop-color") or child.get("stop-color", "black")
                opacity_str = style_dict.get("stop-opacity") or child.get("stop-opacity", "1")

                color = self._parse_color(color_str)
                if color:
                    try:
                        if opacity_str.endswith('%'):
                            opacity = float(opacity_str[:-1]) / 100.0
                        else:
                            opacity = float(opacity_str)
                    except (ValueError, AttributeError):
                        opacity = 1.0
                    color = (color[0], color[1], color[2], int(opacity * 255))
                else:
                    color = (0, 0, 0, 255)

                stops.append(GradientStop(offset=offset, color=color))

        return sorted(stops, key=lambda s: s.offset)

    def _resolve_gradient_refs(self):
        """Resolve gradient href references."""
        for grad in self.gradients.values():
            if grad.href and grad.href in self.gradients:
                ref = self.gradients[grad.href]
                # Inherit stops if not defined
                if not grad.stops and ref.stops:
                    grad.stops = ref.stops.copy()

    def _parse_pattern(self, elem: ET.Element):
        """Parse a pattern element."""
        pattern_id = elem.get("id")
        if not pattern_id:
            return

        # Get pattern attributes
        x = self._parse_length(elem.get("x", "0"))
        y = self._parse_length(elem.get("y", "0"))
        width = self._parse_length(elem.get("width", "0"))
        height = self._parse_length(elem.get("height", "0"))

        pattern_units = elem.get("patternUnits", "objectBoundingBox")
        pattern_content_units = elem.get("patternContentUnits", "userSpaceOnUse")

        # Parse transform
        transform = None
        transform_str = elem.get("patternTransform")
        if transform_str:
            transform = self._parse_transform(transform_str)

        # Parse href reference
        href = None
        for attr in ["href", "{http://www.w3.org/1999/xlink}href"]:
            href_val = elem.get(attr)
            if href_val and href_val.startswith("#"):
                href = href_val[1:]
                break

        # Parse viewBox
        viewbox = None
        viewbox_str = elem.get("viewBox")
        if viewbox_str:
            try:
                parts = viewbox_str.replace(",", " ").split()
                if len(parts) >= 4:
                    viewbox = tuple(float(p) for p in parts[:4])
            except ValueError:
                pass

        # Parse child elements
        elements = []
        for child in elem:
            child_tag = self._strip_ns(child.tag)
            if child_tag not in ["title", "desc", "metadata"]:
                parsed = self._parse_element(child, Transform.identity(), Style(), depth=1)
                if parsed:
                    elements.append(parsed)

        pattern = Pattern(
            id=pattern_id,
            x=x,
            y=y,
            width=width,
            height=height,
            pattern_units=pattern_units,
            pattern_content_units=pattern_content_units,
            transform=transform,
            href=href,
            viewbox=viewbox,
            elements=elements
        )

        self.patterns[pattern_id] = pattern

    def _resolve_pattern_refs(self):
        """Resolve pattern href references."""
        for pattern in self.patterns.values():
            if pattern.href and pattern.href in self.patterns:
                ref = self.patterns[pattern.href]
                # Inherit elements if not defined
                if not pattern.elements and ref.elements:
                    pattern.elements = ref.elements.copy()
                # Inherit dimensions if not set
                if pattern.width == 0 and ref.width > 0:
                    pattern.width = ref.width
                if pattern.height == 0 and ref.height > 0:
                    pattern.height = ref.height

    def _parse_children(self, parent: ET.Element, parent_transform: Transform,
                        parent_style: Style, parent_text_anchor: str = "start",
                        parent_font_family: str = "Arial", parent_font_size: float = 16,
                        depth: int = 0, ancestors: list = None) -> list[SVGElement]:
        """Parse child elements."""
        if ancestors is None:
            ancestors = []

        elements = []
        # Children will have parent as their closest ancestor
        # ancestors is already in closest-first order
        child_ancestors = [parent] + list(ancestors)

        for child in parent:
            tag = self._strip_ns(child.tag)

            # Skip defs, metadata, etc.
            # symbol elements only render when referenced by <use>
            if tag in ("defs", "metadata", "title", "desc", "style",
                       "linearGradient", "radialGradient", "SVGTestCase",
                       "OperatorScript", "Paragraph", "symbol"):
                continue

            elem = self._parse_element(child, parent_transform, parent_style, parent_text_anchor,
                                       parent_font_family, parent_font_size, depth=depth,
                                       ancestors=child_ancestors)
            if elem:
                elements.append(elem)

        return elements

    def _parse_element(self, elem: ET.Element, parent_transform: Transform,
                       parent_style: Style, parent_text_anchor: str = "start",
                       parent_font_family: str = "Arial", parent_font_size: float = 16,
                       depth: int = 0, ancestors: list = None) -> Optional[SVGElement]:
        """Parse a single SVG element."""
        # Prevent infinite recursion
        if depth > self.MAX_PARSE_DEPTH:
            return None

        if ancestors is None:
            ancestors = []

        tag = self._strip_ns(elem.tag)

        # Parse style (inheriting from parent, using ancestors for CSS matching)
        style = self._parse_style(elem, parent_style, ancestors)

        # Extract text properties from this element (for inheritance to children)
        text_anchor = parent_text_anchor
        font_family = parent_font_family
        font_size = parent_font_size

        style_str = elem.get("style", "")

        if elem.get("text-anchor"):
            text_anchor = elem.get("text-anchor")
        elif "text-anchor" in style_str:
            for part in style_str.split(";"):
                if "text-anchor" in part:
                    key, _, value = part.partition(":")
                    if key.strip() == "text-anchor":
                        text_anchor = value.strip()

        if elem.get("font-family"):
            font_family = elem.get("font-family")
        elif "font-family" in style_str:
            for part in style_str.split(";"):
                if "font-family" in part:
                    key, _, value = part.partition(":")
                    if key.strip() == "font-family":
                        font_family = value.strip()

        if elem.get("font-size"):
            font_size = self._parse_length(elem.get("font-size"))
        elif "font-size" in style_str:
            for part in style_str.split(";"):
                if "font-size" in part:
                    key, _, value = part.partition(":")
                    if key.strip() == "font-size":
                        font_size = self._parse_length(value.strip())

        # Parse transform from attribute and CSS
        # Note: SVG 1.1 doesn't allow transform on symbol elements, so skip it
        if tag == "symbol":
            transform_str = ""
        else:
            transform_str = elem.get("transform", "")

            # Also check CSS for transform property
            css_transform_str = self._get_css_transform(elem)
            if css_transform_str and not transform_str:
                transform_str = css_transform_str
            elif css_transform_str and transform_str:
                # CSS transform is applied after attribute transform (CSS takes precedence per SVG2)
                transform_str = css_transform_str

        local_transform = self._parse_transform(transform_str) if transform_str else Transform.identity()
        transform = parent_transform.multiply(local_transform)

        # Parse clip-path and mask attributes
        clip_path_id = self._parse_url_reference(elem.get("clip-path", ""))
        mask_id = self._parse_url_reference(elem.get("mask", ""))

        result = None
        if tag == "rect":
            result = self._parse_rect(elem, style, transform)
        elif tag == "circle":
            result = self._parse_circle(elem, style, transform)
        elif tag == "ellipse":
            result = self._parse_ellipse(elem, style, transform)
        elif tag == "line":
            result = self._parse_line(elem, style, transform)
        elif tag == "polyline":
            result = self._parse_polyline(elem, style, transform)
        elif tag == "polygon":
            result = self._parse_polygon(elem, style, transform)
        elif tag == "path":
            result = self._parse_path(elem, style, transform)
            # Store path data by ID for textPath references
            path_id = elem.get("id")
            if path_id:
                d = elem.get("d", "")
                if d:
                    self.path_data_by_id[path_id] = d
        elif tag == "g" or tag == "a":
            # Both g (group) and a (anchor) elements render children as a group
            # Children will have this element as their closest ancestor
            child_ancestors = [elem] + list(ancestors)
            result = self._parse_group(elem, style, transform, parent_style, text_anchor, font_family, font_size, depth + 1, child_ancestors)
        elif tag == "symbol":
            # Symbol is like a group but may have viewBox
            child_ancestors = [elem] + list(ancestors)
            result = self._parse_symbol(elem, style, transform, parent_style, text_anchor, font_family, font_size, depth + 1, child_ancestors)
        elif tag == "switch":
            child_ancestors = [elem] + list(ancestors)
            result = self._parse_switch(elem, style, transform, parent_style, text_anchor, font_family, font_size, depth + 1, child_ancestors)
        elif tag == "text":
            result = self._parse_text(elem, style, transform, text_anchor, font_family, font_size)
        elif tag == "use":
            result = self._parse_use(elem, style, transform, parent_style, depth + 1)
        elif tag == "svg":
            child_ancestors = [elem] + list(ancestors)
            result = self._parse_nested_svg(elem, style, transform, parent_style, text_anchor, font_family, font_size, depth + 1, child_ancestors)
        elif tag == "image":
            result = self._parse_image(elem, style, transform)

        # Set clip path and mask on the parsed element
        if result:
            if clip_path_id:
                result.clip_path_id = clip_path_id
            if mask_id:
                result.mask_id = mask_id
            # Track element by ID for feImage references
            elem_id = elem.get("id")
            if elem_id:
                self.elements_by_id[elem_id] = result

        return result

    def _parse_url_reference(self, value: str) -> Optional[str]:
        """Parse a url() reference and return the ID."""
        if not value:
            return None
        value = value.strip()
        if value.startswith("url(") and value.endswith(")"):
            ref = value[4:-1].strip()
            # Strip quotes if present (SVG 2 allows quoted URLs)
            if (ref.startswith("'") and ref.endswith("'")) or (ref.startswith('"') and ref.endswith('"')):
                ref = ref[1:-1]
            if ref.startswith("#"):
                return ref[1:]
            return ref
        return None

    def _parse_style(self, elem: ET.Element, parent_style: Style, ancestors: list = None) -> Style:
        """Parse style from element attributes and style attribute.

        Args:
            elem: The element to parse style for
            parent_style: Parent element's style for inheritance
            ancestors: List of ancestor elements (closest first) for CSS matching
        """
        if ancestors is None:
            ancestors = []

        # Start with parent style (inheritance)
        # If parent has fill specified (including None for "none"), inherit it
        # Otherwise use the default black fill
        if parent_style.fill is FILL_NOT_SET:
            inherited_fill = (0, 0, 0, 255)  # Default SVG fill is black
        else:
            inherited_fill = parent_style.fill

        style = Style(
            fill=inherited_fill,
            fill_opacity=parent_style.fill_opacity,
            stroke=parent_style.stroke,
            stroke_width=parent_style.stroke_width,
            stroke_opacity=parent_style.stroke_opacity,
            stroke_linecap=parent_style.stroke_linecap,
            stroke_linejoin=parent_style.stroke_linejoin,
            stroke_miterlimit=parent_style.stroke_miterlimit,
            opacity=parent_style.opacity,
            fill_rule=parent_style.fill_rule,
            display=parent_style.display,
            visibility=parent_style.visibility
        )

        # SVG style cascade order:
        # 1. Presentation attributes (lowest priority)
        # 2. CSS rules in specificity order
        # 3. Style attribute (highest priority)

        style_dict = {}

        # First: Presentation attributes (lowest priority - can be overridden by CSS)
        for attr in ["fill", "stroke", "stroke-width", "fill-opacity",
                     "stroke-opacity", "opacity", "fill-rule",
                     "stroke-linecap", "stroke-linejoin", "stroke-miterlimit",
                     "stroke-dasharray", "filter", "display", "visibility"]:
            val = elem.get(attr)
            if val:
                style_dict[attr] = val

        # Apply complex CSS rules sorted by specificity (lowest first, so higher overrides)
        matching_rules = []
        for selector, properties, specificity, order in self.css_rules:
            if self._selector_matches(selector, elem, ancestors):
                matching_rules.append((specificity, order, properties))

        # Sort by specificity then order (both ascending, so later/more specific wins)
        matching_rules.sort(key=lambda x: (x[0], x[1]))
        for _, _, props in matching_rules:
            style_dict.update(props)

        # Legacy: Apply simple CSS selectors (for backward compatibility)
        # These should already be included in css_rules, but keep for now
        tag_name = self._strip_ns(elem.tag)
        if tag_name in self.css_types and not self.css_rules:
            style_dict.update(self.css_types[tag_name])

        # Apply CSS attribute selectors
        for attr_selector, props in self.css_attr_selectors.items():
            # Handle [attr] (presence), [attr=value], [attr~=value], etc.
            if '=' in attr_selector:
                # Has value comparison
                if '~=' in attr_selector:
                    attr_name, attr_value = attr_selector.split('~=', 1)
                    attr_value = attr_value.strip('"\'')
                    elem_value = elem.get(attr_name.strip(), "")
                    if attr_value in elem_value.split():
                        style_dict.update(props)
                elif '|=' in attr_selector:
                    attr_name, attr_value = attr_selector.split('|=', 1)
                    attr_value = attr_value.strip('"\'')
                    elem_value = elem.get(attr_name.strip(), "")
                    if elem_value == attr_value or elem_value.startswith(attr_value + '-'):
                        style_dict.update(props)
                else:
                    # Simple equality: [attr=value]
                    attr_name, attr_value = attr_selector.split('=', 1)
                    attr_value = attr_value.strip('"\'')
                    if elem.get(attr_name.strip()) == attr_value:
                        style_dict.update(props)
            else:
                # Just presence check: [attr]
                if elem.get(attr_selector) is not None:
                    style_dict.update(props)

        # Apply CSS class selectors
        class_attr = elem.get("class", "")
        if class_attr:
            for class_name in class_attr.split():
                if class_name in self.css_classes:
                    style_dict.update(self.css_classes[class_name])

        # Apply CSS ID selectors (higher priority than class)
        elem_id = elem.get("id", "")
        if elem_id and elem_id in self.css_ids:
            style_dict.update(self.css_ids[elem_id])

        # Parse style attribute (highest priority - overrides everything)
        style_str = elem.get("style", "")
        style_dict.update(self._parse_style_string(style_str))

        # Apply parsed values
        if "fill" in style_dict:
            fill_val = style_dict["fill"]
            if fill_val == "none" or fill_val == "transparent":
                style.fill = None
            elif fill_val.startswith("url("):
                # Gradient reference
                style.fill = fill_val
            else:
                result = self._parse_color_with_alpha(fill_val)
                if result:
                    color, alpha = result
                    # Apply color alpha to fill_opacity
                    if alpha < 1.0:
                        style.fill_opacity *= alpha
                    style.fill = (*color, 255)

        if "stroke" in style_dict:
            stroke_val = style_dict["stroke"]
            if stroke_val == "none" or stroke_val == "transparent":
                style.stroke = None
            elif stroke_val.startswith("url("):
                style.stroke = stroke_val
            else:
                result = self._parse_color_with_alpha(stroke_val)
                if result:
                    color, alpha = result
                    # Apply color alpha to stroke_opacity
                    if alpha < 1.0:
                        style.stroke_opacity *= alpha
                    style.stroke = (*color, 255)

        if "stroke-width" in style_dict:
            style.stroke_width = self._parse_length(style_dict["stroke-width"])

        if "fill-opacity" in style_dict:
            try:
                val = style_dict["fill-opacity"]
                if val.endswith('%'):
                    style.fill_opacity = max(0.0, min(1.0, float(val[:-1]) / 100.0))
                else:
                    style.fill_opacity = max(0.0, min(1.0, float(val)))
            except (ValueError, AttributeError):
                pass

        if "stroke-opacity" in style_dict:
            try:
                val = style_dict["stroke-opacity"]
                if val.endswith('%'):
                    style.stroke_opacity = max(0.0, min(1.0, float(val[:-1]) / 100.0))
                else:
                    style.stroke_opacity = max(0.0, min(1.0, float(val)))
            except (ValueError, AttributeError):
                pass

        if "opacity" in style_dict:
            try:
                val = style_dict["opacity"]
                if val.endswith('%'):
                    style.opacity = max(0.0, min(1.0, float(val[:-1]) / 100.0))
                else:
                    style.opacity = max(0.0, min(1.0, float(val)))
            except (ValueError, AttributeError):
                pass

        if "fill-rule" in style_dict:
            style.fill_rule = style_dict["fill-rule"]

        if "stroke-linecap" in style_dict:
            style.stroke_linecap = style_dict["stroke-linecap"]

        if "stroke-linejoin" in style_dict:
            style.stroke_linejoin = style_dict["stroke-linejoin"]

        if "stroke-miterlimit" in style_dict:
            try:
                style.stroke_miterlimit = float(style_dict["stroke-miterlimit"])
            except (ValueError, AttributeError):
                pass

        if "stroke-dasharray" in style_dict:
            dasharray_str = style_dict["stroke-dasharray"].strip()
            if dasharray_str and dasharray_str.lower() != "none":
                # Parse comma or space separated values
                parts = dasharray_str.replace(",", " ").split()
                try:
                    style.stroke_dasharray = [self._parse_length(p) for p in parts if p]
                except ValueError:
                    pass

        if "display" in style_dict:
            # Only allow display override if parent is NOT display:none
            # CSS spec: children of display:none elements are never rendered
            if parent_style.display != "none":
                style.display = style_dict["display"]

        if "visibility" in style_dict:
            # visibility is inherited but CAN be overridden by children
            # (unlike display:none which hides entire subtree)
            style.visibility = style_dict["visibility"]

        # Parse filter reference
        if "filter" in style_dict:
            filter_val = style_dict["filter"]
            # Handle single url(#id) or multiple urls (SVG 2)
            # For multiple urls like "url(#f1) url(#f2)", use the first one
            if "url(#" in filter_val:
                # Extract first url reference
                import re
                match = re.search(r'url\(#([^)]+)\)', filter_val)
                if match:
                    style.filter_id = match.group(1)

        return style

    def _parse_style_string(self, style_str: str) -> dict[str, str]:
        """Parse CSS-style string into dictionary."""
        result = {}
        for part in style_str.split(";"):
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                result[key.strip()] = val.strip()
        return result

    def _parse_color(self, color_str: str) -> Optional[tuple[int, int, int]]:
        """Parse color string to RGB tuple."""
        if not color_str or color_str == "none":
            return None

        color_str = color_str.strip().lower()

        # currentColor keyword - defaults to black
        if color_str == "currentcolor":
            return (0, 0, 0)

        # Named color
        if color_str in self.COLORS:
            return self.COLORS[color_str]

        # Hex color
        if color_str.startswith("#"):
            hex_str = color_str[1:]
            try:
                if len(hex_str) == 3:
                    # Short form #RGB
                    r = int(hex_str[0] * 2, 16)
                    g = int(hex_str[1] * 2, 16)
                    b = int(hex_str[2] * 2, 16)
                    return (r, g, b)
                elif len(hex_str) == 4:
                    # Short form #RGBA
                    r = int(hex_str[0] * 2, 16)
                    g = int(hex_str[1] * 2, 16)
                    b = int(hex_str[2] * 2, 16)
                    # Alpha is handled separately
                    return (r, g, b)
                elif len(hex_str) == 6:
                    # Full form #RRGGBB
                    r = int(hex_str[0:2], 16)
                    g = int(hex_str[2:4], 16)
                    b = int(hex_str[4:6], 16)
                    return (r, g, b)
                elif len(hex_str) == 8:
                    # Full form #RRGGBBAA
                    r = int(hex_str[0:2], 16)
                    g = int(hex_str[2:4], 16)
                    b = int(hex_str[4:6], 16)
                    # Alpha is handled separately
                    return (r, g, b)
            except ValueError:
                # Invalid hex characters
                return None

        # RGB function
        rgb_match = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color_str)
        if rgb_match:
            return (int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3)))

        # RGB with percentages
        rgb_pct_match = re.match(r"rgb\s*\(\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)", color_str)
        if rgb_pct_match:
            return (
                int(float(rgb_pct_match.group(1)) * 255 / 100),
                int(float(rgb_pct_match.group(2)) * 255 / 100),
                int(float(rgb_pct_match.group(3)) * 255 / 100)
            )

        # HSL function: hsl(h, s%, l%)
        hsl_match = re.match(r"hsl\s*\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)", color_str)
        if hsl_match:
            h = float(hsl_match.group(1)) / 360.0
            s = float(hsl_match.group(2)) / 100.0
            l = float(hsl_match.group(3)) / 100.0
            return self._hsl_to_rgb(h, s, l)

        # HSLA function: hsla(h, s%, l%, a)
        hsla_match = re.match(r"hsla\s*\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)\s*\)", color_str)
        if hsla_match:
            h = float(hsla_match.group(1)) / 360.0
            s = float(hsla_match.group(2)) / 100.0
            l = float(hsla_match.group(3)) / 100.0
            # Alpha is handled separately by stop-opacity, ignore here
            return self._hsl_to_rgb(h, s, l)

        return None

    def _parse_color_with_alpha(self, color_str: str) -> Optional[tuple[tuple[int, int, int], float]]:
        """Parse color string and return (RGB, alpha) tuple. Alpha is 0-1."""
        if not color_str or color_str == "none":
            return None

        color_str = color_str.strip().lower()

        # Extract alpha from various formats
        alpha = 1.0

        # Hex colors with alpha
        if color_str.startswith("#"):
            hex_str = color_str[1:]
            try:
                if len(hex_str) == 4:
                    # Short form #RGBA
                    r = int(hex_str[0] * 2, 16)
                    g = int(hex_str[1] * 2, 16)
                    b = int(hex_str[2] * 2, 16)
                    alpha = int(hex_str[3] * 2, 16) / 255.0
                    return ((r, g, b), alpha)
                elif len(hex_str) == 8:
                    # Full form #RRGGBBAA
                    r = int(hex_str[0:2], 16)
                    g = int(hex_str[2:4], 16)
                    b = int(hex_str[4:6], 16)
                    alpha = int(hex_str[6:8], 16) / 255.0
                    return ((r, g, b), alpha)
            except ValueError:
                pass

        # RGBA function
        rgba_match = re.match(r"rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", color_str)
        if rgba_match:
            a = float(rgba_match.group(4))
            if a > 1:
                a = a / 255.0  # Some use 0-255 for alpha
            return ((int(rgba_match.group(1)), int(rgba_match.group(2)), int(rgba_match.group(3))), a)

        # RGBA with percentages
        rgba_pct_match = re.match(r"rgba\s*\(\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+%?)\s*\)", color_str)
        if rgba_pct_match:
            a_str = rgba_pct_match.group(4)
            if a_str.endswith('%'):
                a = float(a_str[:-1]) / 100.0
            else:
                a = float(a_str)
            return ((
                int(float(rgba_pct_match.group(1)) * 255 / 100),
                int(float(rgba_pct_match.group(2)) * 255 / 100),
                int(float(rgba_pct_match.group(3)) * 255 / 100)
            ), a)

        # HSLA function: hsla(h, s%, l%, a)
        hsla_match = re.match(r"hsla\s*\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)\s*\)", color_str)
        if hsla_match:
            h = float(hsla_match.group(1)) / 360.0
            s = float(hsla_match.group(2)) / 100.0
            l = float(hsla_match.group(3)) / 100.0
            a = float(hsla_match.group(4))
            return (self._hsl_to_rgb(h, s, l), a)

        # Fall back to regular color parsing (no alpha)
        rgb = self._parse_color(color_str)
        if rgb:
            return (rgb, 1.0)
        return None

    def _hsl_to_rgb(self, h: float, s: float, l: float) -> tuple[int, int, int]:
        """Convert HSL to RGB."""
        if s == 0:
            # Achromatic (gray)
            v = int(l * 255)
            return (v, v, v)

        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q

        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

        return (int(r * 255), int(g * 255), int(b * 255))

    def _parse_length(self, length_str: str, ref_dim: float = None) -> float:
        """Parse SVG length value.

        Args:
            length_str: The length string (e.g., "100", "50%", "10px")
            ref_dim: Reference dimension for percentage values (e.g., viewBox width/height)

        Note: For space-separated lists (e.g., "100 120"), returns the first value.
              This handles SVG text coordinate lists where each value applies to a character.
        """
        if not length_str:
            return 0

        length_str = length_str.strip()

        # Handle space-separated lists (common in text x/y coordinates)
        # Take the first value as the starting position
        if ' ' in length_str or '\t' in length_str or ',' in length_str:
            parts = length_str.replace(',', ' ').split()
            if parts:
                length_str = parts[0]
            else:
                return 0

        # Handle percentages
        if length_str.endswith("%"):
            pct = float(length_str[:-1])
            if ref_dim is not None:
                return pct * ref_dim / 100.0
            return pct  # Return raw percentage if no reference

        # Unit conversions (approximate)
        units = {
            "px": 1.0,
            "pt": 1.333,  # 96/72
            "pc": 16.0,
            "mm": 3.7795,  # 96/25.4
            "cm": 37.795,
            "in": 96.0,
            "em": 16.0,
            "ex": 8.0,
            "rem": 16.0,  # Assuming root font size of 16px
            "ch": 8.0,  # Width of '0' character, approximate
            "vw": 10.0,  # Viewport width unit, approximate
            "vh": 10.0,  # Viewport height unit, approximate
            "vmin": 10.0,  # Viewport min, approximate
            "vmax": 10.0,  # Viewport max, approximate
            "vm": 10.0,  # Old viewport min syntax
            "q": 0.945,  # Quarter-millimeter (96/25.4/4)
            "r": 16.0,  # Typo for rem, handle gracefully
            "rlh": 16.0,  # Root line height, approximate
            "lh": 16.0,  # Line height, approximate
        }

        # Sort by length (longest first) to match "vmin" before "in"
        for unit, factor in sorted(units.items(), key=lambda x: -len(x[0])):
            if length_str.endswith(unit):
                try:
                    return float(length_str[:-len(unit)]) * factor
                except ValueError:
                    return 0

        # Plain number
        try:
            return float(length_str)
        except ValueError:
            return 0

    def _parse_transform(self, transform_str: str) -> Transform:
        """Parse transform attribute."""
        result = Transform.identity()

        # Match transform functions
        pattern = r"(\w+)\s*\(([^)]+)\)"
        for match in re.finditer(pattern, transform_str):
            func = match.group(1)
            args_str = match.group(2)
            args = [float(a.strip()) for a in re.split(r"[\s,]+", args_str.strip())]

            if func == "translate":
                tx = args[0]
                ty = args[1] if len(args) > 1 else 0
                result = result.multiply(Transform.translate(tx, ty))
            elif func == "scale":
                sx = args[0]
                sy = args[1] if len(args) > 1 else sx
                result = result.multiply(Transform.scale(sx, sy))
            elif func == "rotate":
                angle = args[0]
                cx = args[1] if len(args) > 1 else 0
                cy = args[2] if len(args) > 2 else 0
                result = result.multiply(Transform.rotate(angle, cx, cy))
            elif func == "skewX":
                result = result.multiply(Transform.skewX(args[0]))
            elif func == "skewY":
                result = result.multiply(Transform.skewY(args[0]))
            elif func == "matrix":
                if len(args) == 6:
                    result = result.multiply(Transform.matrix(*args))

        return result

    def _parse_rect(self, elem: ET.Element, style: Style, transform: Transform) -> RectElement:
        """Parse rect element."""
        # Use viewBox dimensions for percentage resolution
        return RectElement(
            tag="rect",
            style=style,
            transform=transform,
            x=self._parse_length(elem.get("x", "0"), self.viewbox_width),
            y=self._parse_length(elem.get("y", "0"), self.viewbox_height),
            width=self._parse_length(elem.get("width", "0"), self.viewbox_width),
            height=self._parse_length(elem.get("height", "0"), self.viewbox_height),
            rx=self._parse_length(elem.get("rx", "0"), self.viewbox_width),
            ry=self._parse_length(elem.get("ry", "0"), self.viewbox_height)
        )

    def _parse_circle(self, elem: ET.Element, style: Style, transform: Transform) -> CircleElement:
        """Parse circle element."""
        # Get viewport dimensions for percentage resolution
        vb_width = getattr(self, 'viewbox_width', 100) or 100
        vb_height = getattr(self, 'viewbox_height', 100) or 100
        # For radius percentage, use sqrt(vw^2 + vh^2)/sqrt(2) per SVG spec
        import math
        r_ref = math.sqrt(vb_width**2 + vb_height**2) / math.sqrt(2)

        cx_str = elem.get("cx", "0")
        cy_str = elem.get("cy", "0")
        r_str = elem.get("r", "0")

        return CircleElement(
            tag="circle",
            style=style,
            transform=transform,
            cx=self._parse_length(cx_str, vb_width if "%" in cx_str else None),
            cy=self._parse_length(cy_str, vb_height if "%" in cy_str else None),
            r=self._parse_length(r_str, r_ref if "%" in r_str else None)
        )

    def _parse_ellipse(self, elem: ET.Element, style: Style, transform: Transform) -> EllipseElement:
        """Parse ellipse element."""
        return EllipseElement(
            tag="ellipse",
            style=style,
            transform=transform,
            cx=self._parse_length(elem.get("cx", "0")),
            cy=self._parse_length(elem.get("cy", "0")),
            rx=self._parse_length(elem.get("rx", "0")),
            ry=self._parse_length(elem.get("ry", "0"))
        )

    def _parse_line(self, elem: ET.Element, style: Style, transform: Transform) -> LineElement:
        """Parse line element."""
        return LineElement(
            tag="line",
            style=style,
            transform=transform,
            x1=self._parse_length(elem.get("x1", "0")),
            y1=self._parse_length(elem.get("y1", "0")),
            x2=self._parse_length(elem.get("x2", "0")),
            y2=self._parse_length(elem.get("y2", "0"))
        )

    def _parse_points(self, points_str: str) -> list[tuple[float, float]]:
        """Parse points attribute for polyline/polygon."""
        points = []
        if not points_str:
            return points

        nums = re.split(r"[\s,]+", points_str.strip())
        for i in range(0, len(nums) - 1, 2):
            try:
                points.append((float(nums[i]), float(nums[i + 1])))
            except ValueError:
                pass
        return points

    def _parse_polyline(self, elem: ET.Element, style: Style, transform: Transform) -> PolylineElement:
        """Parse polyline element."""
        return PolylineElement(
            tag="polyline",
            style=style,
            transform=transform,
            points=self._parse_points(elem.get("points", ""))
        )

    def _parse_polygon(self, elem: ET.Element, style: Style, transform: Transform) -> PolygonElement:
        """Parse polygon element."""
        return PolygonElement(
            tag="polygon",
            style=style,
            transform=transform,
            points=self._parse_points(elem.get("points", ""))
        )

    def _parse_path(self, elem: ET.Element, style: Style, transform: Transform) -> PathElement:
        """Parse path element."""
        # Use Rust path parser if available for ~10x speedup
        try:
            import vectorstag_rust
            parse_path = vectorstag_rust.parse_path
        except ImportError:
            from .path_parser import parse_path

        d = elem.get("d", "")
        commands = list(parse_path(d))

        return PathElement(
            tag="path",
            style=style,
            transform=transform,
            commands=commands
        )

    def _parse_group(self, elem: ET.Element, style: Style, transform: Transform,
                     parent_style: Style, text_anchor: str = "start",
                     font_family: str = "Arial", font_size: float = 16,
                     depth: int = 0, ancestors: list = None) -> GroupElement:
        """Parse g (group) element."""
        if ancestors is None:
            ancestors = []
        group = GroupElement(
            tag="g",
            style=style,
            transform=transform,
            children=self._parse_children(elem, transform, style, text_anchor, font_family, font_size, depth=depth, ancestors=ancestors)
        )
        return group

    def _parse_symbol(self, elem: ET.Element, style: Style, transform: Transform,
                      parent_style: Style, text_anchor: str = "start",
                      font_family: str = "Arial", font_size: float = 16,
                      depth: int = 0, ancestors: list = None) -> GroupElement:
        """Parse symbol element - similar to group but may have viewBox."""
        if ancestors is None:
            ancestors = []
        # Symbol is like a group, but it can have its own viewBox
        # For now, treat it as a simple group
        group = GroupElement(
            tag="symbol",
            style=style,
            transform=transform,
            children=self._parse_children(elem, transform, style, text_anchor, font_family, font_size, depth=depth, ancestors=ancestors)
        )
        return group

    def _parse_nested_svg(self, elem: ET.Element, style: Style, transform: Transform,
                          parent_style: Style, text_anchor: str = "start",
                          font_family: str = "Arial", font_size: float = 16,
                          depth: int = 0, ancestors: list = None) -> GroupElement:
        """Parse nested svg element - treated as a group with its own coordinate system."""
        # Get parent viewport dimensions for percentage resolution
        parent_vb_width = getattr(self, 'viewbox_width', 100) or 100
        parent_vb_height = getattr(self, 'viewbox_height', 100) or 100

        # Get position offset - percentages relative to parent viewport
        x_str = elem.get("x", "0")
        y_str = elem.get("y", "0")
        x = self._parse_length(x_str, parent_vb_width if "%" in x_str else None)
        y = self._parse_length(y_str, parent_vb_height if "%" in y_str else None)

        # Get dimensions - percentages relative to parent viewport
        # Track whether explicit dimensions were given (for clipping decision)
        width_str = elem.get("width")
        height_str = elem.get("height")
        has_explicit_width = width_str is not None
        has_explicit_height = height_str is not None
        if width_str:
            width = self._parse_length(width_str, parent_vb_width if "%" in width_str else None)
        else:
            width = 0
        if height_str:
            height = self._parse_length(height_str, parent_vb_height if "%" in height_str else None)
        else:
            height = 0

        # Get viewBox if present
        viewbox_str = elem.get("viewBox", "")

        # Check overflow property - default is hidden for nested SVG
        overflow = elem.get("overflow", "hidden")
        style_str = elem.get("style", "")
        if "overflow" in style_str:
            style_dict = self._parse_style_string(style_str)
            if "overflow" in style_dict:
                overflow = style_dict["overflow"]

        # Build the transform for the nested SVG coordinate system
        nested_transform = transform

        # First apply position offset
        if x != 0 or y != 0:
            nested_transform = nested_transform.multiply(Transform.translate(x, y))

        # Determine viewport dimensions for clipping
        clip_width = width
        clip_height = height

        # If there's a viewBox, compute scaling to map viewBox to width/height
        if viewbox_str:
            parts = viewbox_str.replace(",", " ").split()
            if len(parts) >= 4:
                vb_x, vb_y, vb_w, vb_h = map(float, parts[:4])
                if vb_w > 0 and vb_h > 0:
                    # If no explicit dimensions, use PARENT's viewport (not viewBox)
                    # Per SVG spec: nested svg without width/height fills 100% of parent
                    if width <= 0:
                        width = parent_vb_width
                        clip_width = parent_vb_width
                    if height <= 0:
                        height = parent_vb_height
                        clip_height = parent_vb_height

                    if width > 0 and height > 0:
                        # Apply preserveAspectRatio (default: xMidYMid meet)
                        par = elem.get("preserveAspectRatio", "xMidYMid meet")
                        scale_x = width / vb_w
                        scale_y = height / vb_h

                        if "none" in par.lower():
                            # Non-uniform scaling
                            scale = None
                            offset_x, offset_y = 0, 0
                        else:
                            # Uniform scaling with alignment
                            if "meet" in par.lower():
                                scale = min(scale_x, scale_y)
                            else:  # slice
                                scale = max(scale_x, scale_y)

                            # Calculate the scaled content dimensions
                            scaled_w = vb_w * scale
                            scaled_h = vb_h * scale

                            # Calculate alignment offset
                            offset_x, offset_y = 0, 0
                            if "xmid" in par.lower():
                                offset_x = (width - scaled_w) / 2
                            elif "xmax" in par.lower():
                                offset_x = width - scaled_w

                            if "ymid" in par.lower():
                                offset_y = (height - scaled_h) / 2
                            elif "ymax" in par.lower():
                                offset_y = height - scaled_h

                            scale_x = scale_y = scale

                        # Apply transform: first offset for alignment, then scale, then viewBox origin
                        if offset_x != 0 or offset_y != 0:
                            nested_transform = nested_transform.multiply(
                                Transform.translate(offset_x, offset_y)
                            )
                        nested_transform = nested_transform.multiply(
                            Transform.scale(scale_x, scale_y)
                        ).multiply(
                            Transform.translate(-vb_x, -vb_y)
                        )

        # Create clip path for nested SVG viewport (default overflow:hidden)
        # Only clip if explicit width/height were provided (not just viewBox)
        # Per SVG spec: nested svg with only viewBox does NOT create a clipping viewport
        svg_clip_id = None
        should_clip = (overflow in ("hidden", "scroll") and
                       clip_width > 0 and clip_height > 0 and
                       has_explicit_width and has_explicit_height)
        if should_clip:
            # Create a synthetic clip path in nested SVG's coordinate system
            # The clip rect is at (0, 0) in nested coords, with viewport dimensions
            clip_id = f"_nested_svg_clip_{id(elem)}_{x}_{y}_{clip_width}_{clip_height}"
            if clip_id not in self.clip_paths:
                # Clip rect at origin in nested coordinate space
                # (the group's transform already positions content correctly)
                clip_rect = RectElement(
                    tag="rect",
                    style=Style(),
                    transform=Transform(),  # Identity - coords are in nested space
                    x=0,
                    y=0,
                    width=clip_width,
                    height=clip_height,
                    rx=0.0,
                    ry=0.0
                )
                self.clip_paths[clip_id] = ClipPath(
                    id=clip_id,
                    elements=[clip_rect]
                )
            svg_clip_id = clip_id

        # Determine the nested SVG's viewport for children's percentage resolution
        # If viewBox is specified, use viewBox dimensions; otherwise use width/height
        nested_vb_width = clip_width if clip_width > 0 else parent_vb_width
        nested_vb_height = clip_height if clip_height > 0 else parent_vb_height

        # If there's a viewBox, children's percentages should be relative to viewBox
        if viewbox_str:
            parts = viewbox_str.replace(",", " ").split()
            if len(parts) >= 4:
                nested_vb_width = float(parts[2])
                nested_vb_height = float(parts[3])

        # Temporarily update viewport context for children
        old_vb_width = getattr(self, 'viewbox_width', None)
        old_vb_height = getattr(self, 'viewbox_height', None)
        self.viewbox_width = nested_vb_width
        self.viewbox_height = nested_vb_height

        # Parse children with the nested transform
        if ancestors is None:
            ancestors = []
        children = self._parse_children(elem, nested_transform, style, text_anchor, font_family, font_size, depth=depth, ancestors=ancestors)

        # Restore parent viewport context
        self.viewbox_width = old_vb_width
        self.viewbox_height = old_vb_height

        group = GroupElement(
            tag="svg",
            style=style,
            transform=nested_transform,
            children=children
        )

        # Apply clip path to the group
        if svg_clip_id:
            group.clip_path_id = svg_clip_id

        return group

    def _parse_switch(self, elem: ET.Element, style: Style, transform: Transform,
                      parent_style: Style, text_anchor: str = "start",
                      font_family: str = "Arial", font_size: float = 16,
                      depth: int = 0, ancestors: list = None) -> Optional[SVGElement]:
        """Parse switch element - returns first supported child."""
        # The switch element evaluates children in order and renders the first
        # one whose requiredExtensions/requiredFeatures are supported.
        # Since we don't support any extensions, skip children with requiredExtensions.
        for child in elem:
            tag = self._strip_ns(child.tag)
            # Skip non-element nodes
            if tag is None:
                continue
            # Skip foreignObject and elements with requiredExtensions we don't support
            if tag == "foreignObject":
                continue
            if child.get("requiredExtensions"):
                # We don't support any extensions
                continue
            # This child is supported - parse and return it
            if ancestors is None:
                ancestors = []
            result = self._parse_element(child, transform, style, text_anchor, font_family, font_size, depth=depth, ancestors=ancestors)
            if result:
                return result
        return None

    def _parse_text(self, elem: ET.Element, style: Style, transform: Transform,
                    parent_text_anchor: str = "start", parent_font_family: str = "Arial",
                    parent_font_size: float = 16) -> TextElement:
        """Parse text element."""
        # Check for textPath child
        text_path_href = None
        text_path_start_offset = 0
        text_path_data = None

        for child in elem:
            child_tag = self._strip_ns(child.tag)
            if child_tag == "textPath":
                # Get href reference to path element
                href = child.get(f"{self.XLINK_NS}href") or child.get("href")
                if href and href.startswith("#"):
                    text_path_href = href[1:]

                # Get direct path data (SVG 2)
                path_attr = child.get("path")
                if path_attr:
                    text_path_data = path_attr

                # Get startOffset
                start_offset = child.get("startOffset", "0")
                if start_offset.endswith("%"):
                    text_path_start_offset = float(start_offset[:-1]) / 100.0
                else:
                    text_path_start_offset = self._parse_length(start_offset)

        # Get text content - from textPath child if present, otherwise from text element
        text = ""
        for child in elem:
            child_tag = self._strip_ns(child.tag)
            if child_tag == "textPath":
                text = child.text or ""
                # Also get text from tspan children within textPath
                for subchild in child:
                    if subchild.text:
                        text += subchild.text
                    if subchild.tail:
                        text += subchild.tail
                break
        else:
            # No textPath - get text directly
            text = elem.text or ""
            for child in elem:
                if child.tail:
                    text += child.tail

        # Font properties can come from element, style, or parent
        font_family = elem.get("font-family")
        font_size_str = elem.get("font-size")

        # Check inline style
        style_str = elem.get("style", "")
        if "font-family" in style_str and not font_family:
            for part in style_str.split(";"):
                if "font-family" in part:
                    key, _, value = part.partition(":")
                    if key.strip() == "font-family":
                        font_family = value.strip()
        if "font-size" in style_str and not font_size_str:
            for part in style_str.split(";"):
                if "font-size" in part:
                    key, _, value = part.partition(":")
                    if key.strip() == "font-size":
                        font_size_str = value.strip()

        # Use parent values as fallback
        if not font_family:
            font_family = parent_font_family
        if font_size_str:
            font_size = self._parse_length(font_size_str)
        else:
            font_size = parent_font_size

        # Get text-anchor - check element, then style, then inherit from parent
        text_anchor = elem.get("text-anchor")
        if not text_anchor:
            # Check inline style for text-anchor
            style_str = elem.get("style", "")
            if "text-anchor" in style_str:
                for part in style_str.split(";"):
                    if "text-anchor" in part:
                        key, _, value = part.partition(":")
                        if key.strip() == "text-anchor":
                            text_anchor = value.strip()
        if not text_anchor:
            text_anchor = parent_text_anchor

        return TextElement(
            tag="text",
            style=style,
            transform=transform,
            x=self._parse_length(elem.get("x", "0")),
            y=self._parse_length(elem.get("y", "0")),
            text=text.strip(),
            font_family=font_family,
            font_size=font_size,
            text_anchor=text_anchor,
            text_path_href=text_path_href,
            text_path_start_offset=text_path_start_offset,
            text_path_data=text_path_data
        )

    def _parse_image(self, elem: ET.Element, style: Style, transform: Transform) -> ImageElement:
        """Parse image element."""
        # Get href from xlink:href or href attribute
        href = elem.get(f"{self.XLINK_NS}href") or elem.get("href") or ""

        return ImageElement(
            tag="image",
            style=style,
            transform=transform,
            x=self._parse_length(elem.get("x", "0")),
            y=self._parse_length(elem.get("y", "0")),
            width=self._parse_length(elem.get("width", "0")),
            height=self._parse_length(elem.get("height", "0")),
            href=href,
            preserveAspectRatio=elem.get("preserveAspectRatio", "xMidYMid meet"),
            base_path=getattr(self, 'base_path', '')
        )

    def _parse_use(self, elem: ET.Element, style: Style, transform: Transform,
                   parent_style: Style, depth: int = 0) -> Optional[SVGElement]:
        """Parse use element (reference to another element)."""
        # Get the referenced element ID
        href = elem.get(f"{self.XLINK_NS}href") or elem.get("href")
        if not href:
            return None

        if href.startswith("#"):
            href = href[1:]

        # Detect circular references - if we're already parsing this element, skip
        if href in self._use_stack:
            return None

        # Find the referenced element in defs
        ref_elem = self.defs.get(href)
        if ref_elem is None:
            return None

        # Get x, y offset for the use element
        x = self._parse_length(elem.get("x", "0"))
        y = self._parse_length(elem.get("y", "0"))

        # Get use element's width/height (for symbol viewBox)
        use_width_str = elem.get("width")
        use_height_str = elem.get("height")
        use_width = self._parse_length(use_width_str) if use_width_str else None
        use_height = self._parse_length(use_height_str) if use_height_str else None

        # Apply translation for x, y
        if x != 0 or y != 0:
            use_transform = transform.multiply(Transform.translate(x, y))
        else:
            use_transform = transform

        # Check if referenced element is a symbol or svg element
        ref_tag = self._strip_ns(ref_elem.tag)
        symbol_clip_id = None
        if ref_tag == "symbol" or ref_tag == "svg":
            viewbox_str = ref_elem.get("viewBox", "")
            if viewbox_str:
                # Symbol with viewBox - scale content to fit
                parts = viewbox_str.replace(",", " ").split()
                if len(parts) >= 4:
                    vb_x, vb_y, vb_w, vb_h = map(float, parts[:4])
                    if vb_w > 0 and vb_h > 0:
                        # Determine target size
                        # If use has width/height, use those
                        # Otherwise, use the parent SVG dimensions
                        target_w = use_width if use_width is not None else self.viewbox_width
                        target_h = use_height if use_height is not None else self.viewbox_height

                        # Fall back to viewBox dimensions if still no size
                        if target_w <= 0:
                            target_w = vb_w
                        if target_h <= 0:
                            target_h = vb_h

                        # Handle preserveAspectRatio
                        par = ref_elem.get("preserveAspectRatio", "xMidYMid meet")
                        if par == "none":
                            scale_x = target_w / vb_w
                            scale_y = target_h / vb_h
                            offset_x = -vb_x * scale_x
                            offset_y = -vb_y * scale_y
                        else:
                            # Calculate aspect-ratio-preserving scale
                            aspect_vb = vb_w / vb_h
                            aspect_target = target_w / target_h

                            if "meet" in par or ("slice" not in par and "meet" not in par):
                                # meet: scale to fit within bounds
                                if aspect_vb > aspect_target:
                                    scale = target_w / vb_w
                                else:
                                    scale = target_h / vb_h
                            else:
                                # slice: scale to cover bounds
                                if aspect_vb > aspect_target:
                                    scale = target_h / vb_h
                                else:
                                    scale = target_w / vb_w

                            scale_x = scale
                            scale_y = scale
                            scaled_w = vb_w * scale
                            scaled_h = vb_h * scale

                            # Calculate alignment offset
                            x_offset = 0
                            y_offset = 0

                            if "xMid" in par:
                                x_offset = (target_w - scaled_w) / 2
                            elif "xMax" in par:
                                x_offset = target_w - scaled_w

                            if "YMid" in par:
                                y_offset = (target_h - scaled_h) / 2
                            elif "YMax" in par:
                                y_offset = target_h - scaled_h

                            offset_x = x_offset - vb_x * scale
                            offset_y = y_offset - vb_y * scale

                        # Apply viewBox transform
                        # Order: first scale, then translate (for correct viewBox mapping)
                        use_transform = use_transform.multiply(
                            Transform.scale(scale_x, scale_y)
                        ).multiply(
                            Transform.translate(-vb_x, -vb_y)
                        )
                        # Apply alignment offset
                        if "xMid" in par:
                            align_x = (target_w - vb_w * scale_x) / 2
                            use_transform = use_transform.multiply(Transform.translate(align_x / scale_x, 0))
                        elif "xMax" in par:
                            align_x = target_w - vb_w * scale_x
                            use_transform = use_transform.multiply(Transform.translate(align_x / scale_x, 0))
                        if "YMid" in par:
                            align_y = (target_h - vb_h * scale_y) / 2
                            use_transform = use_transform.multiply(Transform.translate(0, align_y / scale_y))
                        elif "YMax" in par:
                            align_y = target_h - vb_h * scale_y
                            use_transform = use_transform.multiply(Transform.translate(0, align_y / scale_y))

                        # Symbols clip content to the use dimensions (overflow:hidden by default)
                        overflow = ref_elem.get("overflow", "hidden")
                        if overflow == "hidden" or overflow == "scroll":
                            clip_id = f"_symbol_clip_{id(ref_elem)}_{target_w}_{target_h}"
                            if clip_id not in self.clip_paths:
                                clip_rect = RectElement(
                                    tag="rect",
                                    style=Style(),
                                    transform=Transform.identity(),
                                    x=0, y=0,
                                    width=target_w, height=target_h,
                                    rx=0, ry=0
                                )
                                self.clip_paths[clip_id] = ClipPath(
                                    id=clip_id,
                                    elements=[clip_rect]
                                )
                            symbol_clip_id = clip_id
            elif use_width is not None and use_height is not None:
                # Symbol without viewBox but use has width/height
                # Content is clipped (not scaled) to use dimensions
                # Check overflow property (default is hidden for symbols)
                overflow = ref_elem.get("overflow", "hidden")
                if overflow in ("hidden", "scroll"):
                    # Create a synthetic clip path
                    clip_id = f"_symbol_clip_{id(ref_elem)}_{use_width}_{use_height}"
                    if clip_id not in self.clip_paths:
                        clip_rect = RectElement(
                            tag="rect",
                            style=Style(),
                            transform=Transform.identity(),
                            x=0, y=0,
                            width=use_width, height=use_height,
                            rx=0, ry=0
                        )
                        self.clip_paths[clip_id] = ClipPath(
                            id=clip_id,
                            elements=[clip_rect]
                        )
                    symbol_clip_id = clip_id

        # Track this reference to detect circular dependencies
        self._use_stack.add(href)
        try:
            # Parse the referenced element with the combined transform
            result = self._parse_element(ref_elem, use_transform, style, depth=depth)
            # Apply symbol clip path if needed
            if result and symbol_clip_id:
                result.clip_path_id = symbol_clip_id
            return result
        finally:
            self._use_stack.discard(href)

    def _compute_elements_bbox(self, elements: list) -> Optional[tuple[float, float, float, float]]:
        """Compute bounding box of all elements."""
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for elem in elements:
            bbox = self._compute_element_bbox(elem)
            if bbox:
                min_x = min(min_x, bbox[0])
                min_y = min(min_y, bbox[1])
                max_x = max(max_x, bbox[2])
                max_y = max(max_y, bbox[3])

        if min_x == float('inf'):
            return None

        return (min_x, min_y, max_x, max_y)

    def _compute_element_bbox(self, elem) -> Optional[tuple[float, float, float, float]]:
        """Compute bounding box of a single element."""
        if isinstance(elem, GroupElement):
            return self._compute_elements_bbox(elem.children)

        # Get base bounding box before transform
        bbox = None

        if isinstance(elem, RectElement):
            bbox = (elem.x, elem.y, elem.x + elem.width, elem.y + elem.height)
        elif isinstance(elem, CircleElement):
            bbox = (elem.cx - elem.r, elem.cy - elem.r,
                    elem.cx + elem.r, elem.cy + elem.r)
        elif isinstance(elem, EllipseElement):
            bbox = (elem.cx - elem.rx, elem.cy - elem.ry,
                    elem.cx + elem.rx, elem.cy + elem.ry)
        elif isinstance(elem, LineElement):
            bbox = (min(elem.x1, elem.x2), min(elem.y1, elem.y2),
                    max(elem.x1, elem.x2), max(elem.y1, elem.y2))
        elif isinstance(elem, (PolygonElement, PolylineElement)):
            if elem.points:
                xs = [p[0] for p in elem.points]
                ys = [p[1] for p in elem.points]
                bbox = (min(xs), min(ys), max(xs), max(ys))
        elif isinstance(elem, PathElement):
            bbox = self._compute_path_bbox(elem.commands)
        elif isinstance(elem, TextElement):
            # Rough estimate for text
            text_width = len(elem.text) * elem.font_size * 0.6
            bbox = (elem.x, elem.y - elem.font_size, elem.x + text_width, elem.y)

        if not bbox:
            return None

        # Apply transform to bounding box corners
        corners = [
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3])
        ]

        transformed = [elem.transform.apply(x, y) for x, y in corners]
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]

        # Expand bbox for stroke width
        stroke_expand = elem.style.stroke_width / 2 if elem.style.stroke else 0

        return (min(xs) - stroke_expand, min(ys) - stroke_expand,
                max(xs) + stroke_expand, max(ys) + stroke_expand)

    def _compute_path_bbox(self, commands: list) -> Optional[tuple[float, float, float, float]]:
        """Compute bounding box from path commands using actual curve extrema."""
        if not commands:
            return None

        points = []
        current = None
        for cmd in commands:
            if cmd[0] == 'M':
                current = (cmd[1], cmd[2])
                points.append(current)
            elif cmd[0] == 'L':
                current = (cmd[1], cmd[2])
                points.append(current)
            elif cmd[0] == 'C':
                # Compute actual cubic bezier extrema (not conservative control point hull)
                p0 = current if current else (0, 0)
                p1 = (cmd[1], cmd[2])
                p2 = (cmd[3], cmd[4])
                p3 = (cmd[5], cmd[6])
                points.extend(self._cubic_bezier_extrema(p0, p1, p2, p3))
                current = p3
            elif cmd[0] == 'Q':
                # Compute actual quadratic bezier extrema
                p0 = current if current else (0, 0)
                p1 = (cmd[1], cmd[2])
                p2 = (cmd[3], cmd[4])
                points.extend(self._quadratic_bezier_extrema(p0, p1, p2))
                current = p2

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def _cubic_bezier_extrema(self, p0: tuple, p1: tuple, p2: tuple, p3: tuple) -> list:
        """Compute actual extrema points of cubic bezier curve.

        For a cubic bezier B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
        Extrema occur at t=0, t=1, or where derivative B'(t)=0.
        """
        import math
        points = [p0, p3]  # Always include endpoints

        # For each dimension, find t values where derivative is 0
        for dim in [0, 1]:
            v0, v1, v2, v3 = p0[dim], p1[dim], p2[dim], p3[dim]

            # B'(t) = at^2 + bt + c where:
            c = 3 * (v1 - v0)
            b = -6 * (v1 - v0) + 6 * (v2 - v1)
            a = 3 * (v1 - v0) - 6 * (v2 - v1) + 3 * (v3 - v2)

            # Solve quadratic at^2 + bt + c = 0
            if abs(a) < 1e-10:
                if abs(b) > 1e-10:
                    t = -c / b
                    if 0 < t < 1:
                        points.append(self._eval_cubic(t, p0, p1, p2, p3))
            else:
                disc = b*b - 4*a*c
                if disc >= 0:
                    sqrt_disc = math.sqrt(disc)
                    for t in [(-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)]:
                        if 0 < t < 1:
                            points.append(self._eval_cubic(t, p0, p1, p2, p3))

        return points

    def _eval_cubic(self, t: float, p0: tuple, p1: tuple, p2: tuple, p3: tuple) -> tuple:
        """Evaluate cubic bezier at parameter t."""
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        return (x, y)

    def _quadratic_bezier_extrema(self, p0: tuple, p1: tuple, p2: tuple) -> list:
        """Compute actual extrema points of quadratic bezier curve."""
        points = [p0, p2]  # Always include endpoints

        # For quadratic B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
        # B'(t) = 2*(1-t)*(P1-P0) + 2*t*(P2-P1) = 2*(P1-P0) + 2*t*(P2-2*P1+P0)
        # Extremum at t = (P0-P1) / (P0 - 2*P1 + P2)
        for dim in [0, 1]:
            v0, v1, v2 = p0[dim], p1[dim], p2[dim]
            denom = v0 - 2*v1 + v2
            if abs(denom) > 1e-10:
                t = (v0 - v1) / denom
                if 0 < t < 1:
                    mt = 1 - t
                    x = mt**2 * p0[0] + 2 * mt * t * p1[0] + t**2 * p2[0]
                    y = mt**2 * p0[1] + 2 * mt * t * p1[1] + t**2 * p2[1]
                    points.append((x, y))

        return points
