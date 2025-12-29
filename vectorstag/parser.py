"""SVG Parser - Parse SVG documents into a renderable structure."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Union
import math


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


@dataclass
class RadialGradient:
    """Radial gradient definition."""
    id: str
    cx: float = 0.5
    cy: float = 0.5
    r: float = 0.5
    fx: Optional[float] = None
    fy: Optional[float] = None
    stops: list[GradientStop] = field(default_factory=list)
    units: str = "objectBoundingBox"
    transform: Optional[Transform] = None
    href: Optional[str] = None


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


@dataclass
class SVGElement:
    """Base class for SVG elements."""
    tag: str
    style: Style
    transform: Transform
    children: list["SVGElement"] = field(default_factory=list)
    clip_path_id: Optional[str] = None
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


@dataclass
class ClipPath:
    """Clip path definition."""
    id: str
    elements: list[SVGElement] = field(default_factory=list)
    clip_path_id: Optional[str] = None  # For nested clip paths (intersection)


@dataclass
class GaussianBlurFilter:
    """Gaussian blur filter definition."""
    id: str
    std_deviation: float = 2.0


@dataclass
class SVGDocument:
    """Parsed SVG document."""
    width: float
    height: float
    viewBox: Optional[tuple[float, float, float, float]] = None
    elements: list[SVGElement] = field(default_factory=list)
    gradients: dict[str, Union[LinearGradient, RadialGradient]] = field(default_factory=dict)
    clip_paths: dict[str, ClipPath] = field(default_factory=dict)
    filters: dict[str, GaussianBlurFilter] = field(default_factory=dict)
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

    def __init__(self):
        self.gradients: dict[str, Union[LinearGradient, RadialGradient]] = {}
        self.clip_paths: dict[str, ClipPath] = {}
        self.defs: dict[str, ET.Element] = {}
        self.default_width = 300
        self.default_height = 150
        # ViewBox dimensions for percentage resolution
        self.viewbox_width = 0
        self.viewbox_height = 0
        # CSS classes from <style> blocks
        self.css_classes: dict[str, dict[str, str]] = {}

    def parse(self, svg_content: str) -> SVGDocument:
        """Parse SVG content string into SVGDocument."""
        # Remove any XML declaration issues
        svg_content = svg_content.strip()

        # Parse XML
        root = ET.fromstring(svg_content)

        # Get dimensions
        width = self._parse_length(root.get("width", str(self.default_width)))
        height = self._parse_length(root.get("height", str(self.default_height)))

        # Parse viewBox
        viewBox = None
        viewbox_str = root.get("viewBox")
        if viewbox_str:
            parts = re.split(r"[\s,]+", viewbox_str.strip())
            if len(parts) == 4:
                viewBox = tuple(float(p) for p in parts)
                # Store viewBox dimensions for percentage resolution
                self.viewbox_width = viewBox[2]
                self.viewbox_height = viewBox[3]
                # If width/height not specified, use viewBox dimensions
                if not root.get("width"):
                    width = viewBox[2]
                if not root.get("height"):
                    height = viewBox[3]
        else:
            # No viewBox - use width/height for percentage resolution
            self.viewbox_width = width
            self.viewbox_height = height

        # Parse preserveAspectRatio attribute
        preserve_aspect_ratio = root.get("preserveAspectRatio", "xMidYMid")
        # Strip optional "meet" or "slice" suffix
        preserve_aspect_ratio = preserve_aspect_ratio.split()[0] if preserve_aspect_ratio else "xMidYMid"

        # Reset state
        self.gradients = {}
        self.clip_paths = {}
        self.filters = {}
        self.defs = {}
        self.css_classes = {}

        # Parse CSS from <style> blocks
        self._parse_css_styles(root)

        # First pass: collect defs
        self._collect_defs(root)

        # Parse clip paths
        self._parse_clip_paths(root)

        # Resolve gradient references
        self._resolve_gradient_refs()

        # Parse root element style (for inherited properties like stroke-width)
        root_style = self._parse_style(root, Style())

        # Parse elements
        elements = self._parse_children(root, Transform.identity(), root_style)

        # If no explicit dimensions and no viewBox, compute from content
        has_explicit_width = root.get("width") is not None
        has_explicit_height = root.get("height") is not None

        if not has_explicit_width or not has_explicit_height or (width == 0 or height == 0):
            if not viewBox:
                # Compute bounding box from elements
                bbox = self._compute_elements_bbox(elements)
                if bbox:
                    min_x, min_y, max_x, max_y = bbox

                    # Set dimensions to fit content (with small padding)
                    padding = 5
                    if not has_explicit_width or width == 0:
                        width = max(max_x + padding, self.default_width)
                    if not has_explicit_height or height == 0:
                        height = max(max_y + padding, self.default_height)

        return SVGDocument(
            width=width,
            height=height,
            viewBox=viewBox,
            elements=elements,
            gradients=self.gradients,
            clip_paths=self.clip_paths,
            filters=self.filters,
            preserve_aspect_ratio=preserve_aspect_ratio
        )

    def parse_file(self, filepath: str) -> SVGDocument:
        """Parse SVG file."""
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
        """Parse CSS text and extract class rules."""
        # Simple CSS parser for class rules: .classname { property: value; }
        # Remove comments
        css_text = re.sub(r'/\*.*?\*/', '', css_text, flags=re.DOTALL)

        # Find class rules
        pattern = r'\.([a-zA-Z_][a-zA-Z0-9_-]*)\s*\{([^}]*)\}'
        for match in re.finditer(pattern, css_text):
            class_name = match.group(1)
            properties_str = match.group(2)

            # Parse properties
            properties = {}
            for prop in properties_str.split(';'):
                prop = prop.strip()
                if ':' in prop:
                    key, value = prop.split(':', 1)
                    properties[key.strip()] = value.strip()

            self.css_classes[class_name] = properties

    def _collect_defs(self, root: ET.Element):
        """Collect all definitions (gradients, filters, etc.)."""
        for elem in root.iter():
            tag = self._strip_ns(elem.tag)
            elem_id = elem.get("id")

            if elem_id:
                self.defs[elem_id] = elem

            if tag == "linearGradient":
                self._parse_linear_gradient(elem)
            elif tag == "radialGradient":
                self._parse_radial_gradient(elem)
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
                            child, Transform.identity(), Style()
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

    def _parse_filter(self, elem: ET.Element):
        """Parse a filter element (basic support for Gaussian blur)."""
        filter_id = elem.get("id")
        if not filter_id:
            return

        # Look for feGaussianBlur child
        for child in elem:
            tag = self._strip_ns(child.tag)
            if tag == "feGaussianBlur":
                std_dev = child.get("stdDeviation", "2")
                try:
                    # stdDeviation can be "x y" for separate x/y, we'll use the first
                    std_dev_val = float(std_dev.split()[0])
                except (ValueError, IndexError):
                    std_dev_val = 2.0

                self.filters[filter_id] = GaussianBlurFilter(
                    id=filter_id,
                    std_deviation=std_dev_val
                )
                break  # Only handle the first blur for now

    def _parse_linear_gradient(self, elem: ET.Element):
        """Parse a linearGradient element."""
        grad_id = elem.get("id")
        if not grad_id:
            return

        # Get href for inheritance
        href = elem.get(f"{self.XLINK_NS}href") or elem.get("href")
        if href and href.startswith("#"):
            href = href[1:]

        # Validate gradientUnits - fall back to default for invalid values
        units = elem.get("gradientUnits", "objectBoundingBox")
        if units not in ("objectBoundingBox", "userSpaceOnUse"):
            units = "objectBoundingBox"

        grad = LinearGradient(
            id=grad_id,
            x1=self._parse_gradient_coord(elem.get("x1", "0%")),
            y1=self._parse_gradient_coord(elem.get("y1", "0%")),
            x2=self._parse_gradient_coord(elem.get("x2", "100%")),
            y2=self._parse_gradient_coord(elem.get("y2", "0%")),
            units=units,
            href=href
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

        # Validate gradientUnits - fall back to default for invalid values
        units = elem.get("gradientUnits", "objectBoundingBox")
        if units not in ("objectBoundingBox", "userSpaceOnUse"):
            units = "objectBoundingBox"

        grad = RadialGradient(
            id=grad_id,
            cx=self._parse_gradient_coord(elem.get("cx", "50%")),
            cy=self._parse_gradient_coord(elem.get("cy", "50%")),
            r=self._parse_gradient_coord(elem.get("r", "50%")),
            units=units,
            href=href
        )

        fx = elem.get("fx")
        fy = elem.get("fy")
        if fx:
            grad.fx = self._parse_gradient_coord(fx)
        if fy:
            grad.fy = self._parse_gradient_coord(fy)

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
        return float(value)

    def _parse_gradient_stops(self, elem: ET.Element) -> list[GradientStop]:
        """Parse gradient stop elements."""
        stops = []
        for child in elem:
            tag = self._strip_ns(child.tag)
            if tag == "stop":
                offset = child.get("offset", "0")
                if offset.endswith("%"):
                    offset = float(offset[:-1]) / 100.0
                else:
                    offset = float(offset)

                # Get color from style or attributes
                style_str = child.get("style", "")
                style_dict = self._parse_style_string(style_str)

                color_str = style_dict.get("stop-color") or child.get("stop-color", "black")
                opacity_str = style_dict.get("stop-opacity") or child.get("stop-opacity", "1")

                color = self._parse_color(color_str)
                if color:
                    opacity = float(opacity_str)
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

    def _parse_children(self, parent: ET.Element, parent_transform: Transform,
                        parent_style: Style, parent_text_anchor: str = "start",
                        parent_font_family: str = "Arial", parent_font_size: float = 16) -> list[SVGElement]:
        """Parse child elements."""
        elements = []

        for child in parent:
            tag = self._strip_ns(child.tag)

            # Skip defs, metadata, etc.
            if tag in ("defs", "metadata", "title", "desc", "style",
                       "linearGradient", "radialGradient", "SVGTestCase",
                       "OperatorScript", "Paragraph"):
                continue

            elem = self._parse_element(child, parent_transform, parent_style, parent_text_anchor,
                                       parent_font_family, parent_font_size)
            if elem:
                elements.append(elem)

        return elements

    def _parse_element(self, elem: ET.Element, parent_transform: Transform,
                       parent_style: Style, parent_text_anchor: str = "start",
                       parent_font_family: str = "Arial", parent_font_size: float = 16) -> Optional[SVGElement]:
        """Parse a single SVG element."""
        tag = self._strip_ns(elem.tag)

        # Parse style (inheriting from parent)
        style = self._parse_style(elem, parent_style)

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

        # Parse transform
        transform_str = elem.get("transform", "")
        local_transform = self._parse_transform(transform_str) if transform_str else Transform.identity()
        transform = parent_transform.multiply(local_transform)

        # Parse clip-path attribute
        clip_path_id = self._parse_url_reference(elem.get("clip-path", ""))

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
        elif tag == "g":
            result = self._parse_group(elem, style, transform, parent_style, text_anchor, font_family, font_size)
        elif tag == "switch":
            result = self._parse_switch(elem, style, transform, parent_style, text_anchor, font_family, font_size)
        elif tag == "text":
            result = self._parse_text(elem, style, transform, text_anchor, font_family, font_size)
        elif tag == "use":
            result = self._parse_use(elem, style, transform, parent_style)

        # Set clip path on the parsed element
        if result and clip_path_id:
            result.clip_path_id = clip_path_id

        return result

    def _parse_url_reference(self, value: str) -> Optional[str]:
        """Parse a url() reference and return the ID."""
        if not value:
            return None
        value = value.strip()
        if value.startswith("url(") and value.endswith(")"):
            ref = value[4:-1].strip()
            if ref.startswith("#"):
                return ref[1:]
            return ref
        return None

    def _parse_style(self, elem: ET.Element, parent_style: Style) -> Style:
        """Parse style from element attributes and style attribute."""
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
            display=parent_style.display
        )

        # First apply CSS classes (lowest priority)
        style_dict = {}
        class_attr = elem.get("class", "")
        if class_attr:
            for class_name in class_attr.split():
                if class_name in self.css_classes:
                    style_dict.update(self.css_classes[class_name])

        # Parse style attribute (higher priority - overrides CSS classes)
        style_str = elem.get("style", "")
        style_dict.update(self._parse_style_string(style_str))

        # Merge with direct attributes (highest priority)
        for attr in ["fill", "stroke", "stroke-width", "fill-opacity",
                     "stroke-opacity", "opacity", "fill-rule",
                     "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "stroke-dasharray", "filter", "display"]:
            val = elem.get(attr)
            if val:
                style_dict[attr] = val

        # Apply parsed values
        if "fill" in style_dict:
            fill_val = style_dict["fill"]
            if fill_val == "none":
                style.fill = None
            elif fill_val.startswith("url("):
                # Gradient reference
                style.fill = fill_val
            else:
                color = self._parse_color(fill_val)
                if color:
                    style.fill = (*color, 255)

        if "stroke" in style_dict:
            stroke_val = style_dict["stroke"]
            if stroke_val == "none":
                style.stroke = None
            elif stroke_val.startswith("url("):
                style.stroke = stroke_val
            else:
                color = self._parse_color(stroke_val)
                if color:
                    style.stroke = (*color, 255)

        if "stroke-width" in style_dict:
            style.stroke_width = self._parse_length(style_dict["stroke-width"])

        if "fill-opacity" in style_dict:
            style.fill_opacity = float(style_dict["fill-opacity"])

        if "stroke-opacity" in style_dict:
            style.stroke_opacity = float(style_dict["stroke-opacity"])

        if "opacity" in style_dict:
            style.opacity = float(style_dict["opacity"])

        if "fill-rule" in style_dict:
            style.fill_rule = style_dict["fill-rule"]

        if "stroke-linecap" in style_dict:
            style.stroke_linecap = style_dict["stroke-linecap"]

        if "stroke-linejoin" in style_dict:
            style.stroke_linejoin = style_dict["stroke-linejoin"]

        if "stroke-miterlimit" in style_dict:
            style.stroke_miterlimit = float(style_dict["stroke-miterlimit"])

        if "stroke-dasharray" in style_dict:
            dasharray_str = style_dict["stroke-dasharray"].strip()
            if dasharray_str and dasharray_str.lower() != "none":
                # Parse comma or space separated values
                parts = dasharray_str.replace(",", " ").split()
                try:
                    style.stroke_dasharray = [float(p) for p in parts if p]
                except ValueError:
                    pass

        if "display" in style_dict:
            # Only allow display override if parent is NOT display:none
            # CSS spec: children of display:none elements are never rendered
            if parent_style.display != "none":
                style.display = style_dict["display"]

        # Parse filter reference
        if "filter" in style_dict:
            filter_val = style_dict["filter"]
            if filter_val.startswith("url(#") and filter_val.endswith(")"):
                style.filter_id = filter_val[5:-1]

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

        # Named color
        if color_str in self.COLORS:
            return self.COLORS[color_str]

        # Hex color
        if color_str.startswith("#"):
            hex_str = color_str[1:]
            if len(hex_str) == 3:
                # Short form #RGB
                r = int(hex_str[0] * 2, 16)
                g = int(hex_str[1] * 2, 16)
                b = int(hex_str[2] * 2, 16)
                return (r, g, b)
            elif len(hex_str) == 6:
                # Full form #RRGGBB
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b)

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

        return None

    def _parse_length(self, length_str: str, ref_dim: float = None) -> float:
        """Parse SVG length value.

        Args:
            length_str: The length string (e.g., "100", "50%", "10px")
            ref_dim: Reference dimension for percentage values (e.g., viewBox width/height)
        """
        if not length_str:
            return 0

        length_str = length_str.strip()

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
        }

        for unit, factor in units.items():
            if length_str.endswith(unit):
                return float(length_str[:-len(unit)]) * factor

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
        return CircleElement(
            tag="circle",
            style=style,
            transform=transform,
            cx=self._parse_length(elem.get("cx", "0")),
            cy=self._parse_length(elem.get("cy", "0")),
            r=self._parse_length(elem.get("r", "0"))
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
        from .path_parser import parse_path

        d = elem.get("d", "")
        commands = parse_path(d)

        return PathElement(
            tag="path",
            style=style,
            transform=transform,
            commands=commands
        )

    def _parse_group(self, elem: ET.Element, style: Style, transform: Transform,
                     parent_style: Style, text_anchor: str = "start",
                     font_family: str = "Arial", font_size: float = 16) -> GroupElement:
        """Parse g (group) element."""
        group = GroupElement(
            tag="g",
            style=style,
            transform=transform,
            children=self._parse_children(elem, transform, style, text_anchor, font_family, font_size)
        )
        return group

    def _parse_switch(self, elem: ET.Element, style: Style, transform: Transform,
                      parent_style: Style, text_anchor: str = "start",
                      font_family: str = "Arial", font_size: float = 16) -> Optional[SVGElement]:
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
            result = self._parse_element(child, transform, style, text_anchor, font_family, font_size)
            if result:
                return result
        return None

    def _parse_text(self, elem: ET.Element, style: Style, transform: Transform,
                    parent_text_anchor: str = "start", parent_font_family: str = "Arial",
                    parent_font_size: float = 16) -> TextElement:
        """Parse text element."""
        # Get text content
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
            text_anchor=text_anchor
        )

    def _parse_use(self, elem: ET.Element, style: Style, transform: Transform,
                   parent_style: Style) -> Optional[SVGElement]:
        """Parse use element (reference to another element)."""
        # Get the referenced element ID
        href = elem.get(f"{self.XLINK_NS}href") or elem.get("href")
        if not href:
            return None

        if href.startswith("#"):
            href = href[1:]

        # Find the referenced element in defs
        ref_elem = self.defs.get(href)
        if ref_elem is None:
            return None

        # Get x, y offset for the use element
        x = self._parse_length(elem.get("x", "0"))
        y = self._parse_length(elem.get("y", "0"))

        # Apply translation for x, y
        if x != 0 or y != 0:
            use_transform = transform.multiply(Transform.translate(x, y))
        else:
            use_transform = transform

        # Parse the referenced element with the combined transform
        return self._parse_element(ref_elem, use_transform, style)

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
        """Compute bounding box from path commands."""
        if not commands:
            return None

        points = []
        for cmd in commands:
            if cmd[0] == 'M':
                points.append((cmd[1], cmd[2]))
            elif cmd[0] == 'L':
                points.append((cmd[1], cmd[2]))
            elif cmd[0] == 'C':
                # Include control points for conservative bbox
                points.append((cmd[1], cmd[2]))
                points.append((cmd[3], cmd[4]))
                points.append((cmd[5], cmd[6]))
            elif cmd[0] == 'Q':
                points.append((cmd[1], cmd[2]))
                points.append((cmd[3], cmd[4]))

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))
