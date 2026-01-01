"""SVG filter primitive definitions."""

from dataclasses import dataclass, field
from typing import Optional


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


# Light source classes
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
    # Track if coordinates are percentages (for userSpaceOnUse viewport conversion)
    x_pct: bool = True
    y_pct: bool = True
    width_pct: bool = True
    height_pct: bool = True
