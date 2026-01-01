"""SVG gradient and pattern definitions."""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.transforms import Transform


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
    transform: Optional["Transform"] = None
    href: Optional[str] = None  # Reference to another gradient
    spread_method: str = "pad"  # pad, reflect, repeat
    # Flags for percentage coordinates (for userSpaceOnUse)
    x1_pct: bool = True
    y1_pct: bool = True
    x2_pct: bool = True
    y2_pct: bool = True


@dataclass
class RadialGradient:
    """Radial gradient definition."""
    id: str
    cx: float = 0.5
    cy: float = 0.5
    r: float = 0.5
    fx: Optional[float] = None
    fy: Optional[float] = None
    fr: float = 0.0  # focal radius
    stops: list[GradientStop] = field(default_factory=list)
    units: str = "objectBoundingBox"
    transform: Optional["Transform"] = None
    href: Optional[str] = None
    spread_method: str = "pad"  # pad, reflect, repeat
    # Flags for percentage coordinates (for userSpaceOnUse)
    cx_pct: bool = True
    cy_pct: bool = True
    r_pct: bool = True
    fx_pct: bool = False
    fy_pct: bool = False
    fr_pct: bool = False


@dataclass
class Pattern:
    """Pattern definition for tiled fills."""
    id: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    pattern_units: str = "objectBoundingBox"
    pattern_content_units: str = "userSpaceOnUse"
    transform: Optional["Transform"] = None
    href: Optional[str] = None  # Reference to another pattern
    viewbox: Optional[tuple[float, float, float, float]] = None
    elements: list = field(default_factory=list)  # Child elements to render
