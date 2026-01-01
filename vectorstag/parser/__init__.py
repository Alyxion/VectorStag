"""VectorStag SVG parser package.

This package contains the dataclass definitions for SVG elements.
The main SVGParser class is in vectorstag.parser (top-level module).
"""

# Re-export all dataclasses for backward compatibility
from .elements import (
    SVGElement, RectElement, CircleElement, EllipseElement,
    LineElement, PolylineElement, PolygonElement, PathElement,
    GroupElement, TextElement, ImageElement,
    ClipPath, Mask, SVGDocument,
)
from .styles import Style, FILL_NOT_SET
from .gradients import GradientStop, LinearGradient, RadialGradient, Pattern
from .filters import (
    FilterPrimitive, Filter,
    FeGaussianBlur, FeOffset, FeFlood, FeBlend, FeComposite,
    FeMerge, FeMergeNode, FeColorMatrix, FeComponentTransfer, FeComponentTransferFunc,
    FeMorphology, FeConvolveMatrix, FeTurbulence, FeDisplacementMap,
    FeImage, FeTile,
    LightSource, FeDistantLight, FePointLight, FeSpotLight,
    FeDiffuseLighting, FeSpecularLighting, FeDropShadow,
)
from ..core.transforms import Transform

__all__ = [
    # Core
    'Transform',
    # Elements
    'SVGElement', 'RectElement', 'CircleElement', 'EllipseElement',
    'LineElement', 'PolylineElement', 'PolygonElement', 'PathElement',
    'GroupElement', 'TextElement', 'ImageElement',
    'ClipPath', 'Mask', 'SVGDocument',
    # Styles
    'Style', 'FILL_NOT_SET',
    # Gradients
    'GradientStop', 'LinearGradient', 'RadialGradient', 'Pattern',
    # Filters
    'FilterPrimitive', 'Filter',
    'FeGaussianBlur', 'FeOffset', 'FeFlood', 'FeBlend', 'FeComposite',
    'FeMerge', 'FeMergeNode', 'FeColorMatrix', 'FeComponentTransfer', 'FeComponentTransferFunc',
    'FeMorphology', 'FeConvolveMatrix', 'FeTurbulence', 'FeDisplacementMap',
    'FeImage', 'FeTile',
    'LightSource', 'FeDistantLight', 'FePointLight', 'FeSpotLight',
    'FeDiffuseLighting', 'FeSpecularLighting', 'FeDropShadow',
]
