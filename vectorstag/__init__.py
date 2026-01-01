"""
VectorStag - Fast Python SVG Renderer

A high-performance SVG rendering library with multiple output formats.

Simple API (like CairoSVG/resvg):
    >>> import vectorstag
    >>> img = vectorstag.svg_to_pil(svg_string, width=400, height=400)
    >>> arr = vectorstag.svg_to_numpy(svg_string, width=400)
    >>> bgr = vectorstag.svg_to_opencv(svg_string, width=400)

Render to existing targets (in-place):
    >>> vectorstag.render_to_numpy(svg, target_array, x=10, y=10, width=100, height=100)

Class-based API for more control:
    >>> renderer = vectorstag.SVGRenderer(antialias=4)
    >>> img = renderer.render(svg_string, width=800, height=600)
"""

__version__ = "0.2.0"

from typing import Union, Optional, Tuple, Literal
from pathlib import Path
import numpy as np
from PIL import Image

from .renderer import SVGRenderer
from .svg_parser import SVGParser
from .canvas import Canvas

__all__ = [
    # Main classes
    "SVGRenderer",
    "SVGParser",
    "Canvas",
    # Simple conversion functions
    "svg_to_pil",
    "svg_to_numpy",
    "svg_to_opencv",
    "svg_to_bytes",
    # Unified render to target (auto-detects type)
    "render_to",
    # Specific render to target (for explicit control)
    "render_to_pil",
    "render_to_numpy",
    "render_to_opencv",
    # File-based functions
    "file_to_pil",
    "file_to_numpy",
    "file_to_opencv",
]

# Type alias for SVG input
SVGInput = Union[str, bytes, Path]


def _normalize_svg(svg: SVGInput) -> str:
    """Convert various SVG input types to string."""
    if isinstance(svg, bytes):
        return svg.decode('utf-8')
    elif isinstance(svg, Path):
        return svg.read_text(encoding='utf-8')
    return svg


def _normalize_background(background) -> Tuple[int, int, int, int]:
    """Normalize background to RGBA tuple."""
    if background is None:
        return (255, 255, 255, 255)  # White opaque
    if isinstance(background, str):
        # Handle color names and hex
        if background.lower() == 'transparent':
            return (0, 0, 0, 0)
        elif background.startswith('#'):
            h = background.lstrip('#')
            if len(h) == 3:
                h = ''.join(c*2 for c in h)
            if len(h) == 6:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
            elif len(h) == 8:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))
    if isinstance(background, (list, tuple)):
        if len(background) == 3:
            return (background[0], background[1], background[2], 255)
        elif len(background) == 4:
            return tuple(background)
    return (255, 255, 255, 255)


# =============================================================================
# Simple Conversion Functions
# =============================================================================

def svg_to_pil(
    svg: SVGInput,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    background: Union[str, Tuple, None] = None,
    antialias: int = 2,
    scale: float = 1.0,
) -> Image.Image:
    """
    Render SVG to a PIL Image.

    Args:
        svg: SVG content as string, bytes, or Path
        width: Output width in pixels (None = use SVG width)
        height: Output height in pixels (None = use SVG height)
        background: Background color (RGBA tuple, hex string, or 'transparent')
        antialias: Anti-aliasing factor (1=none, 2=good, 4=high quality, 8=maximum)
        scale: Additional scale factor

    Returns:
        PIL Image in RGBA format

    Example:
        >>> img = svg_to_pil('<svg>...</svg>', width=400, height=400)
        >>> img = svg_to_pil(Path('icon.svg'), antialias=4)
    """
    svg_str = _normalize_svg(svg)
    bg = _normalize_background(background)
    renderer = SVGRenderer(scale=scale, background=bg, antialias=antialias)
    return renderer.render(svg_str, width, height)


def svg_to_numpy(
    svg: SVGInput,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    background: Union[str, Tuple, None] = None,
    antialias: int = 2,
    scale: float = 1.0,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Render SVG to a NumPy array (RGBA format, shape: H x W x 4).

    Args:
        svg: SVG content as string, bytes, or Path
        width: Output width in pixels
        height: Output height in pixels
        background: Background color
        antialias: Anti-aliasing factor
        scale: Additional scale factor
        dtype: NumPy dtype for output array

    Returns:
        NumPy array with shape (height, width, 4) in RGBA format

    Example:
        >>> arr = svg_to_numpy('<svg>...</svg>', width=400, height=400)
        >>> assert arr.shape == (400, 400, 4)
    """
    img = svg_to_pil(svg, width, height, background=background,
                     antialias=antialias, scale=scale)
    return np.array(img, dtype=dtype)


def svg_to_opencv(
    svg: SVGInput,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    background: Union[str, Tuple, None] = None,
    antialias: int = 2,
    scale: float = 1.0,
    alpha: bool = True,
) -> np.ndarray:
    """
    Render SVG to an OpenCV-compatible array (BGR or BGRA format).

    Args:
        svg: SVG content as string, bytes, or Path
        width: Output width in pixels
        height: Output height in pixels
        background: Background color
        antialias: Anti-aliasing factor
        scale: Additional scale factor
        alpha: If True, return BGRA (4 channels). If False, return BGR (3 channels).

    Returns:
        NumPy array with shape (height, width, 4) for BGRA or (height, width, 3) for BGR

    Example:
        >>> bgra = svg_to_opencv('<svg>...</svg>', width=400)
        >>> bgr = svg_to_opencv('<svg>...</svg>', width=400, alpha=False)
    """
    rgba = svg_to_numpy(svg, width, height, background=background,
                        antialias=antialias, scale=scale)

    if alpha:
        # RGBA -> BGRA (swap R and B channels)
        return rgba[:, :, [2, 1, 0, 3]]
    else:
        # RGBA -> BGR (swap R and B, drop alpha)
        return rgba[:, :, [2, 1, 0]]


def svg_to_bytes(
    svg: SVGInput,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    format: str = 'PNG',
    background: Union[str, Tuple, None] = None,
    antialias: int = 2,
    scale: float = 1.0,
    **kwargs,
) -> bytes:
    """
    Render SVG to image bytes (PNG, JPEG, etc).

    Args:
        svg: SVG content as string, bytes, or Path
        width: Output width in pixels
        height: Output height in pixels
        format: Output format ('PNG', 'JPEG', 'WEBP', etc)
        background: Background color
        antialias: Anti-aliasing factor
        scale: Additional scale factor
        **kwargs: Additional arguments passed to PIL save

    Returns:
        Image data as bytes

    Example:
        >>> png_bytes = svg_to_bytes('<svg>...</svg>', width=400, format='PNG')
        >>> jpeg_bytes = svg_to_bytes('<svg>...</svg>', width=400, format='JPEG', quality=90)
    """
    import io
    img = svg_to_pil(svg, width, height, background=background,
                     antialias=antialias, scale=scale)

    # Convert to RGB for formats that don't support alpha
    if format.upper() in ('JPEG', 'JPG'):
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, _normalize_background(background)[:3])
            bg.paste(img, mask=img.split()[3])
            img = bg

    buffer = io.BytesIO()
    img.save(buffer, format=format.upper(), **kwargs)
    return buffer.getvalue()


# =============================================================================
# File-based Functions
# =============================================================================

def file_to_pil(
    filepath: Union[str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Image.Image:
    """Render SVG file to PIL Image."""
    svg_str = Path(filepath).read_text(encoding='utf-8')
    return svg_to_pil(svg_str, width, height, **kwargs)


def file_to_numpy(
    filepath: Union[str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Render SVG file to NumPy array (RGBA)."""
    svg_str = Path(filepath).read_text(encoding='utf-8')
    return svg_to_numpy(svg_str, width, height, **kwargs)


def file_to_opencv(
    filepath: Union[str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Render SVG file to OpenCV array (BGRA or BGR)."""
    svg_str = Path(filepath).read_text(encoding='utf-8')
    return svg_to_opencv(svg_str, width, height, **kwargs)


# =============================================================================
# Render to Existing Target
# =============================================================================

def render_to(
    svg: SVGInput,
    target: Union[Image.Image, np.ndarray],
    x: int = 0,
    y: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    antialias: int = 2,
    format: Literal['rgba', 'bgra'] = 'rgba',
) -> None:
    """
    Render SVG onto existing target (PIL Image or numpy array).

    Target type is auto-detected:
    - PIL.Image.Image: Uses PIL compositing
    - np.ndarray: Uses RGBA compositing (or BGRA if format='bgra')

    Args:
        svg: SVG content as string, bytes, or Path
        target: Target to render onto (modified in-place)
        x: X offset in target
        y: Y offset in target
        width: Width of rendered SVG (None = fit remaining space)
        height: Height of rendered SVG (None = fit remaining space)
        antialias: Anti-aliasing factor (1=none, 2=2x, 4=4x)
        format: For numpy arrays, 'rgba' (default) or 'bgra' (OpenCV style)

    Example:
        >>> # PIL Image
        >>> canvas = Image.new('RGBA', (800, 600), (255, 255, 255, 255))
        >>> render_to(svg, canvas, x=10, y=10, width=64, height=64)
        >>>
        >>> # NumPy RGBA
        >>> canvas = np.zeros((600, 800, 4), dtype=np.uint8)
        >>> render_to(svg, canvas, x=10, y=10, width=64, height=64)
        >>>
        >>> # OpenCV BGRA
        >>> canvas = np.zeros((600, 800, 4), dtype=np.uint8)
        >>> render_to(svg, canvas, x=10, y=10, width=64, height=64, format='bgra')
    """
    if isinstance(target, Image.Image):
        render_to_pil(svg, target, x, y, width, height, antialias=antialias)
    elif isinstance(target, np.ndarray):
        if format.lower() == 'bgra':
            render_to_opencv(svg, target, x, y, width, height, antialias=antialias)
        else:
            render_to_numpy(svg, target, x, y, width, height, antialias=antialias)
    else:
        raise TypeError(f"Unsupported target type: {type(target)}. "
                       "Expected PIL.Image.Image or numpy.ndarray")


def render_to_pil(
    svg: SVGInput,
    target: Image.Image,
    x: int = 0,
    y: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    antialias: int = 2,
) -> None:
    """
    Render SVG onto an existing PIL Image at specified position.

    Args:
        svg: SVG content
        target: Target PIL Image to render onto (modified in-place)
        x: X offset in target image
        y: Y offset in target image
        width: Width of rendered SVG (None = fit remaining width)
        height: Height of rendered SVG (None = fit remaining height)
        antialias: Anti-aliasing factor

    Example:
        >>> canvas = Image.new('RGBA', (800, 600), (255, 255, 255, 255))
        >>> render_to_pil(svg_icon, canvas, x=10, y=10, width=64, height=64)
    """
    # Calculate dimensions if not specified
    if width is None:
        width = target.width - x
    if height is None:
        height = target.height - y

    # Render SVG with transparent background
    svg_img = svg_to_pil(svg, width, height, background='transparent', antialias=antialias)

    # Composite onto target
    target.paste(svg_img, (x, y), svg_img)


def render_to_numpy(
    svg: SVGInput,
    target: np.ndarray,
    x: int = 0,
    y: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    antialias: int = 2,
) -> None:
    """
    Render SVG onto an existing NumPy array at specified position.

    Supports RGBA (4 channels) and RGB (3 channels) target arrays.

    Args:
        svg: SVG content
        target: Target NumPy array (modified in-place), shape (H, W, 3) or (H, W, 4)
        x: X offset in target array
        y: Y offset in target array
        width: Width of rendered SVG
        height: Height of rendered SVG
        antialias: Anti-aliasing factor

    Example:
        >>> canvas = np.zeros((600, 800, 4), dtype=np.uint8)
        >>> render_to_numpy(svg_icon, canvas, x=10, y=10, width=64, height=64)
    """
    h, w = target.shape[:2]
    channels = target.shape[2] if len(target.shape) == 3 else 1

    # Calculate dimensions if not specified
    if width is None:
        width = w - x
    if height is None:
        height = h - y

    # Clamp to valid region
    render_w = min(width, w - x)
    render_h = min(height, h - y)

    if render_w <= 0 or render_h <= 0:
        return

    # Render SVG
    svg_arr = svg_to_numpy(svg, render_w, render_h, background='transparent', antialias=antialias)

    # Alpha composite onto target
    _alpha_composite_numpy(target, svg_arr, x, y)


def render_to_opencv(
    svg: SVGInput,
    target: np.ndarray,
    x: int = 0,
    y: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    antialias: int = 2,
) -> None:
    """
    Render SVG onto an existing OpenCV array at specified position.

    Supports BGRA (4 channels) and BGR (3 channels) target arrays.

    Args:
        svg: SVG content
        target: Target OpenCV array (modified in-place), shape (H, W, 3) or (H, W, 4)
        x: X offset in target array
        y: Y offset in target array
        width: Width of rendered SVG
        height: Height of rendered SVG
        antialias: Anti-aliasing factor

    Example:
        >>> canvas = np.zeros((600, 800, 4), dtype=np.uint8)  # BGRA
        >>> render_to_opencv(svg_icon, canvas, x=10, y=10, width=64, height=64)
    """
    h, w = target.shape[:2]
    channels = target.shape[2] if len(target.shape) == 3 else 1

    # Calculate dimensions
    if width is None:
        width = w - x
    if height is None:
        height = h - y

    render_w = min(width, w - x)
    render_h = min(height, h - y)

    if render_w <= 0 or render_h <= 0:
        return

    # Render SVG as BGRA
    svg_bgra = svg_to_opencv(svg, render_w, render_h, background='transparent',
                              antialias=antialias, alpha=True)

    # Alpha composite onto target (handles both BGR and BGRA targets)
    _alpha_composite_opencv(target, svg_bgra, x, y)


def _alpha_composite_numpy(target: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """Alpha composite RGBA overlay onto target array."""
    h, w = overlay.shape[:2]
    th, tw = target.shape[:2]
    channels = target.shape[2] if len(target.shape) == 3 else 1

    # Clamp region
    x2 = min(x + w, tw)
    y2 = min(y + h, th)
    ox2 = x2 - x
    oy2 = y2 - y

    if ox2 <= 0 or oy2 <= 0:
        return

    # Get overlay region
    ov = overlay[:oy2, :ox2]
    alpha = ov[:, :, 3:4].astype(np.float32) / 255.0

    if channels >= 4:
        # RGBA target
        dst = target[y:y2, x:x2]
        dst_alpha = dst[:, :, 3:4].astype(np.float32) / 255.0

        # Porter-Duff over operator
        out_alpha = alpha + dst_alpha * (1 - alpha)
        out_alpha_safe = np.maximum(out_alpha, 1e-10)

        for c in range(3):
            dst[:, :, c] = (
                (ov[:, :, c].astype(np.float32) * alpha[:, :, 0] +
                 dst[:, :, c].astype(np.float32) * dst_alpha[:, :, 0] * (1 - alpha[:, :, 0])) /
                out_alpha_safe[:, :, 0]
            ).clip(0, 255).astype(np.uint8)

        dst[:, :, 3] = (out_alpha[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    else:
        # RGB target - simple blend
        dst = target[y:y2, x:x2]
        for c in range(min(3, channels)):
            dst[:, :, c] = (
                ov[:, :, c].astype(np.float32) * alpha[:, :, 0] +
                dst[:, :, c].astype(np.float32) * (1 - alpha[:, :, 0])
            ).clip(0, 255).astype(np.uint8)


def _alpha_composite_opencv(target: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """Alpha composite BGRA overlay onto target array (BGR or BGRA)."""
    h, w = overlay.shape[:2]
    th, tw = target.shape[:2]
    channels = target.shape[2] if len(target.shape) == 3 else 1

    # Clamp region
    x2 = min(x + w, tw)
    y2 = min(y + h, th)
    ox2 = x2 - x
    oy2 = y2 - y

    if ox2 <= 0 or oy2 <= 0:
        return

    # Get overlay region (BGRA)
    ov = overlay[:oy2, :ox2]
    alpha = ov[:, :, 3:4].astype(np.float32) / 255.0

    if channels >= 4:
        # BGRA target
        dst = target[y:y2, x:x2]
        dst_alpha = dst[:, :, 3:4].astype(np.float32) / 255.0

        out_alpha = alpha + dst_alpha * (1 - alpha)
        out_alpha_safe = np.maximum(out_alpha, 1e-10)

        for c in range(3):  # B, G, R
            dst[:, :, c] = (
                (ov[:, :, c].astype(np.float32) * alpha[:, :, 0] +
                 dst[:, :, c].astype(np.float32) * dst_alpha[:, :, 0] * (1 - alpha[:, :, 0])) /
                out_alpha_safe[:, :, 0]
            ).clip(0, 255).astype(np.uint8)

        dst[:, :, 3] = (out_alpha[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    else:
        # BGR target
        dst = target[y:y2, x:x2]
        for c in range(min(3, channels)):
            dst[:, :, c] = (
                ov[:, :, c].astype(np.float32) * alpha[:, :, 0] +
                dst[:, :, c].astype(np.float32) * (1 - alpha[:, :, 0])
            ).clip(0, 255).astype(np.uint8)
