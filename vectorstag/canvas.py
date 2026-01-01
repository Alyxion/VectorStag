"""High-performance subpixel rendering canvas.

This module provides a Canvas class for direct rendering to numpy arrays
with analytical antialiasing (equivalent to infinite supersampling quality).
"""

from typing import List, Tuple, Optional
import numpy as np
import vectorstag_rust


class Canvas:
    """High-performance subpixel rendering canvas.

    Renders shapes directly to a numpy RGBA array with analytical antialiasing,
    achieving quality equivalent to 8x8 supersampling without the memory overhead.

    Example:
        >>> import numpy as np
        >>> from vectorstag import Canvas
        >>>
        >>> # Create a 1920x1080 RGBA image
        >>> image = np.zeros((1080, 1920, 4), dtype=np.uint8)
        >>> canvas = Canvas(image)
        >>>
        >>> # Draw a red polygon with subpixel positioning
        >>> canvas.fill_polygon(
        ...     [(100.3, 50.7), (200.8, 150.2), (50.1, 200.9)],
        ...     color=(255, 0, 0, 255)
        ... )
    """

    def __init__(self, target: np.ndarray):
        """Initialize canvas with target numpy array.

        Args:
            target: RGBA numpy array with shape (H, W, 4) and dtype=uint8.
                    The array is modified in-place during drawing operations.

        Raises:
            ValueError: If target is not a valid RGBA array.
        """
        if not isinstance(target, np.ndarray):
            raise ValueError("Target must be a numpy array")
        if target.dtype != np.uint8:
            raise ValueError("Target must have dtype uint8")
        if len(target.shape) != 3 or target.shape[2] != 4:
            raise ValueError("Target must have shape (H, W, 4)")

        self._target = target
        self._width = target.shape[1]
        self._height = target.shape[0]

    @property
    def width(self) -> int:
        """Width of the canvas in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Height of the canvas in pixels."""
        return self._height

    @property
    def target(self) -> np.ndarray:
        """The underlying numpy array."""
        return self._target

    def fill_polygon(
        self,
        points: List[Tuple[float, float]],
        color: Tuple[int, int, int, int],
        fill_rule: str = 'nonzero',
    ) -> None:
        """Fill polygon with analytical antialiasing.

        Args:
            points: List of (x, y) float tuples defining polygon vertices.
            color: RGBA color as (r, g, b, a) tuple with values 0-255.
            fill_rule: Either 'nonzero' or 'evenodd'.
        """
        vectorstag_rust.canvas_fill_polygon_aa(
            self._target, points,
            color[0], color[1], color[2], color[3],
            0 if fill_rule == 'nonzero' else 1
        )

    def fill_multi_polygon(
        self,
        contours: List[List[Tuple[float, float]]],
        color: Tuple[int, int, int, int],
        fill_rule: str = 'nonzero',
    ) -> None:
        """Fill polygon with multiple contours (supports holes).

        Args:
            contours: List of contours, where each contour is a list of (x, y) points.
                      Outer contours should be counter-clockwise, holes clockwise.
            color: RGBA color as (r, g, b, a) tuple.
            fill_rule: Either 'nonzero' or 'evenodd'.
        """
        vectorstag_rust.canvas_fill_multi_polygon_aa(
            self._target, contours,
            color[0], color[1], color[2], color[3],
            0 if fill_rule == 'nonzero' else 1
        )

    def fill_polygon_linear_gradient(
        self,
        points: List[Tuple[float, float]],
        x1: float, y1: float,
        x2: float, y2: float,
        stops: List[Tuple[float, int, int, int, int]],
        spread_method: str = 'pad',
        fill_rule: str = 'nonzero',
    ) -> None:
        """Fill polygon with linear gradient.

        Args:
            points: Polygon vertices.
            x1, y1: Gradient start point.
            x2, y2: Gradient end point.
            stops: List of (position, r, g, b, a) tuples. Position is 0.0 to 1.0.
            spread_method: 'pad', 'repeat', or 'reflect'.
            fill_rule: 'nonzero' or 'evenodd'.
        """
        spread_map = {'pad': 0, 'repeat': 1, 'reflect': 2}
        vectorstag_rust.canvas_fill_polygon_linear_gradient_aa(
            self._target, points,
            x1, y1, x2, y2,
            stops,
            spread_map.get(spread_method, 0),
            0 if fill_rule == 'nonzero' else 1
        )

    def fill_polygon_radial_gradient(
        self,
        points: List[Tuple[float, float]],
        cx: float, cy: float, radius: float,
        fx: float, fy: float, fr: float,
        stops: List[Tuple[float, int, int, int, int]],
        spread_method: str = 'pad',
        fill_rule: str = 'nonzero',
    ) -> None:
        """Fill polygon with radial gradient.

        Args:
            points: Polygon vertices.
            cx, cy, radius: Outer circle center and radius.
            fx, fy, fr: Focal point and inner radius.
            stops: List of (position, r, g, b, a) tuples.
            spread_method: 'pad', 'repeat', or 'reflect'.
            fill_rule: 'nonzero' or 'evenodd'.
        """
        spread_map = {'pad': 0, 'repeat': 1, 'reflect': 2}
        vectorstag_rust.canvas_fill_polygon_radial_gradient_aa(
            self._target, points,
            cx, cy, radius,
            fx, fy, fr,
            stops,
            spread_map.get(spread_method, 0),
            0 if fill_rule == 'nonzero' else 1
        )

    def fill_rect(
        self,
        x: float, y: float,
        width: float, height: float,
        color: Tuple[int, int, int, int],
    ) -> None:
        """Fill axis-aligned rectangle with subpixel positioning.

        Args:
            x, y: Top-left corner (can be fractional for subpixel positioning).
            width, height: Rectangle dimensions.
            color: RGBA color.
        """
        vectorstag_rust.canvas_fill_rect_aa(
            self._target, x, y, width, height,
            color[0], color[1], color[2], color[3]
        )

    def fill_circle(
        self,
        cx: float, cy: float, radius: float,
        color: Tuple[int, int, int, int],
    ) -> None:
        """Fill circle with analytical antialiasing.

        Uses distance-based AA rather than polygon approximation.

        Args:
            cx, cy: Center coordinates.
            radius: Circle radius.
            color: RGBA color.
        """
        vectorstag_rust.canvas_fill_circle_aa(
            self._target, cx, cy, radius,
            color[0], color[1], color[2], color[3]
        )

    def fill_ellipse(
        self,
        cx: float, cy: float,
        rx: float, ry: float,
        color: Tuple[int, int, int, int],
    ) -> None:
        """Fill ellipse with analytical antialiasing.

        Args:
            cx, cy: Center coordinates.
            rx, ry: Semi-axes (horizontal and vertical radii).
            color: RGBA color.
        """
        vectorstag_rust.canvas_fill_ellipse_aa(
            self._target, cx, cy, rx, ry,
            color[0], color[1], color[2], color[3]
        )

    def stroke_line(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        color: Tuple[int, int, int, int],
        width: float = 1.0,
        cap: str = 'butt',
    ) -> None:
        """Draw antialiased line with subpixel endpoints.

        Args:
            x1, y1: Start point.
            x2, y2: End point.
            color: RGBA color.
            width: Line width in pixels.
            cap: Line cap style: 'butt', 'round', or 'square'.
        """
        cap_map = {'butt': 0, 'round': 1, 'square': 2}
        vectorstag_rust.canvas_stroke_line_aa(
            self._target,
            x1, y1, x2, y2,
            color[0], color[1], color[2], color[3],
            width,
            cap_map.get(cap, 0)
        )

    def masked_blit(
        self,
        src: np.ndarray,
        mask_polygon: List[Tuple[float, float]],
        dst_x: float, dst_y: float,
        opacity: float = 1.0,
    ) -> None:
        """Blit image with polygon mask.

        The mask polygon defines which parts of the source image are visible.
        Mask edges are antialiased.

        Args:
            src: Source RGBA numpy array.
            mask_polygon: Polygon defining the visible region.
            dst_x, dst_y: Destination position (can be fractional).
            opacity: Overall opacity multiplier (0.0 to 1.0).
        """
        vectorstag_rust.canvas_masked_blit_aa(
            self._target, src,
            mask_polygon,
            dst_x, dst_y,
            opacity
        )
