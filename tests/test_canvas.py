"""Tests for the Canvas high-performance rendering API."""

import numpy as np
import pytest
import math

from vectorstag import Canvas


class TestCanvasBasic:
    """Basic Canvas functionality tests."""

    def test_canvas_creation(self):
        """Canvas can be created with valid array."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)
        assert canvas.width == 100
        assert canvas.height == 100
        assert canvas.target is arr

    def test_canvas_invalid_dtype(self):
        """Canvas rejects non-uint8 arrays."""
        arr = np.zeros((100, 100, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            Canvas(arr)

    def test_canvas_invalid_shape(self):
        """Canvas rejects arrays without 4 channels."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            Canvas(arr)


class TestPolygonFill:
    """Tests for polygon fill operations."""

    def test_fill_polygon_basic(self):
        """Basic polygon fill works."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        # Fill a square
        canvas.fill_polygon(
            [(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)],
            (255, 0, 0, 255)
        )

        # Check interior is filled
        assert arr[50, 50, 0] == 255  # Red
        assert arr[50, 50, 3] == 255  # Opaque

        # Check exterior is empty
        assert arr[10, 10, 3] == 0

    def test_fill_polygon_subpixel(self):
        """Polygon fill handles subpixel positioning."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        canvas.fill_polygon(
            [(25.3, 25.7), (74.8, 25.7), (74.8, 74.2), (25.3, 74.2)],
            (0, 255, 0, 255)
        )

        # Left edge at x=25.3 should have partial alpha at column 25
        # (pixel 25 covers [25, 26), edge is at 25.3, so ~0.7 coverage)
        left_edge_alpha = arr[50, 25, 3]  # Middle row, left edge column
        assert 50 < left_edge_alpha < 250, f"Left edge should have partial alpha, got {left_edge_alpha}"

    def test_fill_multi_polygon_with_hole(self):
        """Multi-polygon supports holes."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        outer = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
        inner = [(30.0, 30.0), (30.0, 70.0), (70.0, 70.0), (70.0, 30.0)]

        canvas.fill_multi_polygon([outer, inner], (255, 0, 0, 255))

        # Outer region filled
        assert arr[15, 15, 3] > 0
        # Hole is empty
        assert arr[50, 50, 3] == 0

    def test_fill_polygon_evenodd(self):
        """Even-odd fill rule works."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        # Simple square with evenodd
        canvas.fill_polygon(
            [(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)],
            (255, 0, 0, 255),
            fill_rule='evenodd'
        )

        assert arr[50, 50, 3] > 0


class TestGradients:
    """Tests for gradient fills."""

    def test_linear_gradient_horizontal(self):
        """Horizontal linear gradient works."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        stops = [
            (0.0, 255, 0, 0, 255),  # Red
            (1.0, 0, 0, 255, 255),  # Blue
        ]

        canvas.fill_polygon_linear_gradient(
            [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)],
            10.0, 50.0, 90.0, 50.0,  # Left to right
            stops
        )

        # Left should be reddish
        assert arr[50, 15, 0] > arr[50, 15, 2]
        # Right should be bluish
        assert arr[50, 85, 2] > arr[50, 85, 0]

    def test_radial_gradient(self):
        """Radial gradient works."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        stops = [
            (0.0, 255, 255, 0, 255),  # Yellow center
            (1.0, 128, 0, 128, 255),  # Purple edge
        ]

        canvas.fill_polygon_radial_gradient(
            [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)],
            50.0, 50.0, 50.0,  # Center and radius
            50.0, 50.0, 0.0,   # Focal at center
            stops
        )

        # Center should be yellowish
        center = arr[50, 50, :3]
        assert center[0] > 200 and center[1] > 200


class TestShapes:
    """Tests for basic shape rendering."""

    def test_fill_rect(self):
        """Rectangle fill works."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        canvas.fill_rect(20.5, 30.5, 40.0, 30.0, (255, 128, 0, 255))

        # Interior filled
        assert arr[45, 40, 0] == 255
        assert arr[45, 40, 1] == 128

    def test_fill_circle(self):
        """Circle fill uses distance-based AA."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        canvas.fill_circle(50.0, 50.0, 25.0, (0, 255, 0, 255))

        # Center filled
        assert arr[50, 50, 1] == 255
        # Outside empty
        assert arr[10, 10, 3] == 0
        # Count filled pixels (should be close to pi*r^2)
        filled = np.sum(arr[:, :, 3] > 0)
        expected = math.pi * 25 * 25
        assert abs(filled - expected) < expected * 0.1

    def test_fill_ellipse(self):
        """Ellipse fill works."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        canvas.fill_ellipse(50.0, 50.0, 30.0, 20.0, (0, 0, 255, 255))

        # Center filled (allow minor rounding)
        assert arr[50, 50, 2] >= 250
        assert arr[50, 50, 3] >= 250

    def test_stroke_line(self):
        """Line stroke with caps works."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        canvas.stroke_line(10.0, 50.0, 90.0, 50.0, (255, 0, 0, 255), width=3.0)

        # Line should be visible
        assert arr[50, 50, 3] > 0

    def test_stroke_line_round_cap(self):
        """Round line caps work."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        canvas.stroke_line(30.0, 50.0, 70.0, 50.0, (255, 0, 0, 255),
                          width=10.0, cap='round')

        # Line should be visible in center
        assert arr[50, 50, 3] > 0

        # With width 10, line should extend to y=45-55
        assert arr[47, 50, 3] > 0  # Above center
        assert arr[53, 50, 3] > 0  # Below center


class TestMaskedBlit:
    """Tests for masked image blitting."""

    def test_masked_blit_basic(self):
        """Masked blit works."""
        # Create source
        src = np.zeros((50, 50, 4), dtype=np.uint8)
        src[:, :] = [255, 128, 0, 255]  # Orange

        # Create destination
        dst = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(dst)

        # Circular mask
        mask = [(50.0 + 20.0 * math.cos(a), 50.0 + 20.0 * math.sin(a))
                for a in [i * math.pi / 8 for i in range(16)]]

        canvas.masked_blit(src, mask, 25.0, 25.0, 1.0)

        # Center should have source color
        assert dst[50, 50, 0] == 255
        assert dst[50, 50, 1] == 128

    def test_masked_blit_with_opacity(self):
        """Masked blit respects opacity."""
        src = np.full((50, 50, 4), 255, dtype=np.uint8)
        dst = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(dst)

        mask = [(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)]
        canvas.masked_blit(src, mask, 25.0, 25.0, 0.5)

        # Should have reduced alpha
        center_alpha = dst[50, 50, 3]
        assert 100 < center_alpha < 200


class TestAntialiasing:
    """Tests to verify antialiasing quality."""

    def test_edge_antialiasing(self):
        """Edges are antialiased (not jagged)."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        # Rectangle with fractional coordinates to trigger AA
        canvas.fill_polygon(
            [(25.3, 25.3), (74.7, 25.3), (74.7, 74.7), (25.3, 74.7)],
            (255, 0, 0, 255)
        )

        # Extract edge alphas along left edge (column 25)
        edge_alphas = [arr[y, 25, 3] for y in range(30, 70)]

        # Edge pixels should have consistent partial alpha (not all 255 or 0)
        has_partial = any(0 < a < 255 for a in edge_alphas)
        assert has_partial, "Edge should have partial alpha for antialiasing"

    def test_circle_smooth_edge(self):
        """Circle edges are smooth."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas = Canvas(arr)

        canvas.fill_circle(50.0, 50.0, 30.0, (255, 0, 0, 255))

        # Edge pixels should have varying alpha
        edge_alphas = [arr[20 + i, 50, 3] for i in range(5)]
        unique = len(set(edge_alphas))
        assert unique > 1, "Circle edge should be antialiased"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
