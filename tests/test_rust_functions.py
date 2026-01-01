"""
Unit tests for VectorStag Rust extension functions.

These tests verify that Rust implementations produce identical or near-identical
results compared to Python/NumPy/PIL reference implementations.
"""
import pytest
import numpy as np
from PIL import Image, ImageDraw

# Import Rust extension
try:
    import vectorstag_rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    pytestmark = pytest.mark.skip("Rust extension not available")


# =============================================================================
# Reference Python implementations for comparison
# =============================================================================

def python_fill_polygon_scanline(points: list, width: int, height: int,
                                  min_x: int, min_y: int, rule: str = "nonzero") -> np.ndarray:
    """Python reference implementation for polygon fill using scanline algorithm."""
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(points) < 3:
        return mask

    # Close polygon if needed
    pts = list(points)
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    n = len(pts) - 1

    # Build edge list with direction for nonzero rule
    edges = []
    for i in range(n):
        y1, y2 = pts[i][1], pts[i + 1][1]
        if y1 == y2:
            continue
        x1, x2 = pts[i][0], pts[i + 1][0]

        if y1 > y2:
            y1, y2 = y2, y1
            x1, x2 = x2, x1
            direction = -1
        else:
            direction = 1

        dx = (x2 - x1) / (y2 - y1) if y2 != y1 else 0
        edges.append((y1, y2, x1, dx, direction))

    if not edges:
        return mask

    # Scanline fill
    y_min = max(0, int(min(e[0] for e in edges)))
    y_max = min(height, int(max(e[1] for e in edges)) + 1)

    for y in range(y_min, y_max):
        screen_y = y - min_y if min_y < 0 else y
        if screen_y < 0 or screen_y >= height:
            continue

        wy = y + min_y + 0.5
        intersections = []

        for y1, y2, x1, dx, direction in edges:
            if y1 <= wy < y2:
                x = x1 + (wy - y1) * dx
                intersections.append((x, direction))

        intersections.sort(key=lambda p: p[0])

        if rule == "evenodd":
            for i in range(0, len(intersections) - 1, 2):
                x_start = int(max(0, intersections[i][0] - min_x))
                x_end = int(min(width, intersections[i + 1][0] - min_x + 1))
                if x_start < width and x_end > 0:
                    mask[screen_y, x_start:x_end] = 255
        else:  # nonzero
            winding = 0
            prev_x = None
            for x, direction in intersections:
                if winding != 0 and prev_x is not None:
                    x_start = int(max(0, prev_x - min_x))
                    x_end = int(min(width, x - min_x + 1))
                    if x_start < width and x_end > 0:
                        mask[screen_y, x_start:x_end] = 255
                winding += direction
                prev_x = x

    return mask


def python_interpolate_gradient_colors(t: np.ndarray, offsets: list,
                                        colors: list, opacity: float) -> np.ndarray:
    """Python reference implementation for gradient color interpolation."""
    height, width = t.shape
    pixels = np.zeros((height, width, 4), dtype=np.uint8)

    if not offsets or not colors:
        return pixels

    offsets = np.array(offsets, dtype=np.float32)
    colors_arr = np.array(colors, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            t_val = t[y, x]

            if t_val <= offsets[0]:
                r, g, b, a = colors[0]
            elif t_val >= offsets[-1]:
                r, g, b, a = colors[-1]
            else:
                # Find interval
                idx = np.searchsorted(offsets, t_val, side='right') - 1
                idx = max(0, min(idx, len(offsets) - 2))

                s1_off = offsets[idx]
                s2_off = offsets[idx + 1]
                denom = s2_off - s1_off

                if abs(denom) < 1e-10:
                    ratio = 0.0
                else:
                    ratio = np.clip((t_val - s1_off) / denom, 0, 1)

                c1 = colors_arr[idx]
                c2 = colors_arr[idx + 1]
                r = c1[0] + ratio * (c2[0] - c1[0])
                g = c1[1] + ratio * (c2[1] - c1[1])
                b = c1[2] + ratio * (c2[2] - c1[2])
                a = c1[3] + ratio * (c2[3] - c1[3])

            pixels[y, x, 0] = int(r)
            pixels[y, x, 1] = int(g)
            pixels[y, x, 2] = int(b)
            pixels[y, x, 3] = int(a * opacity)

    return pixels


def python_create_linear_gradient_image(width: int, height: int, offset_x: int, offset_y: int,
                                         x1: float, y1: float, dx: float, dy: float, length: float,
                                         offsets: list, colors: list, opacity: float,
                                         spread_method: int) -> np.ndarray:
    """Python reference for linear gradient image creation."""
    pixels = np.zeros((height, width, 4), dtype=np.uint8)

    if not offsets or not colors or abs(length) < 1e-10:
        return pixels

    for row in range(height):
        wy = row + offset_y
        for col in range(width):
            wx = col + offset_x
            t_raw = ((wx - x1) * dx + (wy - y1) * dy) / length

            # Apply spread method
            if spread_method == 1:  # repeat
                t = t_raw % 1.0
                if t < 0:
                    t += 1.0
            elif spread_method == 2:  # reflect
                t = t_raw % 2.0
                if t < 0:
                    t += 2.0
                if t > 1.0:
                    t = 2.0 - t
            else:  # pad
                t = np.clip(t_raw, 0, 1)

            # Interpolate color
            if t <= offsets[0]:
                r, g, b, a = colors[0]
            elif t >= offsets[-1]:
                r, g, b, a = colors[-1]
            else:
                idx = 0
                for i in range(len(offsets) - 1):
                    if offsets[i] <= t < offsets[i + 1]:
                        idx = i
                        break

                s1_off = offsets[idx]
                s2_off = offsets[idx + 1]
                denom = s2_off - s1_off
                ratio = (t - s1_off) / denom if denom > 1e-10 else 0
                ratio = np.clip(ratio, 0, 1)

                c1 = colors[idx]
                c2 = colors[idx + 1]
                r = c1[0] + ratio * (c2[0] - c1[0])
                g = c1[1] + ratio * (c2[1] - c1[1])
                b = c1[2] + ratio * (c2[2] - c1[2])
                a = c1[3] + ratio * (c2[3] - c1[3])

            pixels[row, col, 0] = int(r)
            pixels[row, col, 1] = int(g)
            pixels[row, col, 2] = int(b)
            pixels[row, col, 3] = int(a * opacity)

    return pixels


def python_create_radial_gradient_image(width: int, height: int, offset_x: int, offset_y: int,
                                         cx: float, cy: float, r: float,
                                         inv_a: float, inv_b: float, inv_c: float,
                                         inv_d: float, inv_e: float, inv_f: float,
                                         offsets: list, colors: list, opacity: float,
                                         spread_method: int) -> np.ndarray:
    """Python reference for radial gradient image creation."""
    pixels = np.zeros((height, width, 4), dtype=np.uint8)

    if not offsets or not colors or abs(r) < 1e-10:
        return pixels

    for row in range(height):
        wy = row + offset_y
        for col in range(width):
            wx = col + offset_x

            # Inverse transform
            gx = inv_a * wx + inv_b * wy + inv_e
            gy = inv_c * wx + inv_d * wy + inv_f

            # Distance from center
            dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
            t_raw = dist / r

            # Apply spread method
            if spread_method == 1:  # repeat
                t = t_raw % 1.0
            elif spread_method == 2:  # reflect
                t = t_raw % 2.0
                if t > 1.0:
                    t = 2.0 - t
            else:  # pad
                t = np.clip(t_raw, 0, 1)

            # Interpolate color (same as linear)
            if t <= offsets[0]:
                r_c, g, b, a = colors[0]
            elif t >= offsets[-1]:
                r_c, g, b, a = colors[-1]
            else:
                idx = 0
                for i in range(len(offsets) - 1):
                    if offsets[i] <= t < offsets[i + 1]:
                        idx = i
                        break

                s1_off = offsets[idx]
                s2_off = offsets[idx + 1]
                denom = s2_off - s1_off
                ratio = (t - s1_off) / denom if denom > 1e-10 else 0
                ratio = np.clip(ratio, 0, 1)

                c1 = colors[idx]
                c2 = colors[idx + 1]
                r_c = c1[0] + ratio * (c2[0] - c1[0])
                g = c1[1] + ratio * (c2[1] - c1[1])
                b = c1[2] + ratio * (c2[2] - c1[2])
                a = c1[3] + ratio * (c2[3] - c1[3])

            pixels[row, col, 0] = int(r_c)
            pixels[row, col, 1] = int(g)
            pixels[row, col, 2] = int(b)
            pixels[row, col, 3] = int(a * opacity)

    return pixels


def python_sample_cubic_bezier(x0, y0, x1, y1, x2, y2, x3, y3, n_samples):
    """Python reference for cubic bezier sampling."""
    points = []
    for i in range(1, n_samples + 1):
        t = i / n_samples
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3 * x0 + 3 * mt2 * t * x1 + 3 * mt * t2 * x2 + t3 * x3
        y = mt3 * y0 + 3 * mt2 * t * y1 + 3 * mt * t2 * y2 + t3 * y3
        points.append((x, y))
    return points


def python_sample_quadratic_bezier(x0, y0, x1, y1, x2, y2, n_samples):
    """Python reference for quadratic bezier sampling."""
    points = []
    for i in range(1, n_samples + 1):
        t = i / n_samples
        mt = 1 - t

        x = mt * mt * x0 + 2 * mt * t * x1 + t * t * x2
        y = mt * mt * y0 + 2 * mt * t * y1 + t * t * y2
        points.append((x, y))
    return points


# =============================================================================
# Test Classes
# =============================================================================

class TestFillPolygonNonzero:
    """Tests for fill_polygon_nonzero Rust function."""

    def test_simple_triangle(self):
        """Test filling a simple triangle."""
        points = [(10.0, 10.0), (50.0, 10.0), (30.0, 50.0)]
        width, height = 60, 60

        rust_result = vectorstag_rust.fill_polygon_nonzero(points, width, height, 0, 0)

        # Basic sanity checks
        assert rust_result.shape == (height, width)
        assert rust_result.dtype == np.uint8
        assert np.any(rust_result == 255)  # Some pixels filled

    def test_square(self):
        """Test filling a square."""
        points = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
        width, height = 60, 60

        rust_result = vectorstag_rust.fill_polygon_nonzero(points, width, height, 0, 0)

        # Check interior is filled
        assert rust_result[30, 30] == 255
        # Check exterior is not filled
        assert rust_result[5, 5] == 0

    def test_concave_polygon(self):
        """Test filling a concave (L-shaped) polygon."""
        points = [
            (10.0, 10.0), (50.0, 10.0), (50.0, 30.0),
            (30.0, 30.0), (30.0, 50.0), (10.0, 50.0)
        ]
        width, height = 60, 60

        rust_result = vectorstag_rust.fill_polygon_nonzero(points, width, height, 0, 0)

        # Check inside L is filled
        assert rust_result[20, 20] == 255
        # Check cutout area is not filled
        assert rust_result[40, 40] == 0


class TestFillPolygonEvenOdd:
    """Tests for fill_polygon_evenodd Rust function."""

    def test_simple_triangle(self):
        """Test filling a simple triangle with evenodd rule."""
        points = [(10.0, 10.0), (50.0, 10.0), (30.0, 50.0)]
        width, height = 60, 60

        rust_result = vectorstag_rust.fill_polygon_evenodd(points, width, height, 0, 0)

        assert rust_result.shape == (height, width)
        assert np.any(rust_result == 255)

    def test_self_intersecting_star(self):
        """Test five-pointed star (self-intersecting) with evenodd rule."""
        import math
        # Create a five-pointed star
        center = (30.0, 30.0)
        outer_r = 25.0
        inner_r = 10.0

        points = []
        for i in range(5):
            angle = math.pi / 2 + i * 2 * math.pi / 5
            points.append((
                center[0] + outer_r * math.cos(angle),
                center[1] - outer_r * math.sin(angle)
            ))
            angle += math.pi / 5
            points.append((
                center[0] + inner_r * math.cos(angle),
                center[1] - inner_r * math.sin(angle)
            ))

        width, height = 60, 60
        rust_result = vectorstag_rust.fill_polygon_evenodd(points, width, height, 0, 0)

        assert rust_result.shape == (height, width)
        # Star points should be filled
        assert np.any(rust_result == 255)


class TestFillMultiPolygonNonzero:
    """Tests for fill_multi_polygon_nonzero Rust function."""

    def test_two_separate_triangles(self):
        """Test filling two separate triangles."""
        polygons = [
            [(10.0, 10.0), (30.0, 10.0), (20.0, 30.0)],
            [(40.0, 10.0), (60.0, 10.0), (50.0, 30.0)]
        ]
        width, height = 70, 40

        rust_result = vectorstag_rust.fill_multi_polygon_nonzero(polygons, width, height, 0, 0)

        assert rust_result.shape == (height, width)
        # Both triangles should have filled pixels
        assert rust_result[20, 20] == 255  # First triangle
        assert rust_result[20, 50] == 255  # Second triangle

    def test_polygon_with_hole(self):
        """Test outer polygon with inner hole (opposite winding)."""
        # Outer square (clockwise)
        outer = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
        # Inner square (counter-clockwise for hole)
        inner = [(20.0, 20.0), (20.0, 40.0), (40.0, 40.0), (40.0, 20.0)]

        polygons = [outer, inner]
        width, height = 60, 60

        rust_result = vectorstag_rust.fill_multi_polygon_nonzero(polygons, width, height, 0, 0)

        # Check that outer area is filled
        assert rust_result[15, 15] == 255
        # Check that hole is not filled
        assert rust_result[30, 30] == 0


class TestFillMultiPolygonEvenOdd:
    """Tests for fill_multi_polygon_evenodd Rust function."""

    def test_two_separate_triangles(self):
        """Test filling two separate triangles with evenodd rule."""
        polygons = [
            [(10.0, 10.0), (30.0, 10.0), (20.0, 30.0)],
            [(40.0, 10.0), (60.0, 10.0), (50.0, 30.0)]
        ]
        width, height = 70, 40

        rust_result = vectorstag_rust.fill_multi_polygon_evenodd(polygons, width, height, 0, 0)

        assert rust_result.shape == (height, width)
        assert rust_result[20, 20] == 255
        assert rust_result[20, 50] == 255


class TestInterpolateGradientColors:
    """Tests for interpolate_gradient_colors Rust function."""

    def test_simple_two_color_gradient(self):
        """Test simple gradient from red to blue."""
        t = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
        offsets = [0.0, 1.0]
        colors = [(255, 0, 0, 255), (0, 0, 255, 255)]
        opacity = 1.0

        rust_result = vectorstag_rust.interpolate_gradient_colors(t, offsets, colors, opacity)
        python_result = python_interpolate_gradient_colors(t, offsets, colors, opacity)

        # Allow small differences due to rounding
        diff = np.abs(rust_result.astype(np.int16) - python_result.astype(np.int16))
        assert np.max(diff) <= 1, f"Max diff: {np.max(diff)}"

    def test_three_color_gradient(self):
        """Test gradient with three stops."""
        t = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
        offsets = [0.0, 0.5, 1.0]
        colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
        opacity = 1.0

        rust_result = vectorstag_rust.interpolate_gradient_colors(t, offsets, colors, opacity)
        python_result = python_interpolate_gradient_colors(t, offsets, colors, opacity)

        diff = np.abs(rust_result.astype(np.int16) - python_result.astype(np.int16))
        assert np.max(diff) <= 1

    def test_opacity(self):
        """Test that opacity is applied correctly."""
        t = np.array([[0.5]], dtype=np.float32)
        offsets = [0.0, 1.0]
        colors = [(255, 0, 0, 255), (0, 0, 255, 255)]
        opacity = 0.5

        rust_result = vectorstag_rust.interpolate_gradient_colors(t, offsets, colors, opacity)

        # Alpha should be around 127 (255 * 0.5)
        assert 126 <= rust_result[0, 0, 3] <= 128

    def test_2d_gradient(self):
        """Test gradient on a 2D array."""
        t = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        offsets = [0.0, 1.0]
        colors = [(255, 0, 0, 255), (0, 0, 255, 255)]
        opacity = 1.0

        rust_result = vectorstag_rust.interpolate_gradient_colors(t, offsets, colors, opacity)
        python_result = python_interpolate_gradient_colors(t, offsets, colors, opacity)

        diff = np.abs(rust_result.astype(np.int16) - python_result.astype(np.int16))
        assert np.max(diff) <= 1


class TestCreateLinearGradientImage:
    """Tests for create_linear_gradient_image Rust function."""

    def test_horizontal_gradient(self):
        """Test horizontal linear gradient."""
        width, height = 100, 50
        offset_x, offset_y = 0, 0
        x1, y1 = 0.0, 25.0
        dx, dy = 1.0, 0.0
        length = 100.0
        offsets = [0.0, 1.0]
        colors = [(255, 0, 0, 255), (0, 0, 255, 255)]
        opacity = 1.0
        spread_method = 0  # pad

        rust_result = vectorstag_rust.create_linear_gradient_image(
            width, height, offset_x, offset_y,
            x1, y1, dx, dy, length,
            offsets, colors, opacity, spread_method
        )

        python_result = python_create_linear_gradient_image(
            width, height, offset_x, offset_y,
            x1, y1, dx, dy, length,
            offsets, colors, opacity, spread_method
        )

        diff = np.abs(rust_result.astype(np.int16) - python_result.astype(np.int16))
        assert np.max(diff) <= 1, f"Max diff: {np.max(diff)}"

    def test_vertical_gradient(self):
        """Test vertical linear gradient."""
        width, height = 50, 100
        offset_x, offset_y = 0, 0
        x1, y1 = 25.0, 0.0
        dx, dy = 0.0, 1.0
        length = 100.0
        offsets = [0.0, 1.0]
        colors = [(0, 255, 0, 255), (255, 255, 0, 255)]
        opacity = 1.0
        spread_method = 0

        rust_result = vectorstag_rust.create_linear_gradient_image(
            width, height, offset_x, offset_y,
            x1, y1, dx, dy, length,
            offsets, colors, opacity, spread_method
        )

        python_result = python_create_linear_gradient_image(
            width, height, offset_x, offset_y,
            x1, y1, dx, dy, length,
            offsets, colors, opacity, spread_method
        )

        diff = np.abs(rust_result.astype(np.int16) - python_result.astype(np.int16))
        assert np.max(diff) <= 1

    def test_repeat_spread(self):
        """Test repeat spread method."""
        width, height = 100, 20
        x1, y1 = 0.0, 10.0
        dx, dy = 1.0, 0.0
        length = 25.0  # Will repeat 4 times
        offsets = [0.0, 1.0]
        colors = [(255, 0, 0, 255), (0, 0, 255, 255)]

        rust_result = vectorstag_rust.create_linear_gradient_image(
            width, height, 0, 0, x1, y1, dx, dy, length,
            offsets, colors, 1.0, 1  # repeat
        )

        # Check that gradient repeats
        assert rust_result.shape == (height, width, 4)
        # First quarter and third quarter should look similar
        diff = np.abs(rust_result[:, 0:25, :].astype(np.int16) -
                     rust_result[:, 50:75, :].astype(np.int16))
        assert np.mean(diff) < 5  # Allow some numerical differences


class TestCreateRadialGradientImage:
    """Tests for create_radial_gradient_image Rust function."""

    def test_simple_radial(self):
        """Test simple radial gradient with identity transform."""
        width, height = 100, 100
        cx, cy = 50.0, 50.0
        r = 50.0
        # Identity inverse transform
        inv_a, inv_b, inv_c, inv_d, inv_e, inv_f = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
        offsets = [0.0, 1.0]
        colors = [(255, 255, 255, 255), (0, 0, 0, 255)]

        rust_result = vectorstag_rust.create_radial_gradient_image(
            width, height, 0, 0,
            cx, cy, r,
            inv_a, inv_b, inv_c, inv_d, inv_e, inv_f,
            offsets, colors, 1.0, 0
        )

        python_result = python_create_radial_gradient_image(
            width, height, 0, 0,
            cx, cy, r,
            inv_a, inv_b, inv_c, inv_d, inv_e, inv_f,
            offsets, colors, 1.0, 0
        )

        diff = np.abs(rust_result.astype(np.int16) - python_result.astype(np.int16))
        assert np.max(diff) <= 1, f"Max diff: {np.max(diff)}"

    def test_radial_center_white(self):
        """Test that center of radial gradient has first stop color."""
        width, height = 100, 100
        cx, cy = 50.0, 50.0
        r = 40.0
        inv_a, inv_b, inv_c, inv_d, inv_e, inv_f = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
        offsets = [0.0, 1.0]
        colors = [(255, 0, 0, 255), (0, 0, 255, 255)]  # Red center, blue edge

        rust_result = vectorstag_rust.create_radial_gradient_image(
            width, height, 0, 0,
            cx, cy, r,
            inv_a, inv_b, inv_c, inv_d, inv_e, inv_f,
            offsets, colors, 1.0, 0
        )

        # Center should be red
        center_color = rust_result[50, 50]
        assert center_color[0] >= 250  # Red
        assert center_color[2] <= 5    # Not blue


class TestSampleCubicBezier:
    """Tests for sample_cubic_bezier Rust function."""

    def test_straight_line(self):
        """Test bezier that forms a straight line."""
        # Control points on a line
        x0, y0 = 0.0, 0.0
        x1, y1 = 33.3, 33.3
        x2, y2 = 66.6, 66.6
        x3, y3 = 100.0, 100.0
        n_samples = 10

        rust_points = vectorstag_rust.sample_cubic_bezier(
            x0, y0, x1, y1, x2, y2, x3, y3, n_samples
        )
        python_points = python_sample_cubic_bezier(
            x0, y0, x1, y1, x2, y2, x3, y3, n_samples
        )

        assert len(rust_points) == len(python_points)

        for (rx, ry), (px, py) in zip(rust_points, python_points):
            assert abs(rx - px) < 0.01
            assert abs(ry - py) < 0.01

    def test_curved_bezier(self):
        """Test actual curved bezier."""
        x0, y0 = 0.0, 0.0
        x1, y1 = 0.0, 100.0
        x2, y2 = 100.0, 100.0
        x3, y3 = 100.0, 0.0
        n_samples = 20

        rust_points = vectorstag_rust.sample_cubic_bezier(
            x0, y0, x1, y1, x2, y2, x3, y3, n_samples
        )
        python_points = python_sample_cubic_bezier(
            x0, y0, x1, y1, x2, y2, x3, y3, n_samples
        )

        assert len(rust_points) == n_samples

        for (rx, ry), (px, py) in zip(rust_points, python_points):
            assert abs(rx - px) < 0.01
            assert abs(ry - py) < 0.01


class TestSampleQuadraticBezier:
    """Tests for sample_quadratic_bezier Rust function."""

    def test_straight_line(self):
        """Test quadratic bezier forming a straight line."""
        x0, y0 = 0.0, 0.0
        x1, y1 = 50.0, 50.0
        x2, y2 = 100.0, 100.0
        n_samples = 10

        rust_points = vectorstag_rust.sample_quadratic_bezier(
            x0, y0, x1, y1, x2, y2, n_samples
        )
        python_points = python_sample_quadratic_bezier(
            x0, y0, x1, y1, x2, y2, n_samples
        )

        assert len(rust_points) == len(python_points)

        for (rx, ry), (px, py) in zip(rust_points, python_points):
            assert abs(rx - px) < 0.01
            assert abs(ry - py) < 0.01

    def test_curved_quadratic(self):
        """Test curved quadratic bezier."""
        x0, y0 = 0.0, 0.0
        x1, y1 = 50.0, 100.0
        x2, y2 = 100.0, 0.0
        n_samples = 20

        rust_points = vectorstag_rust.sample_quadratic_bezier(
            x0, y0, x1, y1, x2, y2, n_samples
        )
        python_points = python_sample_quadratic_bezier(
            x0, y0, x1, y1, x2, y2, n_samples
        )

        for (rx, ry), (px, py) in zip(rust_points, python_points):
            assert abs(rx - px) < 0.01
            assert abs(ry - py) < 0.01


class TestIsSelfIntersecting:
    """Tests for is_self_intersecting Rust function."""

    def test_simple_square_not_intersecting(self):
        """Test that a simple square is not self-intersecting."""
        points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        assert vectorstag_rust.is_self_intersecting(points) == False

    def test_figure_eight_intersecting(self):
        """Test that a figure-8 shape is self-intersecting."""
        points = [
            (0.0, 0.0), (100.0, 100.0),
            (100.0, 0.0), (0.0, 100.0)
        ]
        assert vectorstag_rust.is_self_intersecting(points) == True

    def test_triangle_not_intersecting(self):
        """Test that a simple triangle is not self-intersecting."""
        points = [(50.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        assert vectorstag_rust.is_self_intersecting(points) == False


class TestRenderStrokeClosedPolygon:
    """Tests for render_stroke_closed_polygon Rust function."""

    def test_square_stroke(self):
        """Test stroking a square."""
        points = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
        half_width = 5.0
        miterlimit = 4.0
        width, height = 100, 100

        rust_result = vectorstag_rust.render_stroke_closed_polygon(
            points, half_width, miterlimit, width, height, 0, 0, "miter"
        )

        assert rust_result.shape == (height, width)
        # Stroke should be visible
        assert np.any(rust_result == 255)
        # Center should be empty (not filled)
        assert rust_result[50, 50] == 0
        # Edge should be filled
        assert rust_result[20, 50] == 255  # Top edge

    def test_triangle_stroke(self):
        """Test stroking a triangle."""
        points = [(50.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
        half_width = 3.0
        miterlimit = 4.0
        width, height = 100, 100

        rust_result = vectorstag_rust.render_stroke_closed_polygon(
            points, half_width, miterlimit, width, height, 0, 0, "miter"
        )

        assert rust_result.shape == (height, width)
        assert np.any(rust_result == 255)


class TestFillPolygonsUnion:
    """Tests for fill_polygons_union Rust function."""

    def test_two_overlapping_squares(self):
        """Test union of two overlapping squares."""
        polygons = [
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
            [(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0)]
        ]
        width, height = 80, 80

        rust_result = vectorstag_rust.fill_polygons_union(polygons, width, height, 0, 0)

        assert rust_result.shape == (height, width)
        # Both squares should be filled
        assert rust_result[25, 25] == 255  # In first square only
        assert rust_result[60, 60] == 255  # In second square only
        assert rust_result[40, 40] == 255  # In overlap


# =============================================================================
# Integration Tests - Full SVG Rendering Comparison
# =============================================================================

class TestRendererIntegration:
    """Integration tests comparing Rust-accelerated rendering to pure Python."""

    def test_simple_rect_render(self):
        """Test rendering a simple SVG rectangle."""
        from vectorstag import SVGRenderer

        svg = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" fill="red"/>
        </svg>'''

        renderer = SVGRenderer(antialias=1)
        img = renderer.render(svg)

        assert img is not None
        assert img.size == (100, 100)

        # Check center is red
        arr = np.array(img)
        assert arr[50, 50, 0] > 200  # Red channel

    def test_gradient_rect_render(self):
        """Test rendering a rectangle with linear gradient."""
        from vectorstag import SVGRenderer

        svg = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:rgb(255,0,0)"/>
                    <stop offset="100%" style="stop-color:rgb(0,0,255)"/>
                </linearGradient>
            </defs>
            <rect x="0" y="0" width="100" height="100" fill="url(#grad)"/>
        </svg>'''

        renderer = SVGRenderer(antialias=1)
        img = renderer.render(svg)

        assert img is not None
        arr = np.array(img)

        # Left should be red
        assert arr[50, 10, 0] > arr[50, 10, 2]
        # Right should be blue
        assert arr[50, 90, 2] > arr[50, 90, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
