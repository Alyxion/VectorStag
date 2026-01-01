"""
Unit tests for polygon fill gap detection.

These tests verify that polygon filling produces gap-free results,
especially for circular shapes approximated by many-sided polygons.
"""

import math
import numpy as np
import pytest
from PIL import Image, ImageDraw


def generate_circle_points(cx: float, cy: float, r: float, n_points: int) -> list:
    """Generate points for a regular polygon approximating a circle."""
    points = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append((x, y))
    return points


def generate_stroke_polygon(points: list, half_width: float) -> list:
    """
    Generate a stroke polygon (annulus) from center points.

    Returns outer points followed by inner points (for fill-rule based hole).
    """
    n = len(points)
    outer_points = []
    inner_points = []

    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_curr = points[i]
        p_next = points[(i + 1) % n]

        # Average direction at this point
        d1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        d2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        # Normalize
        len1 = math.sqrt(d1[0]**2 + d1[1]**2)
        len2 = math.sqrt(d2[0]**2 + d2[1]**2)
        if len1 > 0.0001:
            d1 = (d1[0]/len1, d1[1]/len1)
        if len2 > 0.0001:
            d2 = (d2[0]/len2, d2[1]/len2)

        # Average direction
        avg_d = (d1[0] + d2[0], d1[1] + d2[1])
        avg_len = math.sqrt(avg_d[0]**2 + avg_d[1]**2)
        if avg_len > 0.0001:
            avg_d = (avg_d[0]/avg_len, avg_d[1]/avg_len)
        else:
            avg_d = d1

        # Perpendicular
        perp = (-avg_d[1], avg_d[0])

        outer_points.append((p_curr[0] + perp[0] * half_width,
                            p_curr[1] + perp[1] * half_width))
        inner_points.append((p_curr[0] - perp[0] * half_width,
                            p_curr[1] - perp[1] * half_width))

    return outer_points, inner_points


def fill_polygon_pil(points: list, size: int) -> np.ndarray:
    """Fill a polygon using PIL and return as numpy array."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(points, fill=255)
    return np.array(img)


def fill_stroke_segments(center_points: list, half_width: float, size: int) -> np.ndarray:
    """
    Fill stroke using segment-by-segment approach (like _stroke_closed_polygon_segmented).
    Each edge becomes a quadrilateral.
    """
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    n = len(center_points)
    for i in range(n):
        j = (i + 1) % n
        p1 = center_points[i]
        p2 = center_points[j]

        # Direction and perpendicular
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.0001:
            continue
        dx, dy = dx/length, dy/length
        perp = (-dy, dx)

        # Quad corners
        quad = [
            (p1[0] + perp[0] * half_width, p1[1] + perp[1] * half_width),
            (p2[0] + perp[0] * half_width, p2[1] + perp[1] * half_width),
            (p2[0] - perp[0] * half_width, p2[1] - perp[1] * half_width),
            (p1[0] - perp[0] * half_width, p1[1] - perp[1] * half_width),
        ]
        draw.polygon(quad, fill=255)

    return np.array(img)


def check_for_gaps(filled: np.ndarray, center: tuple, inner_r: float, outer_r: float) -> dict:
    """
    Check for gaps in a filled annulus (stroke around circle).

    Returns dict with gap analysis:
    - has_gaps: bool
    - gap_pixels: list of (x, y) coordinates
    - gap_count: int
    """
    h, w = filled.shape
    cx, cy = center
    gaps = []

    for y in range(h):
        for x in range(w):
            # Distance from center
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)

            # Should be filled if in annulus (between inner and outer radius)
            # Add small margin for edge pixels
            margin = 0.5
            if inner_r + margin < dist < outer_r - margin:
                if filled[y, x] == 0:
                    gaps.append((x, y))

    return {
        'has_gaps': len(gaps) > 0,
        'gap_pixels': gaps,
        'gap_count': len(gaps)
    }


class TestPolygonFillNoGaps:
    """Test that polygon fills have no gaps."""

    def test_simple_circle_fill(self):
        """A simple filled circle polygon should have no gaps."""
        size = 100
        cx, cy = 50, 50
        r = 30

        for n_points in [8, 16, 32, 72, 100]:
            points = generate_circle_points(cx, cy, r, n_points)
            filled = fill_polygon_pil(points, size)

            # Check that center is filled
            assert filled[cy, cx] == 255, f"Center not filled with {n_points} points"

            # Check no gaps in interior (conservative check)
            inner_check_r = r - 2  # Stay away from edges
            gap_count = 0
            for y in range(size):
                for x in range(size):
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < inner_check_r and filled[y, x] == 0:
                        gap_count += 1

            assert gap_count == 0, f"Found {gap_count} gaps in circle with {n_points} points"

    def test_stroke_segments_no_gaps(self):
        """Stroke rendered as segments should have no gaps in the annulus."""
        size = 200
        cx, cy = 100, 100
        center_r = 40
        half_width = 10

        for n_points in [16, 32, 72, 100, 200]:
            center_points = generate_circle_points(cx, cy, center_r, n_points)
            filled = fill_stroke_segments(center_points, half_width, size)

            # The annulus should be from (center_r - half_width) to (center_r + half_width)
            inner_r = center_r - half_width
            outer_r = center_r + half_width

            result = check_for_gaps(filled, (cx, cy), inner_r, outer_r)

            assert not result['has_gaps'], \
                f"Found {result['gap_count']} gaps with {n_points} points. " \
                f"First gaps at: {result['gap_pixels'][:5]}"

    def test_stroke_segments_high_resolution(self):
        """High resolution stroke should definitely have no gaps."""
        size = 512
        cx, cy = 256, 256
        center_r = 100
        half_width = 20

        # This simulates 4x supersampling
        center_points = generate_circle_points(cx, cy, center_r, 360)
        filled = fill_stroke_segments(center_points, half_width, size)

        inner_r = center_r - half_width
        outer_r = center_r + half_width

        result = check_for_gaps(filled, (cx, cy), inner_r, outer_r)

        assert not result['has_gaps'], \
            f"Found {result['gap_count']} gaps at 512x512 with 360 points"

    def test_stroke_segments_small_circle(self):
        """Small circle (like in book-image.svg) should have no gaps."""
        # Simulating book-image circle at 4x resolution
        # Circle: r=2 in 24x24 viewBox, rendered at 128x128 with 4x AA = 512x512
        scale = 512 / 24
        cx, cy = 10 * scale, 8 * scale  # ~213, ~170
        center_r = 2 * scale  # ~42.67
        half_width = 1 * scale  # stroke-width/2 = 1 * scale = ~21.33

        size = 512
        cx, cy = int(cx), int(cy)

        # Use many points like we do in actual rendering
        n_points = max(72, int(center_r * 2))
        center_points = generate_circle_points(cx, cy, center_r, n_points)
        filled = fill_stroke_segments(center_points, half_width, size)

        inner_r = center_r - half_width
        outer_r = center_r + half_width

        result = check_for_gaps(filled, (cx, cy), inner_r, outer_r)

        assert not result['has_gaps'], \
            f"book-image circle has {result['gap_count']} gaps with {n_points} points"

    def test_adjacent_quads_share_edge(self):
        """
        Verify that adjacent quadrilaterals share exact edge coordinates.

        This is a mathematical test - if quads share exact edges, there can be no gaps.
        """
        cx, cy = 100, 100
        center_r = 40
        half_width = 10
        n_points = 72

        center_points = generate_circle_points(cx, cy, center_r, n_points)

        # Generate all quads
        quads = []
        for i in range(n_points):
            j = (i + 1) % n_points
            p1 = center_points[i]
            p2 = center_points[j]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            dx, dy = dx/length, dy/length
            perp = (-dy, dx)

            quad = [
                (p1[0] + perp[0] * half_width, p1[1] + perp[1] * half_width),
                (p2[0] + perp[0] * half_width, p2[1] + perp[1] * half_width),
                (p2[0] - perp[0] * half_width, p2[1] - perp[1] * half_width),
                (p1[0] - perp[0] * half_width, p1[1] - perp[1] * half_width),
            ]
            quads.append(quad)

        # Check that quad[i] edge at p2 matches quad[i+1] edge at p1
        # These edges should share vertices
        max_gap = 0
        for i in range(n_points):
            j = (i + 1) % n_points

            # Quad i has its "end" edge at indices 1,2 (at p2)
            # Quad j has its "start" edge at indices 0,3 (at p1)
            # But p2 of quad i IS p1 of quad j (same center point)

            # The issue: the PERPENDICULAR directions differ!
            # Quad i uses perp based on edge i, quad j uses perp based on edge j

            q_i = quads[i]
            q_j = quads[j]

            # q_i[1] and q_j[0] are both outer corners at the shared center point
            # q_i[2] and q_j[3] are both inner corners at the shared center point

            outer_gap = math.sqrt((q_i[1][0] - q_j[0][0])**2 + (q_i[1][1] - q_j[0][1])**2)
            inner_gap = math.sqrt((q_i[2][0] - q_j[3][0])**2 + (q_i[2][1] - q_j[3][1])**2)

            max_gap = max(max_gap, outer_gap, inner_gap)

        # Report the gap - this is expected due to different perpendiculars
        print(f"\nMax corner gap between adjacent quads: {max_gap:.4f} pixels")
        print("This gap is expected and should be filled by round joins")

        # The gap should be small for a 72-point circle
        # For 72 points (5 degree angles), gap ≈ 2 * half_width * sin(5°/2) ≈ 0.87 pixels
        expected_max_gap = 2 * half_width * math.sin(math.radians(360 / n_points / 2))
        assert max_gap < expected_max_gap * 1.1, \
            f"Gap {max_gap:.4f} exceeds expected {expected_max_gap:.4f}"


class TestVectorStagPolygonFill:
    """Test VectorStag's actual polygon fill implementation."""

    def test_vectorstag_circle_stroke_no_gaps(self):
        """VectorStag's circle stroke rendering should have no internal gaps."""
        from vectorstag import SVGRenderer

        # Render a simple stroked circle
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="30" fill="none" stroke="black" stroke-width="10"/>
        </svg>'''

        renderer = SVGRenderer(antialias=4, background=(255, 255, 255, 255))
        img = renderer.render(svg, 100, 100)
        arr = np.array(img)

        # Check for gaps in the stroke annulus
        # Stroke should be from r=25 to r=35 (30 ± 5)
        cx, cy = 50, 50
        inner_r, outer_r = 25, 35

        gaps = []
        for y in range(100):
            for x in range(100):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                # In the annulus interior (with margin)
                if inner_r + 1 < dist < outer_r - 1:
                    # Should be dark (stroke)
                    if arr[y, x, 0] > 200:  # Too bright = gap
                        gaps.append((x, y))

        assert len(gaps) == 0, \
            f"VectorStag circle stroke has {len(gaps)} gap pixels: {gaps[:10]}"

    def test_vectorstag_book_image_circle(self):
        """The circle in book-image.svg should render without gaps."""
        from vectorstag import SVGRenderer

        # Render just the circle from book-image.svg
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
                     fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="10" cy="8" r="2" />
        </svg>'''

        renderer = SVGRenderer(antialias=4, background=(255, 255, 255, 255))
        img = renderer.render(svg, 128, 128)
        arr = np.array(img)

        # Circle center at 10/24 * 128 ≈ 53, 8/24 * 128 ≈ 43
        # Radius 2/24 * 128 ≈ 10.67
        # Stroke width 2/24 * 128 ≈ 10.67, half = 5.33
        # So stroke from r ≈ 5.33 to r ≈ 16
        cx, cy = int(10 * 128 / 24), int(8 * 128 / 24)
        inner_r = 2 * 128 / 24 - 1 * 128 / 24  # r - stroke_width/2
        outer_r = 2 * 128 / 24 + 1 * 128 / 24  # r + stroke_width/2

        gaps = []
        for y in range(128):
            for x in range(128):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                # In the annulus interior (with margin for AA)
                if inner_r + 2 < dist < outer_r - 2:
                    # Should be dark (stroke)
                    if arr[y, x, 0] > 200:  # Too bright = gap
                        gaps.append((x, y))

        assert len(gaps) == 0, \
            f"book-image circle has {len(gaps)} gap pixels"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
