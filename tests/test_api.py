"""
Comprehensive pytest tests for VectorStag public API.

Tests cover:
- Simple conversion functions: svg_to_pil, svg_to_numpy, svg_to_opencv, svg_to_bytes
- File-based functions: file_to_pil, file_to_numpy, file_to_opencv
- Render-to-target functions: render_to_pil, render_to_numpy, render_to_opencv
- Helper functions: _normalize_svg, _normalize_background
- Various antialias levels, background colors, and edge cases
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import io
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import vectorstag
from vectorstag import (
    svg_to_pil, svg_to_numpy, svg_to_opencv, svg_to_bytes,
    file_to_pil, file_to_numpy, file_to_opencv,
    render_to_pil, render_to_numpy, render_to_opencv,
    SVGRenderer,
    _normalize_svg, _normalize_background
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_svg():
    """A simple red rectangle SVG."""
    return '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="10" width="80" height="80" fill="red"/>
    </svg>'''


@pytest.fixture
def circle_svg():
    """A blue circle SVG."""
    return '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="40" fill="blue"/>
    </svg>'''


@pytest.fixture
def gradient_svg():
    """SVG with linear gradient."""
    return '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:red;stop-opacity:1" />
                <stop offset="100%" style="stop-color:blue;stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect x="0" y="0" width="100" height="100" fill="url(#grad1)"/>
    </svg>'''


@pytest.fixture
def transparent_svg():
    """SVG with transparency."""
    return '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="10" width="80" height="80" fill="green" fill-opacity="0.5"/>
    </svg>'''


@pytest.fixture
def temp_svg_file(simple_svg):
    """Create a temporary SVG file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
        f.write(simple_svg)
        return Path(f.name)


# =============================================================================
# Test _normalize_svg helper
# =============================================================================

class TestNormalizeSvg:
    """Tests for _normalize_svg helper function."""

    def test_string_input(self, simple_svg):
        """String input should pass through unchanged."""
        result = _normalize_svg(simple_svg)
        assert result == simple_svg
        assert isinstance(result, str)

    def test_bytes_input(self, simple_svg):
        """Bytes input should be decoded to string."""
        svg_bytes = simple_svg.encode('utf-8')
        result = _normalize_svg(svg_bytes)
        assert result == simple_svg
        assert isinstance(result, str)

    def test_path_input(self, temp_svg_file, simple_svg):
        """Path input should read file contents."""
        result = _normalize_svg(temp_svg_file)
        assert result == simple_svg
        assert isinstance(result, str)
        # Cleanup
        temp_svg_file.unlink()


# =============================================================================
# Test _normalize_background helper
# =============================================================================

class TestNormalizeBackground:
    """Tests for _normalize_background helper function."""

    def test_none_returns_white(self):
        """None should return white opaque."""
        result = _normalize_background(None)
        assert result == (255, 255, 255, 255)

    def test_transparent_string(self):
        """'transparent' string should return fully transparent."""
        result = _normalize_background('transparent')
        assert result == (0, 0, 0, 0)

    def test_transparent_uppercase(self):
        """'TRANSPARENT' should also work."""
        result = _normalize_background('TRANSPARENT')
        assert result == (0, 0, 0, 0)

    def test_hex_color_6_digits(self):
        """6-digit hex color."""
        result = _normalize_background('#ff0000')
        assert result == (255, 0, 0, 255)

    def test_hex_color_3_digits(self):
        """3-digit hex color (shorthand)."""
        result = _normalize_background('#f00')
        assert result == (255, 0, 0, 255)

    def test_hex_color_8_digits(self):
        """8-digit hex color with alpha."""
        result = _normalize_background('#ff000080')
        assert result == (255, 0, 0, 128)

    def test_rgb_tuple(self):
        """RGB tuple should get full opacity."""
        result = _normalize_background((255, 128, 0))
        assert result == (255, 128, 0, 255)

    def test_rgba_tuple(self):
        """RGBA tuple should pass through."""
        result = _normalize_background((255, 128, 0, 64))
        assert result == (255, 128, 0, 64)

    def test_rgb_list(self):
        """RGB list should work like tuple."""
        result = _normalize_background([100, 150, 200])
        assert result == (100, 150, 200, 255)

    def test_rgba_list(self):
        """RGBA list should work like tuple."""
        result = _normalize_background([100, 150, 200, 50])
        assert result == (100, 150, 200, 50)

    def test_invalid_returns_white(self):
        """Invalid input should return white."""
        result = _normalize_background('invalid_color')
        assert result == (255, 255, 255, 255)


# =============================================================================
# Test svg_to_pil
# =============================================================================

class TestSvgToPil:
    """Tests for svg_to_pil function."""

    def test_basic_render(self, simple_svg):
        """Basic rendering returns PIL Image."""
        img = svg_to_pil(simple_svg)
        assert isinstance(img, Image.Image)
        assert img.mode == 'RGBA'

    def test_explicit_dimensions(self, simple_svg):
        """Rendering with explicit dimensions."""
        img = svg_to_pil(simple_svg, width=200, height=150)
        assert img.size == (200, 150)

    def test_width_only(self, simple_svg):
        """Rendering with width only scales height proportionally."""
        img = svg_to_pil(simple_svg, width=200)
        # SVG is 100x100, so height should also scale to 200
        assert img.size[0] == 200

    def test_height_only(self, simple_svg):
        """Rendering with height only scales width proportionally."""
        img = svg_to_pil(simple_svg, height=200)
        assert img.size[1] == 200

    def test_transparent_background(self, simple_svg):
        """Rendering with transparent background."""
        img = svg_to_pil(simple_svg, width=100, background='transparent')
        arr = np.array(img)
        # Corner pixel should be transparent
        assert arr[0, 0, 3] == 0

    def test_colored_background(self, simple_svg):
        """Rendering with colored background."""
        img = svg_to_pil(simple_svg, width=100, background='#00ff00')
        arr = np.array(img)
        # Corner pixel should be green (not red)
        assert arr[0, 0, 0] == 0   # R
        assert arr[0, 0, 1] == 255  # G
        assert arr[0, 0, 2] == 0   # B

    def test_antialias_1(self, circle_svg):
        """Rendering with no antialiasing."""
        img = svg_to_pil(circle_svg, width=100, antialias=1)
        assert img.size == (100, 100)

    def test_antialias_2(self, circle_svg):
        """Rendering with 2x antialiasing."""
        img = svg_to_pil(circle_svg, width=100, antialias=2)
        assert img.size == (100, 100)

    def test_antialias_4(self, circle_svg):
        """Rendering with 4x antialiasing (high quality)."""
        img = svg_to_pil(circle_svg, width=100, antialias=4)
        assert img.size == (100, 100)

    def test_scale_factor(self, simple_svg):
        """Rendering with scale factor."""
        img = svg_to_pil(simple_svg, scale=2.0)
        # Default 100x100 scaled by 2 = 200x200
        assert img.size == (200, 200)

    def test_bytes_input(self, simple_svg):
        """Rendering from bytes input."""
        svg_bytes = simple_svg.encode('utf-8')
        img = svg_to_pil(svg_bytes)
        assert isinstance(img, Image.Image)

    def test_path_input(self, temp_svg_file):
        """Rendering from Path input."""
        img = svg_to_pil(temp_svg_file)
        assert isinstance(img, Image.Image)
        temp_svg_file.unlink()

    def test_gradient_render(self, gradient_svg):
        """Rendering SVG with gradient."""
        img = svg_to_pil(gradient_svg, width=100, height=100)
        arr = np.array(img)
        # Left side should be more red, right side more blue
        left_red = arr[50, 10, 0]
        right_red = arr[50, 90, 0]
        assert left_red > right_red

    def test_transparent_svg(self, transparent_svg):
        """Rendering SVG with semi-transparent fill."""
        img = svg_to_pil(transparent_svg, width=100, background='transparent')
        arr = np.array(img)
        # Center pixel should have some transparency
        center_alpha = arr[50, 50, 3]
        assert 0 < center_alpha < 255


# =============================================================================
# Test svg_to_numpy
# =============================================================================

class TestSvgToNumpy:
    """Tests for svg_to_numpy function."""

    def test_basic_render(self, simple_svg):
        """Basic rendering returns numpy array."""
        arr = svg_to_numpy(simple_svg)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8

    def test_rgba_shape(self, simple_svg):
        """Output has correct RGBA shape."""
        arr = svg_to_numpy(simple_svg, width=100, height=100)
        assert arr.shape == (100, 100, 4)

    def test_explicit_dtype(self, simple_svg):
        """Rendering with explicit dtype."""
        arr = svg_to_numpy(simple_svg, width=50, dtype=np.float32)
        assert arr.dtype == np.float32

    def test_red_rectangle(self, simple_svg):
        """Verify red color in the rectangle area."""
        arr = svg_to_numpy(simple_svg, width=100, height=100, background='transparent')
        # Center of rectangle should be red
        center = arr[50, 50]
        assert center[0] > 200  # R
        assert center[1] < 50   # G
        assert center[2] < 50   # B
        assert center[3] > 200  # A (opaque)

    def test_antialias_levels(self, circle_svg):
        """Test different antialias levels produce valid output."""
        for aa in [1, 2, 4, 8]:
            arr = svg_to_numpy(circle_svg, width=50, height=50, antialias=aa)
            assert arr.shape == (50, 50, 4)
            assert arr.dtype == np.uint8


# =============================================================================
# Test svg_to_opencv
# =============================================================================

class TestSvgToOpencv:
    """Tests for svg_to_opencv function."""

    def test_bgra_output(self, simple_svg):
        """Default output is BGRA (4 channels)."""
        arr = svg_to_opencv(simple_svg, width=100, height=100)
        assert arr.shape == (100, 100, 4)

    def test_bgr_output(self, simple_svg):
        """Output with alpha=False is BGR (3 channels)."""
        arr = svg_to_opencv(simple_svg, width=100, height=100, alpha=False)
        assert arr.shape == (100, 100, 3)

    def test_color_channel_order(self, simple_svg):
        """Verify BGR channel order (red rect should have high B channel at index 2)."""
        # Note: svg has red rectangle, in BGR format Red is at index 2
        arr = svg_to_opencv(simple_svg, width=100, height=100, background='transparent')
        center = arr[50, 50]
        # In BGRA: B=0, G=1, R=2, A=3
        assert center[2] > 200  # R channel (index 2 in BGR)
        assert center[1] < 50   # G channel
        assert center[0] < 50   # B channel

    def test_bgr_color_order(self, simple_svg):
        """Verify BGR output has correct channel order."""
        arr = svg_to_opencv(simple_svg, width=100, height=100, alpha=False,
                            background='white')
        center = arr[50, 50]
        # Red in BGR: B=0, G=0, R=255
        assert center[2] > 200  # R
        assert center[0] < 100  # B (might have some blending)


# =============================================================================
# Test svg_to_bytes
# =============================================================================

class TestSvgToBytes:
    """Tests for svg_to_bytes function."""

    def test_png_output(self, simple_svg):
        """Default PNG output."""
        data = svg_to_bytes(simple_svg, width=100)
        assert isinstance(data, bytes)
        assert len(data) > 0
        # Check PNG magic bytes
        assert data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_jpeg_output(self, simple_svg):
        """JPEG output."""
        data = svg_to_bytes(simple_svg, width=100, format='JPEG')
        assert isinstance(data, bytes)
        # Check JPEG magic bytes
        assert data[:2] == b'\xff\xd8'

    def test_webp_output(self, simple_svg):
        """WEBP output."""
        data = svg_to_bytes(simple_svg, width=100, format='WEBP')
        assert isinstance(data, bytes)
        # Check WEBP magic bytes (RIFF header)
        assert data[:4] == b'RIFF'
        assert data[8:12] == b'WEBP'

    def test_png_roundtrip(self, simple_svg):
        """PNG can be loaded back by PIL."""
        data = svg_to_bytes(simple_svg, width=100, format='PNG')
        img = Image.open(io.BytesIO(data))
        assert img.size == (100, 100)
        assert img.mode == 'RGBA'

    def test_jpeg_quality(self, simple_svg):
        """JPEG with quality parameter."""
        low_q = svg_to_bytes(simple_svg, width=100, format='JPEG', quality=10)
        high_q = svg_to_bytes(simple_svg, width=100, format='JPEG', quality=95)
        # Higher quality should produce larger file
        assert len(high_q) > len(low_q)


# =============================================================================
# Test file_to_* functions
# =============================================================================

class TestFileFunctions:
    """Tests for file_to_pil, file_to_numpy, file_to_opencv."""

    def test_file_to_pil(self, temp_svg_file):
        """file_to_pil renders from file path."""
        img = file_to_pil(temp_svg_file)
        assert isinstance(img, Image.Image)
        temp_svg_file.unlink()

    def test_file_to_pil_string_path(self, temp_svg_file):
        """file_to_pil works with string path."""
        img = file_to_pil(str(temp_svg_file))
        assert isinstance(img, Image.Image)
        temp_svg_file.unlink()

    def test_file_to_numpy(self, temp_svg_file):
        """file_to_numpy renders from file path."""
        arr = file_to_numpy(temp_svg_file)
        assert isinstance(arr, np.ndarray)
        assert arr.shape[-1] == 4  # RGBA
        temp_svg_file.unlink()

    def test_file_to_opencv(self, temp_svg_file):
        """file_to_opencv renders from file path."""
        arr = file_to_opencv(temp_svg_file)
        assert isinstance(arr, np.ndarray)
        assert arr.shape[-1] == 4  # BGRA
        temp_svg_file.unlink()

    def test_file_to_opencv_bgr(self, temp_svg_file):
        """file_to_opencv with alpha=False."""
        arr = file_to_opencv(temp_svg_file, alpha=False)
        assert arr.shape[-1] == 3  # BGR
        temp_svg_file.unlink()


# =============================================================================
# Test render_to_* functions
# =============================================================================

class TestRenderToTarget:
    """Tests for render_to_pil, render_to_numpy, render_to_opencv."""

    def test_render_to_pil_basic(self, simple_svg):
        """Basic render_to_pil onto existing canvas."""
        canvas = Image.new('RGBA', (200, 200), (255, 255, 255, 255))
        render_to_pil(simple_svg, canvas, x=50, y=50, width=50, height=50)

        arr = np.array(canvas)
        # Position (75, 75) should be red (center of 50x50 icon at offset 50,50)
        assert arr[75, 75, 0] > 200  # R
        # Position (10, 10) should still be white background
        assert arr[10, 10, 0] == 255  # R
        assert arr[10, 10, 1] == 255  # G
        assert arr[10, 10, 2] == 255  # B

    def test_render_to_pil_preserves_target(self, simple_svg):
        """render_to_pil preserves existing content via alpha compositing."""
        canvas = Image.new('RGBA', (200, 200), (0, 0, 255, 255))  # Blue canvas
        render_to_pil(simple_svg, canvas, x=50, y=50, width=50, height=50)

        arr = np.array(canvas)
        # Corners should still be blue
        assert arr[10, 10, 2] == 255  # B
        assert arr[10, 10, 0] == 0    # R

    def test_render_to_numpy_basic(self, simple_svg):
        """Basic render_to_numpy onto existing array."""
        canvas = np.zeros((200, 200, 4), dtype=np.uint8)
        canvas[:, :] = [255, 255, 255, 255]  # White

        render_to_numpy(simple_svg, canvas, x=50, y=50, width=50, height=50)

        # Center of icon should be red
        assert canvas[75, 75, 0] > 200  # R

    def test_render_to_numpy_rgb(self, simple_svg):
        """render_to_numpy works on RGB (3 channel) array."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        canvas[:, :] = [255, 255, 255]  # White

        render_to_numpy(simple_svg, canvas, x=50, y=50, width=50, height=50)

        # Center of icon should be red
        assert canvas[75, 75, 0] > 200  # R

    def test_render_to_opencv_basic(self, simple_svg):
        """Basic render_to_opencv onto existing BGRA array."""
        canvas = np.zeros((200, 200, 4), dtype=np.uint8)
        canvas[:, :] = [255, 255, 255, 255]  # White in BGRA

        render_to_opencv(simple_svg, canvas, x=50, y=50, width=50, height=50)

        # Center of icon should be red (R is at index 2 in BGRA)
        assert canvas[75, 75, 2] > 200  # R channel

    def test_render_to_opencv_bgr(self, simple_svg):
        """render_to_opencv works on BGR (3 channel) array."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        canvas[:, :] = [255, 255, 255]  # White in BGR

        render_to_opencv(simple_svg, canvas, x=50, y=50, width=50, height=50)

        # Center of icon should be red (R is at index 2 in BGR)
        assert canvas[75, 75, 2] > 200

    def test_render_out_of_bounds(self, simple_svg):
        """Rendering outside target bounds is handled gracefully."""
        canvas = np.zeros((100, 100, 4), dtype=np.uint8)

        # Render completely outside - should not crash
        render_to_numpy(simple_svg, canvas, x=200, y=200, width=50, height=50)

        # Canvas should be unchanged
        assert canvas.sum() == 0

    def test_render_partial_clip(self, simple_svg):
        """Rendering partially outside target is clipped."""
        canvas = np.zeros((100, 100, 4), dtype=np.uint8)
        canvas[:, :] = [255, 255, 255, 255]

        # Render with icon extending beyond right edge
        render_to_numpy(simple_svg, canvas, x=80, y=40, width=50, height=50)

        # Should have rendered what fits
        # (80+20 = 100, so center at x=100+25=?? - well, it's clipped)
        # Just verify no crash and canvas was modified
        assert True  # If we get here, no crash

    def test_render_default_dimensions(self, simple_svg):
        """render_to_* uses remaining space when dimensions not specified."""
        canvas = Image.new('RGBA', (200, 200), (255, 255, 255, 255))
        render_to_pil(simple_svg, canvas, x=50, y=50)  # No width/height

        # Should fill from (50,50) to (200,200) = 150x150
        arr = np.array(canvas)
        # Just verify it rendered something
        assert arr[100, 100, 0] > 0 or arr[100, 100, 1] > 0 or arr[100, 100, 2] > 0

    def test_render_to_numpy_default_dimensions(self, simple_svg):
        """render_to_numpy uses remaining space when dimensions not specified."""
        canvas = np.zeros((200, 200, 4), dtype=np.uint8)
        canvas[:, :] = [255, 255, 255, 255]

        # No width/height - should use remaining space
        render_to_numpy(simple_svg, canvas, x=50, y=50)
        # Verify something was rendered
        assert canvas[100, 100].sum() < 1020  # Not all white anymore

    def test_render_to_opencv_default_dimensions(self, simple_svg):
        """render_to_opencv uses remaining space when dimensions not specified."""
        canvas = np.zeros((200, 200, 4), dtype=np.uint8)
        canvas[:, :] = [255, 255, 255, 255]

        # No width/height - should use remaining space
        render_to_opencv(simple_svg, canvas, x=50, y=50)
        # Verify something was rendered (center should be red in BGRA)
        assert canvas[100, 100, 2] > 200  # R channel in BGRA

    def test_render_to_opencv_out_of_bounds(self, simple_svg):
        """render_to_opencv handles out of bounds gracefully."""
        canvas = np.zeros((100, 100, 4), dtype=np.uint8)
        render_to_opencv(simple_svg, canvas, x=150, y=150, width=50, height=50)
        # Should do nothing, no crash
        assert canvas.sum() == 0

    def test_render_negative_result_size(self, simple_svg):
        """render_to_* handles when result size would be negative."""
        canvas = np.zeros((100, 100, 4), dtype=np.uint8)
        # Width larger than remaining space
        render_to_numpy(simple_svg, canvas, x=90, y=90, width=50, height=50)
        # Should just render what fits, no crash
        assert True


# =============================================================================
# Test SVGRenderer class
# =============================================================================

class TestSVGRendererClass:
    """Tests for the SVGRenderer class."""

    def test_init_defaults(self):
        """Default initialization."""
        renderer = SVGRenderer()
        assert renderer.scale == 1.0
        assert renderer.antialias >= 1
        assert renderer.background is not None

    def test_init_custom(self):
        """Custom initialization parameters."""
        renderer = SVGRenderer(scale=2.0, antialias=4, background=(0, 0, 0, 0))
        assert renderer.scale == 2.0
        assert renderer.antialias == 4
        assert renderer.background == (0, 0, 0, 0)

    def test_render_method(self, simple_svg):
        """render() method works."""
        renderer = SVGRenderer()
        img = renderer.render(simple_svg, width=100, height=100)
        assert isinstance(img, Image.Image)
        assert img.size == (100, 100)

    def test_render_file_method(self, temp_svg_file):
        """render_file() method works."""
        renderer = SVGRenderer()
        img = renderer.render_file(str(temp_svg_file), width=100, height=100)
        assert isinstance(img, Image.Image)
        temp_svg_file.unlink()


# =============================================================================
# Test edge cases and error handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_svg(self):
        """Empty SVG content."""
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
        img = svg_to_pil(svg)
        assert isinstance(img, Image.Image)

    def test_zero_dimensions(self, simple_svg):
        """Zero dimensions should handle gracefully."""
        # This might raise or return small image - just shouldn't crash
        try:
            img = svg_to_pil(simple_svg, width=0, height=0)
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable to raise error

    def test_very_large_dimensions(self, simple_svg):
        """Very large dimensions should work (within reason)."""
        img = svg_to_pil(simple_svg, width=1000, height=1000, antialias=1)
        assert img.size == (1000, 1000)

    def test_negative_coordinates_svg(self):
        """SVG with negative viewBox coordinates."""
        svg = '''<svg viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="0" cy="0" r="40" fill="red"/>
        </svg>'''
        img = svg_to_pil(svg, width=100, height=100)
        assert isinstance(img, Image.Image)

    def test_complex_svg(self):
        """SVG with multiple elements and transforms."""
        svg = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <g transform="translate(50, 50)">
                <rect x="-20" y="-20" width="40" height="40" fill="red" transform="rotate(45)"/>
                <circle cx="0" cy="0" r="10" fill="blue"/>
            </g>
        </svg>'''
        img = svg_to_pil(svg, width=100, height=100)
        assert isinstance(img, Image.Image)

    def test_unicode_text_svg(self):
        """SVG with Unicode text content."""
        svg = '''<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="50" font-size="20">Hello 世界 🌍</text>
        </svg>'''
        img = svg_to_pil(svg, width=200)
        assert isinstance(img, Image.Image)


# =============================================================================
# Test alpha composite helpers (internal functions)
# =============================================================================

class TestAlphaCompositeHelpers:
    """Tests for internal alpha composite helper functions."""

    def test_alpha_composite_numpy_out_of_bounds(self):
        """_alpha_composite_numpy handles completely out of bounds overlay."""
        from vectorstag import _alpha_composite_numpy
        target = np.zeros((100, 100, 4), dtype=np.uint8)
        overlay = np.ones((50, 50, 4), dtype=np.uint8) * 255
        # Place overlay completely outside target
        _alpha_composite_numpy(target, overlay, 200, 200)
        # Target should be unchanged
        assert target.sum() == 0

    def test_alpha_composite_opencv_out_of_bounds(self):
        """_alpha_composite_opencv handles completely out of bounds overlay."""
        from vectorstag import _alpha_composite_opencv
        target = np.zeros((100, 100, 4), dtype=np.uint8)
        overlay = np.ones((50, 50, 4), dtype=np.uint8) * 255
        # Place overlay completely outside target
        _alpha_composite_opencv(target, overlay, 200, 200)
        # Target should be unchanged
        assert target.sum() == 0

    def test_alpha_composite_negative_offset(self):
        """_alpha_composite handles negative offsets gracefully."""
        from vectorstag import _alpha_composite_numpy
        target = np.zeros((100, 100, 4), dtype=np.uint8)
        overlay = np.ones((50, 50, 4), dtype=np.uint8) * 255
        # Negative offset that would make region empty
        _alpha_composite_numpy(target, overlay, -100, -100)
        # Should not crash, may or may not modify target


# =============================================================================
# Test module exports
# =============================================================================

class TestModuleExports:
    """Test that all expected functions are exported."""

    def test_version(self):
        """Module has version string."""
        assert hasattr(vectorstag, '__version__')
        assert isinstance(vectorstag.__version__, str)

    def test_exports(self):
        """All expected functions are exported."""
        expected = [
            'SVGRenderer', 'SVGParser',
            'svg_to_pil', 'svg_to_numpy', 'svg_to_opencv', 'svg_to_bytes',
            'file_to_pil', 'file_to_numpy', 'file_to_opencv',
            'render_to_pil', 'render_to_numpy', 'render_to_opencv',
        ]
        for name in expected:
            assert hasattr(vectorstag, name), f"Missing export: {name}"


# =============================================================================
# Performance sanity checks
# =============================================================================

class TestPerformance:
    """Basic performance sanity checks."""

    def test_render_time_reasonable(self, simple_svg):
        """Rendering should complete in reasonable time."""
        import time
        start = time.time()
        for _ in range(10):
            svg_to_pil(simple_svg, width=100, height=100, antialias=2)
        elapsed = time.time() - start
        # 10 renders should take less than 5 seconds
        assert elapsed < 5.0, f"Rendering too slow: {elapsed:.2f}s for 10 renders"

    def test_antialias_8x_works(self, simple_svg):
        """8x antialiasing should work (may be slower)."""
        img = svg_to_pil(simple_svg, width=100, height=100, antialias=8)
        assert img.size == (100, 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
