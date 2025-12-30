"""SVG Renderer - Render parsed SVG to PIL Images."""

import math
from typing import Optional, Union, List, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter
import numpy as np

# Try to import Rust extension for performance
try:
    import vectorstag_rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from .parser import (
    SVGParser, SVGDocument, SVGElement, Transform, Style,
    RectElement, CircleElement, EllipseElement, LineElement,
    PolylineElement, PolygonElement, PathElement, GroupElement,
    TextElement, LinearGradient, RadialGradient, GradientStop,
    ClipPath, FILL_NOT_SET
)


class SVGRenderer:
    """Render SVG documents to PIL Images."""

    # Maximum recursion depth for rendering (prevents infinite loops)
    MAX_RENDER_DEPTH = 100

    def __init__(self, scale: float = 1.0, background: Optional[tuple[int, int, int, int]] = None,
                 antialias: int = 2, preserve_aspect_ratio: bool = False):
        """
        Initialize renderer.

        Args:
            scale: Scale factor for rendering
            background: Background color (RGBA). Default is white.
            antialias: Anti-aliasing factor (1=none, 2=2x supersampling, 4=4x). Default is 2.
            preserve_aspect_ratio: If False (default), match CairoSVG behavior (stretch if viewBox).
                                   If True, always preserve aspect ratio.
        """
        self.scale = scale
        self.background = background or (255, 255, 255, 255)
        self.antialias = max(1, antialias)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.parser = SVGParser()

    def render(self, svg_content: str, width: Optional[int] = None,
               height: Optional[int] = None) -> Image.Image:
        """
        Render SVG content to a PIL Image.

        Args:
            svg_content: SVG content as string
            width: Override width (optional)
            height: Override height (optional)

        Returns:
            PIL Image with rendered SVG
        """
        doc = self.parser.parse(svg_content)
        return self.render_document(doc, width, height)

    def render_file(self, filepath: str, width: Optional[int] = None,
                    height: Optional[int] = None) -> Image.Image:
        """Render SVG file to a PIL Image."""
        doc = self.parser.parse_file(filepath)
        return self.render_document(doc, width, height)

    def render_document(self, doc: SVGDocument, width: Optional[int] = None,
                        height: Optional[int] = None) -> Image.Image:
        """Render parsed SVG document to a PIL Image."""
        # Determine source dimensions (from viewBox or document size)
        if doc.viewBox:
            src_x, src_y, src_w, src_h = doc.viewBox
        else:
            src_x, src_y = 0, 0
            src_w, src_h = doc.width, doc.height

        # Determine output dimensions
        out_width = int((width or doc.width) * self.scale)
        out_height = int((height or doc.height) * self.scale)

        # Apply supersampling for anti-aliasing
        aa = self.antialias
        render_width = out_width * aa
        render_height = out_height * aa

        # Create image at higher resolution for anti-aliasing
        image = Image.new("RGBA", (render_width, render_height), self.background)

        # Calculate scaling transform (including AA factor)
        scale_x = render_width / src_w if src_w else 1
        scale_y = render_height / src_h if src_h else 1

        # Determine whether to stretch or preserve aspect ratio
        # Only stretch if preserveAspectRatio="none"
        should_stretch = (doc.preserve_aspect_ratio == "none")

        if should_stretch:
            # Non-uniform scaling (stretch to fill)
            offset_x = -src_x * scale_x
            offset_y = -src_y * scale_y
            transform = Transform.translate(offset_x, offset_y).multiply(
                Transform.scale(scale_x, scale_y)
            )
        else:
            # Uniform scaling (preserve aspect ratio)
            scale = min(scale_x, scale_y)
            # Center the content
            offset_x = (render_width - src_w * scale) / 2 - src_x * scale
            offset_y = (render_height - src_h * scale) / 2 - src_y * scale
            transform = Transform.translate(offset_x, offset_y).multiply(
                Transform.scale(scale)
            )

        # Create render context
        ctx = RenderContext(image, doc.gradients, transform, doc.clip_paths, doc.filters)

        # Render all elements
        for element in doc.elements:
            self._render_element(ctx, element)

        # Apply viewBox clipping - hide content outside viewBox bounds
        if doc.viewBox:
            vb_x, vb_y, vb_w, vb_h = doc.viewBox

            # The viewBox defines what region of the SVG coordinate space is visible.
            # Our transform maps viewBox content to screen. The viewBox origin (vb_x, vb_y)
            # maps to screen position: offset + vb_x * scale (for uniform) or offset + vb_x * scale_x (stretched)
            #
            # Since offset already accounts for centering: offset = (render - src_w * scale)/2 - src_x * scale
            # And src_x = vb_x, the viewBox origin maps to: (render - src_w * scale)/2

            if should_stretch:
                # In stretched mode, viewBox maps exactly to render dimensions
                clip_x1 = 0
                clip_y1 = 0
                clip_x2 = render_width
                clip_y2 = render_height
            else:
                # Uniform scaling - viewBox is centered with possible letterboxing
                # The viewBox content occupies: src_w * scale width, src_h * scale height
                # Centered at (render_width/2, render_height/2)
                content_w = src_w * scale
                content_h = src_h * scale
                # Use floor for lower bounds and ceil for upper bounds to avoid cutting off content
                clip_x1 = math.floor((render_width - content_w) / 2)
                clip_y1 = math.floor((render_height - content_h) / 2)
                clip_x2 = math.ceil((render_width + content_w) / 2)
                clip_y2 = math.ceil((render_height + content_h) / 2)

            # Clamp to image bounds
            clip_x1 = max(0, clip_x1)
            clip_y1 = max(0, clip_y1)
            clip_x2 = min(render_width, clip_x2)
            clip_y2 = min(render_height, clip_y2)

            # Only apply clipping if we have a valid rectangle
            if clip_x2 > clip_x1 and clip_y2 > clip_y1:
                # Create mask and apply clipping
                mask = Image.new("L", (render_width, render_height), 0)
                from PIL import ImageDraw
                draw = ImageDraw.Draw(mask)
                draw.rectangle([clip_x1, clip_y1, clip_x2, clip_y2], fill=255)

                # Apply mask to alpha channel
                r, g, b, a = image.split()
                a = ImageChops.multiply(a, mask)
                image = Image.merge("RGBA", (r, g, b, a))

        # Downscale for anti-aliasing effect
        if aa > 1:
            image = image.resize((out_width, out_height), Image.LANCZOS)

        return image

    def _render_element(self, ctx: "RenderContext", element: SVGElement, depth: int = 0):
        """Render a single element."""
        # Prevent infinite recursion
        if depth > self.MAX_RENDER_DEPTH:
            return

        # Skip elements with display: none
        if element.style.display == "none":
            return

        # Check if element has a clip path
        if element.clip_path_id and element.clip_path_id in ctx.clip_paths:
            self._render_element_with_clip(ctx, element, depth)
            return

        # Apply Gaussian blur filter if present
        if element.style.filter_id and element.style.filter_id in ctx.filters:
            self._render_element_with_filter(ctx, element, depth)
            return

        if isinstance(element, GroupElement):
            for child in element.children:
                self._render_element(ctx, child, depth + 1)
        elif isinstance(element, RectElement):
            self._render_rect(ctx, element)
        elif isinstance(element, CircleElement):
            self._render_circle(ctx, element)
        elif isinstance(element, EllipseElement):
            self._render_ellipse(ctx, element)
        elif isinstance(element, LineElement):
            self._render_line(ctx, element)
        elif isinstance(element, PolylineElement):
            self._render_polyline(ctx, element)
        elif isinstance(element, PolygonElement):
            self._render_polygon(ctx, element)
        elif isinstance(element, PathElement):
            self._render_path(ctx, element)
        elif isinstance(element, TextElement):
            self._render_text(ctx, element)

    def _render_element_with_clip(self, ctx: "RenderContext", element: SVGElement, depth: int = 0):
        """Render an element with a clip path applied."""
        clip_path = ctx.clip_paths[element.clip_path_id]

        # Create a temporary image for the element
        temp_image = Image.new("RGBA", ctx.image.size, (0, 0, 0, 0))
        temp_ctx = RenderContext(temp_image, ctx.gradients, ctx.base_transform, ctx.clip_paths, ctx.filters)

        # Render the element without clipping to the temp image
        element_copy = element
        element_copy.clip_path_id = None  # Temporarily remove clip path

        if isinstance(element, GroupElement):
            for child in element.children:
                self._render_element(temp_ctx, child, depth + 1)
        elif isinstance(element, RectElement):
            self._render_rect(temp_ctx, element)
        elif isinstance(element, CircleElement):
            self._render_circle(temp_ctx, element)
        elif isinstance(element, EllipseElement):
            self._render_ellipse(temp_ctx, element)
        elif isinstance(element, LineElement):
            self._render_line(temp_ctx, element)
        elif isinstance(element, PolylineElement):
            self._render_polyline(temp_ctx, element)
        elif isinstance(element, PolygonElement):
            self._render_polygon(temp_ctx, element)
        elif isinstance(element, PathElement):
            self._render_path(temp_ctx, element)
        elif isinstance(element, TextElement):
            self._render_text(temp_ctx, element)

        # Create clip mask from clip path shapes
        mask = self._create_clip_mask(ctx, clip_path, element.transform)

        # Apply the mask and composite onto main image
        temp_image.putalpha(ImageChops.multiply(temp_image.split()[3], mask))
        ctx.image.alpha_composite(temp_image)

    def _render_element_with_filter(self, ctx: "RenderContext", element: SVGElement, depth: int = 0):
        """Render an element with a filter (e.g., Gaussian blur) applied."""
        filter_def = ctx.filters[element.style.filter_id]

        # Calculate blur radius first to determine region padding
        scale = math.sqrt(abs(ctx.base_transform.a * ctx.base_transform.d -
                              ctx.base_transform.b * ctx.base_transform.c))
        blur_radius = filter_def.std_deviation * scale

        # Get element bounding box in screen coordinates
        elem_bbox = self._get_element_bbox(element, ctx.base_transform)

        # Determine render region - use element bbox + blur padding for efficiency
        # Blur spreads beyond element, so we need padding of ~3x blur radius
        blur_padding = int(blur_radius * 3) + 2

        if elem_bbox:
            ex, ey, ew, eh = elem_bbox
            # Region in screen coordinates with padding
            region_x = max(0, int(ex) - blur_padding)
            region_y = max(0, int(ey) - blur_padding)
            region_x2 = min(ctx.image.width, int(ex + ew) + blur_padding)
            region_y2 = min(ctx.image.height, int(ey + eh) + blur_padding)
            region_w = region_x2 - region_x
            region_h = region_y2 - region_y

            # Only use region optimization if region is significantly smaller than full image
            use_region = region_w * region_h < ctx.image.width * ctx.image.height * 0.5
        else:
            use_region = False

        if use_region and region_w > 0 and region_h > 0:
            # Create smaller temp image for just the region
            temp_image = Image.new("RGBA", (region_w, region_h), (0, 0, 0, 0))

            # Create offset transform that shifts rendering to region coordinates
            offset_transform = Transform(1, 0, 0, 1, -region_x, -region_y)
            adjusted_base = offset_transform.multiply(ctx.base_transform)
            temp_ctx = RenderContext(temp_image, ctx.gradients, adjusted_base, ctx.clip_paths, ctx.filters)
        else:
            # Fall back to full image rendering
            region_x, region_y = 0, 0
            temp_image = Image.new("RGBA", ctx.image.size, (0, 0, 0, 0))
            temp_ctx = RenderContext(temp_image, ctx.gradients, ctx.base_transform, ctx.clip_paths, ctx.filters)

        # Render the element without filter to the temp image
        old_filter_id = element.style.filter_id
        element.style.filter_id = None  # Temporarily remove filter

        if isinstance(element, GroupElement):
            for child in element.children:
                self._render_element(temp_ctx, child, depth + 1)
        elif isinstance(element, RectElement):
            self._render_rect(temp_ctx, element)
        elif isinstance(element, CircleElement):
            self._render_circle(temp_ctx, element)
        elif isinstance(element, EllipseElement):
            self._render_ellipse(temp_ctx, element)
        elif isinstance(element, LineElement):
            self._render_line(temp_ctx, element)
        elif isinstance(element, PolylineElement):
            self._render_polyline(temp_ctx, element)
        elif isinstance(element, PolygonElement):
            self._render_polygon(temp_ctx, element)
        elif isinstance(element, PathElement):
            self._render_path(temp_ctx, element)
        elif isinstance(element, TextElement):
            self._render_text(temp_ctx, element)

        # Restore filter_id
        element.style.filter_id = old_filter_id

        # Apply Gaussian blur filter
        # PIL GaussianBlur radius - at least 1 pixel
        if blur_radius >= 0.5:
            # Use premultiplied alpha to avoid blending with black from transparent pixels
            # Memory-optimized: work with uint8 arrays, avoid float32 where possible
            arr = np.array(temp_image, dtype=np.uint8)
            alpha = arr[:, :, 3].astype(np.float32)

            # Create premultiplied channels and blur them
            blur = ImageFilter.GaussianBlur(radius=blur_radius)
            blurred_channels = []

            # Premultiply and blur RGB channels
            for c in range(3):
                channel = arr[:, :, c].astype(np.float32)
                premult = (channel * alpha / 255.0).astype(np.uint8)
                blurred = np.array(Image.fromarray(premult, mode='L').filter(blur), dtype=np.float32)
                blurred_channels.append(blurred)
                del channel, premult

            # Blur alpha channel
            alpha_blurred = np.array(Image.fromarray(arr[:, :, 3], mode='L').filter(blur), dtype=np.float32)
            del arr, alpha

            # Un-premultiply: R = R' * 255 / A (avoid division by zero)
            alpha_safe = np.maximum(alpha_blurred, 1.0)
            result = np.empty((temp_image.height, temp_image.width, 4), dtype=np.uint8)
            for c in range(3):
                result[:, :, c] = np.clip(blurred_channels[c] * 255.0 / alpha_safe, 0, 255).astype(np.uint8)
            result[:, :, 3] = alpha_blurred.astype(np.uint8)
            del blurred_channels, alpha_blurred, alpha_safe

            temp_image = Image.fromarray(result, mode='RGBA')
            del result

        # Apply filter region clipping
        if elem_bbox:
            ex, ey, ew, eh = elem_bbox

            if filter_def.filter_units == "objectBoundingBox":
                # Filter region is relative to element bbox
                fx = ex + filter_def.x * ew
                fy = ey + filter_def.y * eh
                fw = filter_def.width * ew
                fh = filter_def.height * eh
            else:
                # userSpaceOnUse - filter region in user coordinates, transform to pixels
                fx, fy = ctx.base_transform.apply(filter_def.x, filter_def.y)
                fx2, fy2 = ctx.base_transform.apply(filter_def.x + filter_def.width,
                                                     filter_def.y + filter_def.height)
                fw, fh = fx2 - fx, fy2 - fy

            # Create clip mask for filter region (in temp_image coordinates)
            filter_mask = Image.new("L", temp_image.size, 0)
            filter_draw = ImageDraw.Draw(filter_mask)
            # Adjust coordinates for region offset
            filter_draw.rectangle([int(fx - region_x), int(fy - region_y),
                                   int(fx + fw - region_x), int(fy + fh - region_y)], fill=255)

            # Apply mask to blurred image
            temp_image.putalpha(ImageChops.multiply(temp_image.split()[3], filter_mask))

        # Composite blurred image onto main image at correct position
        ctx.image.alpha_composite(temp_image, (region_x, region_y))

    def _get_element_bbox(self, element: SVGElement, transform: Transform) -> Optional[tuple]:
        """Get element bounding box in screen coordinates."""
        if isinstance(element, RectElement):
            corners = [
                (element.x, element.y),
                (element.x + element.width, element.y),
                (element.x + element.width, element.y + element.height),
                (element.x, element.y + element.height)
            ]
        elif isinstance(element, CircleElement):
            corners = [
                (element.cx - element.r, element.cy - element.r),
                (element.cx + element.r, element.cy + element.r)
            ]
        elif isinstance(element, EllipseElement):
            corners = [
                (element.cx - element.rx, element.cy - element.ry),
                (element.cx + element.rx, element.cy + element.ry)
            ]
        else:
            # For other elements, return None (no clipping)
            return None

        # Transform corners
        combined = transform.multiply(element.transform)
        transformed = [combined.apply(x, y) for x, y in corners]
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]

        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def _create_clip_mask(self, ctx: "RenderContext", clip_path: ClipPath,
                          element_transform: Transform) -> Image.Image:
        """Create a mask image from a clip path."""
        mask = Image.new("L", ctx.image.size, 0)

        for clip_elem in clip_path.elements:
            full_transform = ctx.base_transform.multiply(element_transform).multiply(clip_elem.transform)

            # PathElement may have multiple subpaths - handle each separately
            if isinstance(clip_elem, PathElement):
                polygons = self._path_to_polygons(clip_elem.commands)
                for poly in polygons:
                    if len(poly) >= 3:
                        transformed = [full_transform.apply(x, y) for x, y in poly]
                        self._fill_polygon_nonzero(mask, transformed)
            else:
                # Get points from the clip element
                points = self._get_element_points(clip_elem)
                if not points:
                    continue

                # Transform points
                transformed = [full_transform.apply(x, y) for x, y in points]

                # Fill polygon with nonzero winding rule (SVG default for clip-path)
                if len(transformed) >= 3:
                    self._fill_polygon_nonzero(mask, transformed)

        # Handle nested clip path (for intersection)
        if clip_path.clip_path_id and clip_path.clip_path_id in ctx.clip_paths:
            nested_clip = ctx.clip_paths[clip_path.clip_path_id]
            nested_mask = self._create_clip_mask(ctx, nested_clip, element_transform)
            # Intersection: keep only where both masks are white
            mask = ImageChops.multiply(mask, nested_mask)

        return mask

    def _fill_polygon_nonzero(self, mask: Image.Image, points: List[Tuple[float, float]]):
        """Fill a polygon using nonzero winding rule (handles self-intersecting polygons)."""
        if len(points) < 3:
            return

        width, height = mask.size
        mask_arr = np.array(mask)

        # Get bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_y = max(0, int(min(ys)))
        max_y = min(height - 1, int(max(ys)) + 1)
        min_x = max(0, int(min(xs)))
        max_x = min(width - 1, int(max(xs)) + 1)

        # Build edge list
        edges = []
        n = len(points)
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            if p1[1] != p2[1]:  # Skip horizontal edges
                if p1[1] > p2[1]:
                    p1, p2 = p2, p1  # Ensure p1.y < p2.y
                    direction = -1
                else:
                    direction = 1
                edges.append((p1[0], p1[1], p2[0], p2[1], direction))

        # Scanline fill with winding count
        for y in range(min_y, max_y + 1):
            # Find intersections with edges
            intersections = []
            for x1, y1, x2, y2, direction in edges:
                if y1 <= y < y2:
                    # Compute x intersection
                    t = (y - y1) / (y2 - y1)
                    x_intersect = x1 + t * (x2 - x1)
                    intersections.append((x_intersect, direction))

            # Sort by x
            intersections.sort(key=lambda p: p[0])

            # Fill using winding count
            winding = 0
            prev_x = None
            for x_int, direction in intersections:
                if winding != 0 and prev_x is not None:
                    # Fill from prev_x to x_int
                    x_start = max(min_x, int(prev_x))
                    x_end = min(max_x, int(x_int))
                    if x_start <= x_end:
                        mask_arr[y, x_start:x_end + 1] = 255
                winding += direction
                prev_x = x_int

        # Update mask
        mask.paste(Image.fromarray(mask_arr))

    def _get_element_points(self, element: SVGElement) -> List[Tuple[float, float]]:
        """Get the outline points of an element for clipping."""
        if isinstance(element, RectElement):
            x, y, w, h = element.x, element.y, element.width, element.height
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        elif isinstance(element, CircleElement):
            n_points = max(32, int(element.r * 2))
            points = []
            for i in range(n_points):
                angle = 2 * math.pi * i / n_points
                px = element.cx + element.r * math.cos(angle)
                py = element.cy + element.r * math.sin(angle)
                points.append((px, py))
            return points

        elif isinstance(element, EllipseElement):
            n_points = max(32, int(max(element.rx, element.ry) * 2))
            points = []
            for i in range(n_points):
                angle = 2 * math.pi * i / n_points
                px = element.cx + element.rx * math.cos(angle)
                py = element.cy + element.ry * math.sin(angle)
                points.append((px, py))
            return points

        elif isinstance(element, PolygonElement):
            return list(element.points)

        elif isinstance(element, PathElement):
            # Convert path to polygons and return all points
            polygons = self._path_to_polygons(element.commands)
            all_points = []
            for poly in polygons:
                all_points.extend(poly)
            return all_points

        return []

    def _get_fill_color(self, ctx: "RenderContext", style: Style,
                        bbox: Optional[tuple[float, float, float, float]] = None
                        ) -> Optional[tuple[int, int, int, int]]:
        """Get fill color, resolving gradients if needed."""
        fill = style.fill
        if fill is None or fill is FILL_NOT_SET:
            return None

        if isinstance(fill, str):
            if fill.startswith("url("):
                # This is a gradient reference
                return None  # Will be handled by gradient fill
            return None

        # Apply opacity
        r, g, b, a = fill
        a = int(a * style.fill_opacity * style.opacity)
        return (r, g, b, a)

    def _get_stroke_color(self, ctx: "RenderContext", style: Style
                          ) -> Optional[tuple[int, int, int, int]]:
        """Get stroke color."""
        stroke = style.stroke
        if stroke is None:
            return None

        if isinstance(stroke, str):
            return None

        r, g, b, a = stroke
        a = int(a * style.stroke_opacity * style.opacity)
        return (r, g, b, a)

    def _transform_points(self, ctx: "RenderContext", transform: Transform,
                          points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Apply transforms to a list of points."""
        full_transform = ctx.base_transform.multiply(transform)
        return [full_transform.apply(x, y) for x, y in points]

    def _render_rect(self, ctx: "RenderContext", rect: RectElement):
        """Render a rectangle."""
        transform = ctx.base_transform.multiply(rect.transform)

        # Get corners
        x, y = rect.x, rect.y
        w, h = rect.width, rect.height

        if w <= 0 or h <= 0:
            return

        # Handle rounded corners
        rx, ry = rect.rx, rect.ry
        if rx == 0 and ry == 0:
            # Simple rectangle
            corners = [
                (x, y), (x + w, y), (x + w, y + h), (x, y + h)
            ]
            corners = self._transform_points(ctx, rect.transform, corners)
            self._fill_and_stroke_polygon(ctx, corners, rect.style, rect.transform,
                                          (rect.x, rect.y, rect.width, rect.height))
        else:
            # Rounded rectangle - approximate with path
            if rx == 0:
                rx = ry
            if ry == 0:
                ry = rx
            rx = min(rx, w / 2)
            ry = min(ry, h / 2)

            # Create path for rounded rect
            points = self._rounded_rect_path(x, y, w, h, rx, ry)
            points = self._transform_points(ctx, rect.transform, points)
            self._fill_and_stroke_polygon(ctx, points, rect.style, rect.transform,
                                          (x, y, w, h))

    def _rounded_rect_path(self, x: float, y: float, w: float, h: float,
                           rx: float, ry: float) -> list[tuple[float, float]]:
        """Generate points for a rounded rectangle."""
        points = []
        n_curve = 8  # Points per corner

        # Top edge (left to right)
        points.append((x + rx, y))
        points.append((x + w - rx, y))

        # Top-right corner
        for i in range(n_curve + 1):
            angle = -math.pi / 2 + (math.pi / 2) * i / n_curve
            px = x + w - rx + rx * math.cos(angle)
            py = y + ry + ry * math.sin(angle)
            points.append((px, py))

        # Right edge
        points.append((x + w, y + h - ry))

        # Bottom-right corner
        for i in range(n_curve + 1):
            angle = 0 + (math.pi / 2) * i / n_curve
            px = x + w - rx + rx * math.cos(angle)
            py = y + h - ry + ry * math.sin(angle)
            points.append((px, py))

        # Bottom edge
        points.append((x + rx, y + h))

        # Bottom-left corner
        for i in range(n_curve + 1):
            angle = math.pi / 2 + (math.pi / 2) * i / n_curve
            px = x + rx + rx * math.cos(angle)
            py = y + h - ry + ry * math.sin(angle)
            points.append((px, py))

        # Left edge
        points.append((x, y + ry))

        # Top-left corner
        for i in range(n_curve + 1):
            angle = math.pi + (math.pi / 2) * i / n_curve
            px = x + rx + rx * math.cos(angle)
            py = y + ry + ry * math.sin(angle)
            points.append((px, py))

        return points

    def _render_circle(self, ctx: "RenderContext", circle: CircleElement):
        """Render a circle."""
        if circle.r <= 0:
            return

        # Generate circle points - use more points for smoother strokes
        scale = self._get_scale(ctx, circle.transform)
        scaled_r = circle.r * scale
        stroke_w = circle.style.stroke_width * scale if circle.style.stroke else 0
        # More points for larger circles and thicker strokes
        n_points = max(72, int(scaled_r * 2), int(stroke_w * 8))
        n_points = min(n_points, 360)

        points = []
        for i in range(n_points):
            angle = 2 * math.pi * i / n_points
            px = circle.cx + circle.r * math.cos(angle)
            py = circle.cy + circle.r * math.sin(angle)
            points.append((px, py))

        points = self._transform_points(ctx, circle.transform, points)
        bbox = (circle.cx - circle.r, circle.cy - circle.r,
                circle.r * 2, circle.r * 2)
        self._fill_and_stroke_polygon(ctx, points, circle.style, circle.transform, bbox)

    def _render_ellipse(self, ctx: "RenderContext", ellipse: EllipseElement):
        """Render an ellipse."""
        if ellipse.rx <= 0 or ellipse.ry <= 0:
            return

        # Generate ellipse points - use more points for smoother strokes
        scale = self._get_scale(ctx, ellipse.transform)
        scaled_r = max(ellipse.rx, ellipse.ry) * scale
        stroke_w = ellipse.style.stroke_width * scale if ellipse.style.stroke else 0
        # More points for larger ellipses and thicker strokes
        n_points = max(72, int(scaled_r * 2), int(stroke_w * 8))
        n_points = min(n_points, 360)

        points = []
        for i in range(n_points):
            angle = 2 * math.pi * i / n_points
            px = ellipse.cx + ellipse.rx * math.cos(angle)
            py = ellipse.cy + ellipse.ry * math.sin(angle)
            points.append((px, py))

        points = self._transform_points(ctx, ellipse.transform, points)
        bbox = (ellipse.cx - ellipse.rx, ellipse.cy - ellipse.ry,
                ellipse.rx * 2, ellipse.ry * 2)
        self._fill_and_stroke_polygon(ctx, points, ellipse.style, ellipse.transform, bbox)

    def _render_line(self, ctx: "RenderContext", line: LineElement):
        """Render a line."""
        points = [(line.x1, line.y1), (line.x2, line.y2)]
        points = self._transform_points(ctx, line.transform, points)

        # Use proper stroke rendering with linecap
        self._stroke_path(ctx, points, line.style, line.transform, closed=False)

    def _render_polyline(self, ctx: "RenderContext", polyline: PolylineElement):
        """Render a polyline."""
        if len(polyline.points) < 2:
            return

        points = self._transform_points(ctx, polyline.transform, polyline.points)
        bbox = self._compute_bbox(polyline.points)

        # Polylines can be filled (fill closes the path from last to first point)
        fill = self._get_fill_color(ctx, polyline.style)
        fill_ref = polyline.style.fill if isinstance(polyline.style.fill, str) else None

        if fill and len(points) >= 3:
            self._fill_polygon_with_gradient_check(
                ctx, points, polyline.style, polyline.transform, bbox, fill, fill_ref
            )
        elif fill_ref and fill_ref.startswith("url(") and len(points) >= 3:
            self._fill_polygon_with_gradient_check(
                ctx, points, polyline.style, polyline.transform, bbox, None, fill_ref
            )

        # Stroke is rendered as open path (not closed)
        self._stroke_path(ctx, points, polyline.style, polyline.transform, closed=False)

    def _render_polygon(self, ctx: "RenderContext", polygon: PolygonElement):
        """Render a polygon."""
        if len(polygon.points) < 3:
            return

        points = self._transform_points(ctx, polygon.transform, polygon.points)
        bbox = self._compute_bbox(polygon.points)
        self._fill_and_stroke_polygon(ctx, points, polygon.style, polygon.transform, bbox)

    def _render_path(self, ctx: "RenderContext", path: PathElement):
        """Render a path."""
        if not path.commands:
            return

        # Convert path to polygons
        polygons = self._path_to_polygons(path.commands)

        if not polygons:
            return

        # Compute bounding box from commands
        bbox = self._compute_path_bbox(path.commands)

        # Transform all polygons
        transformed_polygons = []
        for polygon_points in polygons:
            if len(polygon_points) >= 2:
                points = self._transform_points(ctx, path.transform, polygon_points)
                transformed_polygons.append((polygon_points, points))

        if not transformed_polygons:
            return

        # Fill - for evenodd rule with multiple subpaths, combine them
        fill = self._get_fill_color(ctx, path.style)
        fill_ref = path.style.fill if isinstance(path.style.fill, str) else None

        if (fill or (fill_ref and fill_ref.startswith("url("))) and len(transformed_polygons) > 0:
            # Collect all transformed points for combined fill
            all_points = []
            for _, points in transformed_polygons:
                if len(points) >= 3:
                    all_points.append(points)

            if all_points:
                if len(all_points) > 1:
                    # Multiple subpaths - combine and apply fill rule
                    if path.style.fill_rule == "evenodd":
                        self._fill_multi_polygon_evenodd(ctx, all_points, fill, fill_ref,
                                                         path.style, path.transform, bbox)
                    else:
                        # Nonzero rule - combine subpaths (creates holes with opposite winding)
                        self._fill_multi_polygon_nonzero(ctx, all_points, fill, fill_ref,
                                                          path.style, path.transform, bbox)
                else:
                    # Single polygon
                    for points in all_points:
                        if fill:
                            self._fill_polygon_with_gradient_check(
                                ctx, points, path.style, path.transform, bbox, fill, fill_ref
                            )
                        elif fill_ref:
                            self._fill_polygon_with_gradient_check(
                                ctx, points, path.style, path.transform, bbox, None, fill_ref
                            )

        # Stroke each polygon separately
        for polygon_points, points in transformed_polygons:
            is_closed = len(polygon_points) > 2 and self._point_distance(
                polygon_points[0], polygon_points[-1]
            ) < 0.01
            self._stroke_path(ctx, points, path.style, path.transform, closed=is_closed)

    def _path_to_polygons(self, commands: list[tuple]) -> list[list[tuple[float, float]]]:
        """Convert path commands to a list of polygons."""
        polygons = []
        current_polygon = []
        current_x, current_y = 0.0, 0.0
        subpath_start_x, subpath_start_y = 0.0, 0.0

        for cmd in commands:
            cmd_type = cmd[0]

            if cmd_type == 'M':
                if current_polygon:
                    polygons.append(current_polygon)
                current_polygon = [(cmd[1], cmd[2])]
                current_x, current_y = cmd[1], cmd[2]
                subpath_start_x, subpath_start_y = current_x, current_y

            elif cmd_type == 'L':
                # If polygon is empty (after Z), start from subpath start
                if not current_polygon:
                    current_polygon = [(subpath_start_x, subpath_start_y)]
                current_polygon.append((cmd[1], cmd[2]))
                current_x, current_y = cmd[1], cmd[2]

            elif cmd_type == 'C':
                # Cubic bezier - sample it
                if not current_polygon:
                    current_polygon = [(subpath_start_x, subpath_start_y)]
                x1, y1, x2, y2, x, y = cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], cmd[6]
                bezier_points = self._sample_cubic_bezier(
                    current_x, current_y, x1, y1, x2, y2, x, y
                )
                current_polygon.extend(bezier_points)
                current_x, current_y = x, y

            elif cmd_type == 'Q':
                # Quadratic bezier - sample it
                if not current_polygon:
                    current_polygon = [(subpath_start_x, subpath_start_y)]
                x1, y1, x, y = cmd[1], cmd[2], cmd[3], cmd[4]
                bezier_points = self._sample_quadratic_bezier(
                    current_x, current_y, x1, y1, x, y
                )
                current_polygon.extend(bezier_points)
                current_x, current_y = x, y

            elif cmd_type == 'Z':
                if current_polygon:
                    # Close the path
                    if current_polygon[0] != current_polygon[-1]:
                        current_polygon.append(current_polygon[0])
                    polygons.append(current_polygon)
                    current_x, current_y = subpath_start_x, subpath_start_y
                    current_polygon = []

        if current_polygon:
            polygons.append(current_polygon)

        return polygons

    def _sample_cubic_bezier(self, x0: float, y0: float,
                             x1: float, y1: float,
                             x2: float, y2: float,
                             x3: float, y3: float,
                             n_samples: int = 16) -> list[tuple[float, float]]:
        """Sample points along a cubic bezier curve."""
        # Use Rust implementation if available
        if HAS_RUST:
            return vectorstag_rust.sample_cubic_bezier(x0, y0, x1, y1, x2, y2, x3, y3, n_samples)
        # Fallback to Python implementation
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

    def _sample_quadratic_bezier(self, x0: float, y0: float,
                                 x1: float, y1: float,
                                 x2: float, y2: float,
                                 n_samples: int = 12) -> list[tuple[float, float]]:
        """Sample points along a quadratic bezier curve."""
        # Use Rust implementation if available
        if HAS_RUST:
            return vectorstag_rust.sample_quadratic_bezier(x0, y0, x1, y1, x2, y2, n_samples)
        # Fallback to Python implementation
        points = []
        for i in range(1, n_samples + 1):
            t = i / n_samples
            mt = 1 - t

            x = mt * mt * x0 + 2 * mt * t * x1 + t * t * x2
            y = mt * mt * y0 + 2 * mt * t * y1 + t * t * y2
            points.append((x, y))

        return points

    def _stroke_path(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                     style: Style, element_transform: Transform, closed: bool = False,
                     bbox: tuple = None):
        """Render a stroke with proper linecap and linejoin."""
        # Check for gradient stroke
        if isinstance(style.stroke, str) and style.stroke.startswith("url("):
            self._stroke_path_with_gradient(ctx, points, style, element_transform, closed, bbox)
            return

        stroke = self._get_stroke_color(ctx, style)
        if not stroke or len(points) < 2:
            return

        width = style.stroke_width * self._get_scale(ctx, element_transform)
        if width < 0.5:
            return

        # Handle stroke-dasharray
        if style.stroke_dasharray and len(style.stroke_dasharray) > 0:
            self._stroke_dashed_path(ctx, points, style, element_transform, width, stroke, closed)
            return

        half_width = width / 2.0

        # Use polygon-based rendering for proper stroke geometry:
        # - Always for closed paths (to handle miter/bevel joins correctly at corners)
        # - For open paths with many points (smooth curves) to avoid gaps
        if closed or len(points) >= 8:
            if closed:
                self._stroke_closed_polygon(ctx, points, stroke, half_width,
                                            style.stroke_miterlimit, style.stroke_linejoin)
            else:
                self._stroke_open_polygon(ctx, points, stroke, half_width, style.stroke_linecap)
            return

        int_width = max(1, int(width))

        # Use temp image for semi-transparent strokes
        if stroke[3] < 255:
            temp = Image.new("RGBA", ctx.image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(temp, "RGBA")
        else:
            temp = None
            draw = ImageDraw.Draw(ctx.image, "RGBA")

        # Draw line segments
        if closed:
            stroke_points = list(points)
            if stroke_points[0] != stroke_points[-1]:
                stroke_points.append(stroke_points[0])
            for i in range(len(stroke_points) - 1):
                draw.line([stroke_points[i], stroke_points[i + 1]], fill=stroke, width=int_width)
        else:
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=stroke, width=int_width)

        # Draw round caps for line endpoints if needed
        if style.stroke_linecap == "round" and not closed:
            radius = int_width / 2.0
            if radius > 0:
                x, y = points[0]
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=stroke)
                x, y = points[-1]
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=stroke)

        # Draw round joints at corner vertices only (not smooth curve points)
        if style.stroke_linejoin == "round" and len(points) < 8:
            radius = int_width / 2.0
            if radius > 0:
                # For closed paths, draw at all original vertices
                # For open paths, draw at interior vertices only
                joint_points = points[:-1] if (closed and len(points) > 1 and
                    self._point_distance(points[0], points[-1]) < 0.01) else points
                start_idx = 0 if closed else 1
                end_idx = len(joint_points) if closed else len(joint_points) - 1
                for i in range(start_idx, end_idx):
                    x, y = joint_points[i]
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=stroke)

        if temp is not None:
            ctx.image.alpha_composite(temp)

    def _stroke_path_with_gradient(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                                    style: Style, element_transform: Transform,
                                    closed: bool, bbox: tuple):
        """Render a stroke with gradient fill."""
        if len(points) < 2:
            return

        width = style.stroke_width * self._get_scale(ctx, element_transform)
        if width < 0.5:
            return

        half_width = width / 2.0

        # Build stroke polygon
        stroke_polygon = self._build_stroke_polygon(points, half_width, style.stroke_linecap,
                                                    style.stroke_linejoin, closed, style.stroke_miterlimit)
        if not stroke_polygon or len(stroke_polygon) < 3:
            return

        # Get bounding box for gradient
        if bbox is None:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            # Convert back to user space for bbox
            inv_transform = ctx.base_transform.multiply(element_transform)
            # Use screen-space bbox
            sxs = [p[0] for p in stroke_polygon]
            sys = [p[1] for p in stroke_polygon]
            screen_bbox = (min(sxs), min(sys), max(sxs) - min(sxs), max(sys) - min(sys))
        else:
            screen_bbox = bbox

        # Create a temporary style with fill set to the stroke gradient
        stroke_ref = style.stroke
        opacity = style.stroke_opacity * style.opacity

        # Fill the stroke polygon with gradient
        self._fill_polygon_with_gradient_check(ctx, stroke_polygon, style, element_transform,
                                               screen_bbox, None, stroke_ref)

    def _stroke_dashed_path(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                            style: Style, element_transform: Transform, width: float,
                            stroke: Tuple[int, int, int, int], closed: bool):
        """Render a dashed stroke along a path."""
        if len(points) < 2:
            return

        dasharray = style.stroke_dasharray
        if not dasharray:
            return

        # Normalize dasharray (must have even number of elements)
        if len(dasharray) % 2 == 1:
            dasharray = dasharray * 2  # Repeat to make even

        scale = self._get_scale(ctx, element_transform)
        scaled_dashes = [d * scale for d in dasharray]

        # Walk along the path and collect dash segments
        total_dash_length = sum(scaled_dashes)
        if total_dash_length <= 0:
            return

        # Compute cumulative distances along path
        distances = [0.0]
        for i in range(1, len(points)):
            d = self._point_distance(points[i-1], points[i])
            distances.append(distances[-1] + d)

        total_length = distances[-1]
        if total_length <= 0:
            return

        # Generate dash segments
        dash_idx = 0
        dash_offset = 0.0
        current_pos = 0.0
        is_on = True  # Start with a dash (not gap)

        half_width = width / 2.0

        while current_pos < total_length:
            dash_len = scaled_dashes[dash_idx % len(scaled_dashes)]
            segment_end = min(current_pos + dash_len, total_length)

            if is_on and dash_len > 0:
                # Extract points for this dash segment
                segment_points = self._extract_path_segment(points, distances, current_pos, segment_end)
                if len(segment_points) >= 2:
                    # Render this dash segment
                    self._stroke_open_polygon(ctx, segment_points, stroke, half_width, style.stroke_linecap)

            current_pos = segment_end
            dash_idx += 1
            is_on = not is_on

    def _extract_path_segment(self, points: List[Tuple[float, float]],
                              distances: List[float], start_dist: float,
                              end_dist: float) -> List[Tuple[float, float]]:
        """Extract a segment of the path between two distance values."""
        result = []

        for i in range(len(points) - 1):
            d0, d1 = distances[i], distances[i + 1]
            p0, p1 = points[i], points[i + 1]

            # Check if this segment overlaps with [start_dist, end_dist]
            if d1 <= start_dist or d0 >= end_dist:
                continue

            # Compute start point of overlap
            if d0 < start_dist:
                t = (start_dist - d0) / (d1 - d0) if d1 > d0 else 0
                start_pt = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
            else:
                start_pt = p0

            # Compute end point of overlap
            if d1 > end_dist:
                t = (end_dist - d0) / (d1 - d0) if d1 > d0 else 1
                end_pt = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
            else:
                end_pt = p1

            if not result:
                result.append(start_pt)
            result.append(end_pt)

        return result

    def _stroke_open_polygon(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                             stroke: Tuple[int, int, int, int], half_width: float,
                             linecap: str):
        """Render an open path stroke using polygon fill (gap-free)."""
        n = len(points)
        if n < 2:
            return

        # Remove duplicate consecutive points
        clean_points = [points[0]]
        for p in points[1:]:
            if self._point_distance(clean_points[-1], p) > 0.01:
                clean_points.append(p)

        points = clean_points
        n = len(points)
        if n < 2:
            return

        # Compute left and right edge points
        left_points = []
        right_points = []

        for i in range(n):
            p_curr = points[i]

            if i == 0:
                # First point - use direction to next point
                d = self._normalize(self._subtract(points[1], points[0]))
            elif i == n - 1:
                # Last point - use direction from previous point
                d = self._normalize(self._subtract(points[n-1], points[n-2]))
            else:
                # Middle point - average of incoming and outgoing directions
                d1 = self._normalize(self._subtract(p_curr, points[i-1]))
                d2 = self._normalize(self._subtract(points[i+1], p_curr))
                d = self._normalize((d1[0] + d2[0], d1[1] + d2[1]))

            # Perpendicular
            perp = (-d[1], d[0])

            left_pt = (p_curr[0] + perp[0] * half_width, p_curr[1] + perp[1] * half_width)
            right_pt = (p_curr[0] - perp[0] * half_width, p_curr[1] - perp[1] * half_width)

            left_points.append(left_pt)
            right_points.append(right_pt)

        # Build stroke polygon: left edge forward, end cap, right edge backward, start cap
        stroke_polygon = list(left_points)

        # End cap
        if linecap == "round":
            # Add semicircle at end (from left edge around to right edge)
            end_pt = points[-1]
            d = self._normalize(self._subtract(points[-1], points[-2]))
            perp = (-d[1], d[0])
            n_cap = 12
            for j in range(1, n_cap):
                angle = math.pi * j / n_cap
                # Start from left side (perp direction), go around to right side (-perp direction)
                cap_x = end_pt[0] + half_width * (perp[0] * math.cos(angle) + d[0] * math.sin(angle))
                cap_y = end_pt[1] + half_width * (perp[1] * math.cos(angle) + d[1] * math.sin(angle))
                stroke_polygon.append((cap_x, cap_y))
        elif linecap == "square":
            # Extend by half_width
            d = self._normalize(self._subtract(points[-1], points[-2]))
            stroke_polygon.append((left_points[-1][0] + d[0] * half_width,
                                   left_points[-1][1] + d[1] * half_width))
            stroke_polygon.append((right_points[-1][0] + d[0] * half_width,
                                   right_points[-1][1] + d[1] * half_width))

        # Right edge backward
        stroke_polygon.extend(reversed(right_points))

        # Start cap
        if linecap == "round":
            # Add semicircle at start (from right edge around to left edge)
            start_pt = points[0]
            d = self._normalize(self._subtract(points[1], points[0]))
            perp = (-d[1], d[0])
            n_cap = 12
            for j in range(1, n_cap):
                angle = math.pi * j / n_cap
                # Start from right side (-perp direction), go around to left side (perp direction)
                cap_x = start_pt[0] + half_width * (-perp[0] * math.cos(angle) - d[0] * math.sin(angle))
                cap_y = start_pt[1] + half_width * (-perp[1] * math.cos(angle) - d[1] * math.sin(angle))
                stroke_polygon.append((cap_x, cap_y))
        elif linecap == "square":
            d = self._normalize(self._subtract(points[1], points[0]))
            stroke_polygon.append((right_points[0][0] - d[0] * half_width,
                                   right_points[0][1] - d[1] * half_width))
            stroke_polygon.append((left_points[0][0] - d[0] * half_width,
                                   left_points[0][1] - d[1] * half_width))

        # Draw the stroke polygon (memory-optimized for semi-transparent strokes)
        if stroke[3] < 255:
            xs = [p[0] for p in stroke_polygon]
            ys = [p[1] for p in stroke_polygon]
            min_x, max_x = max(0, int(min(xs))), min(ctx.image.width, int(max(xs)) + 1)
            min_y, max_y = max(0, int(min(ys))), min(ctx.image.height, int(max(ys)) + 1)
            if min_x < max_x and min_y < max_y:
                temp = Image.new("RGBA", (max_x - min_x, max_y - min_y), (0, 0, 0, 0))
                draw = ImageDraw.Draw(temp, "RGBA")
                local_poly = [(x - min_x, y - min_y) for x, y in stroke_polygon]
                draw.polygon(local_poly, fill=stroke)
                ctx.image.alpha_composite(temp, (min_x, min_y))
        else:
            draw = ImageDraw.Draw(ctx.image, "RGBA")
            draw.polygon(stroke_polygon, fill=stroke)

    def _stroke_closed_polygon(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                               stroke: Tuple[int, int, int, int], half_width: float,
                               miterlimit: float = 4.0, linejoin: str = "miter"):
        """Render a closed polygon stroke.

        Uses outer/inner polygon approach for convex shapes, and segment-based
        approach for non-convex shapes (like NP flag) where stroke can self-intersect.
        """
        n = len(points)
        if n < 3:
            return

        # Remove duplicate consecutive points
        clean_points = [points[0]]
        for p in points[1:]:
            if self._point_distance(clean_points[-1], p) > 0.01:
                clean_points.append(p)
        if len(clean_points) > 1 and self._point_distance(clean_points[-1], clean_points[0]) < 0.01:
            clean_points = clean_points[:-1]

        points = clean_points
        n = len(points)
        if n < 3:
            return

        # Check if shape has reflex angles (non-convex) - stroke may self-intersect
        has_reflex = False
        sign = None
        for i in range(n):
            p_prev = points[(i - 1) % n]
            p_curr = points[i]
            p_next = points[(i + 1) % n]
            d1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            d2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign and abs(cross) > 0.01:
                has_reflex = True
                break

        if has_reflex:
            self._stroke_closed_polygon_segmented(ctx, points, stroke, half_width, miterlimit, linejoin)
        else:
            self._stroke_closed_polygon_outline(ctx, points, stroke, half_width, miterlimit, linejoin)

    def _stroke_closed_polygon_outline(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                                        stroke: Tuple[int, int, int, int], half_width: float,
                                        miterlimit: float = 4.0, linejoin: str = "miter"):
        """Render closed polygon stroke using outer/inner outline approach."""
        n = len(points)

        # Compute left and right edge points with miter joins
        left_points = []
        right_points = []

        for i in range(n):
            p_prev = points[(i - 1) % n]
            p_curr = points[i]
            p_next = points[(i + 1) % n]

            d1 = self._normalize(self._subtract(p_curr, p_prev))
            d2 = self._normalize(self._subtract(p_next, p_curr))

            perp1 = (-d1[1], d1[0])
            perp2 = (-d2[1], d2[0])

            cross = d1[0] * d2[1] - d1[1] * d2[0]

            if abs(cross) > 0.001:
                # Compute miter intersection
                left_p1 = (p_curr[0] + perp1[0] * half_width, p_curr[1] + perp1[1] * half_width)
                left_p2 = (p_curr[0] + perp2[0] * half_width, p_curr[1] + perp2[1] * half_width)
                right_p1 = (p_curr[0] - perp1[0] * half_width, p_curr[1] - perp1[1] * half_width)
                right_p2 = (p_curr[0] - perp2[0] * half_width, p_curr[1] - perp2[1] * half_width)

                left_pt = self._line_intersection(left_p1, d1, left_p2, d2)
                right_pt = self._line_intersection(right_p1, d1, right_p2, d2)

                if left_pt is None:
                    left_pt = left_p1
                if right_pt is None:
                    right_pt = right_p1

                # Apply miterlimit
                max_miter = miterlimit * half_width
                left_dist = math.sqrt((left_pt[0] - p_curr[0])**2 + (left_pt[1] - p_curr[1])**2)
                right_dist = math.sqrt((right_pt[0] - p_curr[0])**2 + (right_pt[1] - p_curr[1])**2)

                if left_dist > max_miter:
                    avg_perp = self._normalize((perp1[0] + perp2[0], perp1[1] + perp2[1]))
                    left_pt = (p_curr[0] + avg_perp[0] * half_width, p_curr[1] + avg_perp[1] * half_width)
                if right_dist > max_miter:
                    avg_perp = self._normalize((perp1[0] + perp2[0], perp1[1] + perp2[1]))
                    right_pt = (p_curr[0] - avg_perp[0] * half_width, p_curr[1] - avg_perp[1] * half_width)
            else:
                # Nearly collinear
                avg_perp = self._normalize((perp1[0] + perp2[0], perp1[1] + perp2[1]))
                left_pt = (p_curr[0] + avg_perp[0] * half_width, p_curr[1] + avg_perp[1] * half_width)
                right_pt = (p_curr[0] - avg_perp[0] * half_width, p_curr[1] - avg_perp[1] * half_width)

            left_points.append(left_pt)
            right_points.append(right_pt)

        # Render stroke as individual segments (quads) plus miter triangles
        # This correctly handles the ring shape for closed polygon strokes
        temp = Image.new("RGBA", ctx.image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp, "RGBA")

        # Draw each edge as a quadrilateral
        for i in range(n):
            j = (i + 1) % n
            quad = [left_points[i], left_points[j], right_points[j], right_points[i]]
            draw.polygon(quad, fill=stroke)

        # Draw round joins at corners if linejoin is "round"
        if linejoin == "round":
            for i in range(n):
                x, y = points[i]
                draw.ellipse([x - half_width, y - half_width,
                              x + half_width, y + half_width], fill=stroke)

        ctx.image.alpha_composite(temp)

    def _stroke_closed_polygon_segmented(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                                          stroke: Tuple[int, int, int, int], half_width: float,
                                          miterlimit: float = 4.0, linejoin: str = "miter"):
        """Render closed polygon stroke using segment-based approach.

        Used for non-convex shapes where the outline approach would self-intersect.
        """
        n = len(points)

        temp = Image.new("RGBA", ctx.image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp, "RGBA")

        # For each edge, draw a quadrilateral
        for i in range(n):
            j = (i + 1) % n
            p1 = points[i]
            p2 = points[j]

            # Direction and perpendicular for this edge
            d = self._normalize(self._subtract(p2, p1))
            perp = (-d[1], d[0])

            # Simple perpendicular offset for each edge
            quad = [
                (p1[0] + perp[0] * half_width, p1[1] + perp[1] * half_width),
                (p2[0] + perp[0] * half_width, p2[1] + perp[1] * half_width),
                (p2[0] - perp[0] * half_width, p2[1] - perp[1] * half_width),
                (p1[0] - perp[0] * half_width, p1[1] - perp[1] * half_width),
            ]
            draw.polygon(quad, fill=stroke)

        # Fill corners with miter triangles
        for i in range(n):
            p_prev = points[(i - 1) % n]
            p_curr = points[i]
            p_next = points[(i + 1) % n]

            d1 = self._normalize(self._subtract(p_curr, p_prev))
            d2 = self._normalize(self._subtract(p_next, p_curr))

            perp1 = (-d1[1], d1[0])
            perp2 = (-d2[1], d2[0])

            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < 0.001:
                continue

            # Compute miter point
            left_p1 = (p_curr[0] + perp1[0] * half_width, p_curr[1] + perp1[1] * half_width)
            left_p2 = (p_curr[0] + perp2[0] * half_width, p_curr[1] + perp2[1] * half_width)
            right_p1 = (p_curr[0] - perp1[0] * half_width, p_curr[1] - perp1[1] * half_width)
            right_p2 = (p_curr[0] - perp2[0] * half_width, p_curr[1] - perp2[1] * half_width)

            if cross > 0:
                # Left turn - miter on inside (right)
                miter_pt = self._line_intersection(right_p1, d1, right_p2, d2)
                if miter_pt:
                    miter_dist = math.sqrt((miter_pt[0] - p_curr[0])**2 + (miter_pt[1] - p_curr[1])**2)
                    if miter_dist <= miterlimit * half_width:
                        tri = [right_p1, miter_pt, right_p2]
                        draw.polygon(tri, fill=stroke)
            else:
                # Right turn - miter on inside (left)
                miter_pt = self._line_intersection(left_p1, d1, left_p2, d2)
                if miter_pt:
                    miter_dist = math.sqrt((miter_pt[0] - p_curr[0])**2 + (miter_pt[1] - p_curr[1])**2)
                    if miter_dist <= miterlimit * half_width:
                        tri = [left_p1, miter_pt, left_p2]
                        draw.polygon(tri, fill=stroke)

        # Draw round joins at corners if linejoin is "round"
        if linejoin == "round":
            for i in range(n):
                x, y = points[i]
                draw.ellipse([x - half_width, y - half_width,
                              x + half_width, y + half_width], fill=stroke)

        ctx.image.alpha_composite(temp)

    def _build_stroke_polygon(self, points: List[Tuple[float, float]],
                              half_width: float, linecap: str, linejoin: str,
                              closed: bool, miterlimit: float = 4.0) -> List[Tuple[float, float]]:
        """Build a polygon representing the stroke outline."""
        if len(points) < 2:
            return []

        # Remove duplicate consecutive points
        clean_points = [points[0]]
        for p in points[1:]:
            if self._point_distance(clean_points[-1], p) > 0.01:
                clean_points.append(p)
        points = clean_points

        if len(points) < 2:
            return []

        left_side = []   # Left side of stroke
        right_side = []  # Right side of stroke

        n = len(points)

        for i in range(n):
            p = points[i]

            # Get direction vectors
            if i == 0:
                # First point
                d = self._normalize(self._subtract(points[1], points[0]))
                perp = (-d[1], d[0])

                if closed and n > 2:
                    # Use join with last segment
                    d_prev = self._normalize(self._subtract(points[0], points[-1]))
                    left_pt, right_pt = self._compute_join(
                        points[-1], p, points[1], d_prev, d, half_width, linejoin, miterlimit
                    )
                else:
                    # Apply linecap
                    left_pt = (p[0] + perp[0] * half_width, p[1] + perp[1] * half_width)
                    right_pt = (p[0] - perp[0] * half_width, p[1] - perp[1] * half_width)

                    if linecap == "square":
                        left_pt = (left_pt[0] - d[0] * half_width, left_pt[1] - d[1] * half_width)
                        right_pt = (right_pt[0] - d[0] * half_width, right_pt[1] - d[1] * half_width)
                    elif linecap == "round":
                        # Add round cap points
                        cap_points = self._round_cap(p, d, half_width, start=True)
                        left_side.extend(cap_points)

            elif i == n - 1:
                # Last point
                d = self._normalize(self._subtract(points[i], points[i-1]))
                perp = (-d[1], d[0])

                if closed and n > 2:
                    # Use join with first segment
                    d_next = self._normalize(self._subtract(points[1], points[0]))
                    left_pt, right_pt = self._compute_join(
                        points[i-1], p, points[0], d, d_next, half_width, linejoin, miterlimit
                    )
                else:
                    # Apply linecap
                    left_pt = (p[0] + perp[0] * half_width, p[1] + perp[1] * half_width)
                    right_pt = (p[0] - perp[0] * half_width, p[1] - perp[1] * half_width)

                    if linecap == "square":
                        left_pt = (left_pt[0] + d[0] * half_width, left_pt[1] + d[1] * half_width)
                        right_pt = (right_pt[0] + d[0] * half_width, right_pt[1] + d[1] * half_width)

            else:
                # Middle point - compute join
                d_prev = self._normalize(self._subtract(p, points[i-1]))
                d_next = self._normalize(self._subtract(points[i+1], p))
                left_pt, right_pt = self._compute_join(
                    points[i-1], p, points[i+1], d_prev, d_next, half_width, linejoin, miterlimit
                )

            left_side.append(left_pt)
            right_side.append(right_pt)

            # Add end cap for last point (non-closed paths)
            if i == n - 1 and not closed and linecap == "round":
                d = self._normalize(self._subtract(points[i], points[i-1]))
                cap_points = self._round_cap(p, d, half_width, start=False)
                right_side.extend(cap_points)

        # Combine into single polygon (left side forward, right side backward)
        right_side.reverse()
        return left_side + right_side

    def _compute_join(self, p_prev: Tuple[float, float], p: Tuple[float, float],
                      p_next: Tuple[float, float], d_prev: Tuple[float, float],
                      d_next: Tuple[float, float], half_width: float,
                      linejoin: str, miterlimit: float = 4.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute join points at a vertex."""
        perp_prev = (-d_prev[1], d_prev[0])
        perp_next = (-d_next[1], d_next[0])

        # Check turn direction (cross product)
        cross = d_prev[0] * d_next[1] - d_prev[1] * d_next[0]

        if abs(cross) < 0.001:
            # Nearly collinear - use simple perpendicular
            left_pt = (p[0] + perp_prev[0] * half_width, p[1] + perp_prev[1] * half_width)
            right_pt = (p[0] - perp_prev[0] * half_width, p[1] - perp_prev[1] * half_width)
            return left_pt, right_pt

        # Calculate max miter length based on miterlimit
        max_miter_length = miterlimit * half_width

        if linejoin == "bevel" or linejoin == "round":
            # For bevel/round, use the outer points directly
            left_prev = (p[0] + perp_prev[0] * half_width, p[1] + perp_prev[1] * half_width)
            left_next = (p[0] + perp_next[0] * half_width, p[1] + perp_next[1] * half_width)
            right_prev = (p[0] - perp_prev[0] * half_width, p[1] - perp_prev[1] * half_width)
            right_next = (p[0] - perp_next[0] * half_width, p[1] - perp_next[1] * half_width)

            if cross > 0:  # Left turn
                # Miter on right, bevel on left
                right_pt = self._line_intersection(
                    (p[0] - perp_prev[0] * half_width, p[1] - perp_prev[1] * half_width),
                    d_prev,
                    (p[0] - perp_next[0] * half_width, p[1] - perp_next[1] * half_width),
                    d_next
                )
                if right_pt is None:
                    right_pt = right_prev
                left_pt = left_prev  # Bevel uses the corner point
            else:  # Right turn
                left_pt = self._line_intersection(
                    left_prev, d_prev, left_next, d_next
                )
                if left_pt is None:
                    left_pt = left_prev
                right_pt = right_prev

            return left_pt, right_pt

        else:  # miter
            # Compute miter intersection
            left_prev = (p[0] + perp_prev[0] * half_width, p[1] + perp_prev[1] * half_width)
            left_next = (p[0] + perp_next[0] * half_width, p[1] + perp_next[1] * half_width)
            right_prev = (p[0] - perp_prev[0] * half_width, p[1] - perp_prev[1] * half_width)
            right_next = (p[0] - perp_next[0] * half_width, p[1] - perp_next[1] * half_width)

            left_pt = self._line_intersection(left_prev, d_prev, left_next, d_next)
            right_pt = self._line_intersection(right_prev, d_prev, right_next, d_next)

            if left_pt is None:
                left_pt = left_prev
            if right_pt is None:
                right_pt = right_prev

            # Apply miterlimit - if miter extends too far, fall back to bevel
            if left_pt:
                left_dist = math.sqrt((left_pt[0] - p[0])**2 + (left_pt[1] - p[1])**2)
                if left_dist > max_miter_length:
                    # Fall back to bevel-like point
                    avg_perp = self._normalize((perp_prev[0] + perp_next[0], perp_prev[1] + perp_next[1]))
                    left_pt = (p[0] + avg_perp[0] * half_width, p[1] + avg_perp[1] * half_width)

            if right_pt:
                right_dist = math.sqrt((right_pt[0] - p[0])**2 + (right_pt[1] - p[1])**2)
                if right_dist > max_miter_length:
                    # Fall back to bevel-like point
                    avg_perp = self._normalize((perp_prev[0] + perp_next[0], perp_prev[1] + perp_next[1]))
                    right_pt = (p[0] - avg_perp[0] * half_width, p[1] - avg_perp[1] * half_width)

            return left_pt, right_pt

    def _round_cap(self, p: Tuple[float, float], d: Tuple[float, float],
                   half_width: float, start: bool) -> List[Tuple[float, float]]:
        """Generate points for a round line cap."""
        points = []
        n_points = 8
        perp = (-d[1], d[0])

        if start:
            # Start cap - goes from right to left (clockwise)
            for i in range(n_points + 1):
                angle = math.pi / 2 + math.pi * i / n_points
                px = p[0] + half_width * (d[0] * math.cos(angle) - d[1] * math.sin(angle))
                py = p[1] + half_width * (d[1] * math.cos(angle) + d[0] * math.sin(angle))
                points.append((px, py))
        else:
            # End cap - goes from left to right
            for i in range(n_points + 1):
                angle = -math.pi / 2 + math.pi * i / n_points
                px = p[0] + half_width * (d[0] * math.cos(angle) - d[1] * math.sin(angle))
                py = p[1] + half_width * (d[1] * math.cos(angle) + d[0] * math.sin(angle))
                points.append((px, py))

        return points

    def _line_intersection(self, p1: Tuple[float, float], d1: Tuple[float, float],
                           p2: Tuple[float, float], d2: Tuple[float, float]
                           ) -> Optional[Tuple[float, float]]:
        """Find intersection of two lines defined by point and direction."""
        det = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(det) < 1e-10:
            return None

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        t = (dx * d2[1] - dy * d2[0]) / det

        return (p1[0] + t * d1[0], p1[1] + t * d1[1])

    def _normalize(self, v: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize a 2D vector."""
        length = math.sqrt(v[0] * v[0] + v[1] * v[1])
        if length < 1e-10:
            return (1.0, 0.0)
        return (v[0] / length, v[1] / length)

    def _subtract(self, a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        """Subtract two 2D points/vectors."""
        return (a[0] - b[0], a[1] - b[1])

    def _point_distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _fill_and_stroke_polygon(self, ctx: "RenderContext",
                                 points: list[tuple[float, float]],
                                 style: Style,
                                 element_transform: Transform,
                                 bbox: tuple[float, float, float, float]):
        """Fill and stroke a polygon."""
        fill = self._get_fill_color(ctx, style)
        fill_ref = style.fill if isinstance(style.fill, str) else None

        # Fill
        if len(points) >= 3:
            self._fill_polygon_with_gradient_check(
                ctx, points, style, element_transform, bbox, fill, fill_ref
            )

        # Stroke with proper linecap/linejoin
        if len(points) >= 2:
            self._stroke_path(ctx, points, style, element_transform, closed=True)

    def _fill_polygon_with_gradient_check(self, ctx: "RenderContext",
                                          points: list[tuple[float, float]],
                                          style: Style,
                                          element_transform: Transform,
                                          bbox: tuple[float, float, float, float],
                                          fill: Optional[tuple[int, int, int, int]],
                                          fill_ref: Optional[str]):
        """Fill a polygon, handling gradients if needed."""
        if fill_ref and fill_ref.startswith("url("):
            # Extract gradient ID - handle fallback colors like "url(#id) rgb(0,0,0)"
            end_paren = fill_ref.find(")")
            if end_paren != -1:
                match = fill_ref[4:end_paren]  # Remove "url(" and extract up to ")"
            else:
                match = fill_ref[4:]
            if match.startswith("#"):
                match = match[1:]

            if match in ctx.gradients:
                gradient = ctx.gradients[match]
                self._fill_polygon_with_gradient(ctx, points, gradient, bbox,
                                                 style.fill_opacity * style.opacity,
                                                 style.fill_rule,
                                                 element_transform)
                return

        # Simple fill with fill-rule support
        if fill and len(points) >= 3:
            self._fill_polygon_with_rule(ctx, points, fill, style.fill_rule)

    def _fill_polygon_with_rule(self, ctx: "RenderContext",
                                points: list[tuple[float, float]],
                                fill: tuple[int, int, int, int],
                                fill_rule: str):
        """Fill a polygon with the specified fill rule."""
        if fill_rule == "evenodd":
            self._fill_polygon_evenodd(ctx, points, fill)
        else:
            # Check if polygon is self-intersecting (stars, complex shapes)
            # PIL's polygon() uses even-odd rule, so we need scanline nonzero for self-intersecting
            if self._is_self_intersecting(points):
                self._fill_polygon_nonzero_color(ctx, points, fill)
            else:
                # Simple non-intersecting polygon - PIL's polygon works fine
                if fill[3] < 255:
                    # Memory-optimized: use cropped-size temp instead of full-size
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    min_x, max_x = max(0, int(min(xs))), min(ctx.image.width, int(max(xs)) + 1)
                    min_y, max_y = max(0, int(min(ys))), min(ctx.image.height, int(max(ys)) + 1)
                    if min_x < max_x and min_y < max_y:
                        temp = Image.new("RGBA", (max_x - min_x, max_y - min_y), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(temp, "RGBA")
                        local_points = [(x - min_x, y - min_y) for x, y in points]
                        draw.polygon(local_points, fill=fill)
                        ctx.image.alpha_composite(temp, (min_x, min_y))
                else:
                    draw = ImageDraw.Draw(ctx.image, "RGBA")
                    draw.polygon(points, fill=fill)

    def _is_self_intersecting(self, points: list[tuple[float, float]]) -> bool:
        """Check if a polygon has self-intersecting edges (optimized)."""
        # Use Rust implementation if available (140x faster)
        if HAS_RUST:
            return vectorstag_rust.is_self_intersecting(points)

        n = len(points)
        if n < 4:
            return False

        # For very complex polygons, assume they might be self-intersecting
        # to avoid O(n²) complexity on thousands of edges
        if n > 200:
            return True  # Conservative: use scanline for complex shapes

        # Close the polygon
        pts = list(points)
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        n = len(pts) - 1

        # Use numpy for faster computation
        pts_arr = np.array(pts)

        # Check only a sample of edge pairs for medium-sized polygons
        max_checks = 5000
        total_pairs = n * (n - 3) // 2  # Non-adjacent pairs

        if total_pairs <= max_checks:
            # Full check for smaller polygons
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    if self._segments_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                        return True
        else:
            # Sample-based check for medium polygons
            import random
            checked = 0
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    if self._segments_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                        return True
                    checked += 1
                    if checked >= max_checks:
                        return False  # Assume OK if no intersection found in sample

        return False

    def _segments_intersect(self, A, B, C, D) -> bool:
        """Check if line segment AB intersects with CD."""
        def ccw(P, Q, R):
            return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def _fill_polygon_nonzero_color(self, ctx: "RenderContext",
                                     points: list[tuple[float, float]],
                                     fill: tuple[int, int, int, int]):
        """Fill a polygon using nonzero winding rule with a color (vectorized)."""
        if len(points) < 3:
            return

        width, height = ctx.image.size

        # Get bounding box using numpy for speed
        pts_arr = np.array(points)
        min_x = max(0, int(pts_arr[:, 0].min()))
        max_x = min(width - 1, int(pts_arr[:, 0].max()) + 1)
        min_y = max(0, int(pts_arr[:, 1].min()))
        max_y = min(height - 1, int(pts_arr[:, 1].max()) + 1)

        if min_x >= max_x or min_y >= max_y:
            return

        crop_width = max_x - min_x
        crop_height = max_y - min_y

        # Use Rust implementation if available (much faster)
        if HAS_RUST:
            mask_arr = vectorstag_rust.fill_polygon_nonzero(
                points, crop_width, crop_height, min_x, min_y
            )
        else:
            # Create mask using nonzero winding rule
            mask_arr = np.zeros((crop_height, crop_width), dtype=np.uint8)

            # Close the polygon
            pts = np.vstack([pts_arr, pts_arr[0:1]]) if not np.allclose(pts_arr[0], pts_arr[-1]) else pts_arr

            # Build edge arrays for vectorized processing
            p1 = pts[:-1]
            p2 = pts[1:]

            # Filter non-horizontal edges
            non_horiz = p1[:, 1] != p2[:, 1]
            if not np.any(non_horiz):
                return

            p1_f = p1[non_horiz]
            p2_f = p2[non_horiz]

            # Determine direction and sort so p1.y < p2.y
            swap = p1_f[:, 1] > p2_f[:, 1]
            p1_f[swap], p2_f[swap] = p2_f[swap].copy(), p1_f[swap].copy()
            directions = np.where(swap, -1, 1)

            # Extract edge data
            x1 = p1_f[:, 0]
            y1 = p1_f[:, 1]
            x2 = p2_f[:, 0]
            y2 = p2_f[:, 1]
            dy = y2 - y1
            dx = x2 - x1

            # Vectorized scanline fill
            for y in range(crop_height):
                screen_y = y + min_y + 0.5

                # Find edges that cross this scanline
                active = (y1 <= screen_y) & (screen_y < y2)
                if not np.any(active):
                    continue

                # Compute x intersections for active edges
                t = (screen_y - y1[active]) / dy[active]
                x_intersects = x1[active] + t * dx[active]
                dirs = directions[active]

                # Sort by x
                sort_idx = np.argsort(x_intersects)
                x_sorted = x_intersects[sort_idx]
                dirs_sorted = dirs[sort_idx]

                # Fill using winding count (nonzero rule)
                winding = 0
                prev_x = None
                for i in range(len(x_sorted)):
                    x_int = x_sorted[i]
                    direction = dirs_sorted[i]
                    if winding != 0 and prev_x is not None:
                        x_start = max(0, int(prev_x - min_x))
                        x_end = min(crop_width, int(x_int - min_x))
                        if x_start < x_end:
                            mask_arr[y, x_start:x_end] = 255
                    winding += direction
                    prev_x = x_int

        # Apply fill using mask (memory-optimized: no full-size temp)
        mask_img = Image.fromarray(mask_arr, "L")
        fill_img = Image.new("RGBA", (crop_width, crop_height), fill[:3] + (255,))
        self._composite_masked_fill(ctx, fill_img, mask_img, min_x, min_y, fill[3])

    def _fill_multi_polygon_evenodd(self, ctx: "RenderContext",
                                     polygons: list[list[tuple[float, float]]],
                                     fill: Optional[tuple[int, int, int, int]],
                                     fill_ref: Optional[str],
                                     style: Style,
                                     element_transform: Transform,
                                     bbox: tuple[float, float, float, float]):
        """Fill multiple polygons using evenodd rule (creates holes where they overlap)."""
        if not polygons:
            return

        # Collect all points and compute combined bounding box
        all_xs = []
        all_ys = []
        for poly in polygons:
            all_xs.extend(p[0] for p in poly)
            all_ys.extend(p[1] for p in poly)

        if not all_xs:
            return

        min_x, max_x = int(min(all_xs)), int(max(all_xs)) + 1
        min_y, max_y = int(min(all_ys)), int(max(all_ys)) + 1

        # Clip to image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(ctx.image.width, max_x)
        max_y = min(ctx.image.height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return

        width = max_x - min_x
        height = max_y - min_y

        # Use Rust implementation if available (much faster)
        if HAS_RUST:
            mask = vectorstag_rust.fill_multi_polygon_evenodd(polygons, width, height, min_x, min_y)
        else:
            # Create mask using scanline algorithm with even-odd rule across ALL polygons
            mask = np.zeros((height, width), dtype=np.uint8)

            # Build edge list from all polygons
            all_edges = []
            for poly in polygons:
                closed_points = list(poly)
                if closed_points[0] != closed_points[-1]:
                    closed_points.append(closed_points[0])

                n = len(closed_points) - 1
                for i in range(n):
                    p1 = closed_points[i]
                    p2 = closed_points[i + 1]
                    all_edges.append((p1, p2))

            # For each scanline
            for y in range(height):
                screen_y = y + min_y + 0.5  # Center of pixel

                # Find intersections with all edges from all polygons
                intersections = []

                for p1, p2 in all_edges:
                    # Check if edge crosses this scanline
                    if (p1[1] <= screen_y < p2[1]) or (p2[1] <= screen_y < p1[1]):
                        # Compute x intersection
                        if abs(p2[1] - p1[1]) > 1e-10:
                            t = (screen_y - p1[1]) / (p2[1] - p1[1])
                            x_intersect = p1[0] + t * (p2[0] - p1[0])
                            intersections.append(x_intersect)

                # Sort intersections
                intersections.sort()

                # Fill between pairs (even-odd rule)
                for i in range(0, len(intersections) - 1, 2):
                    x_start = max(0, int(intersections[i] - min_x))
                    x_end = min(width, int(intersections[i + 1] - min_x) + 1)
                    mask[y, x_start:x_end] = 255

        # Apply fill using mask (memory-optimized: no full-size temp)
        mask_img = Image.fromarray(mask, "L")

        if fill_ref and fill_ref.startswith("url("):
            # Gradient fill
            match = fill_ref[4:-1]
            if match.startswith("#"):
                match = match[1:]
            if match in ctx.gradients:
                gradient = ctx.gradients[match]
                grad_img = self._create_gradient_for_mask(
                    ctx, gradient, width, height, bbox, min_x, min_y,
                    style.fill_opacity * style.opacity,
                    element_transform
                )
                self._composite_gradient_masked(ctx, grad_img, mask_img, min_x, min_y)
        elif fill:
            # Solid fill
            fill_img = Image.new("RGBA", (width, height), fill[:3] + (255,))
            self._composite_masked_fill(ctx, fill_img, mask_img, min_x, min_y, fill[3])

    def _fill_multi_polygon_nonzero(self, ctx: "RenderContext",
                                      polygons: list[list[tuple[float, float]]],
                                      fill: Optional[tuple[int, int, int, int]],
                                      fill_ref: Optional[str],
                                      style: Style,
                                      element_transform: Transform,
                                      bbox: tuple[float, float, float, float]):
        """Fill multiple polygons using nonzero winding rule (creates holes with opposite winding)."""
        if not polygons:
            return

        # Collect all points and compute combined bounding box
        all_xs = []
        all_ys = []
        for poly in polygons:
            all_xs.extend(p[0] for p in poly)
            all_ys.extend(p[1] for p in poly)

        if not all_xs:
            return

        min_x, max_x = int(min(all_xs)), int(max(all_xs)) + 1
        min_y, max_y = int(min(all_ys)), int(max(all_ys)) + 1

        # Clip to image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(ctx.image.width, max_x)
        max_y = min(ctx.image.height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return

        width = max_x - min_x
        height = max_y - min_y

        # Create mask using scanline algorithm with nonzero winding rule
        mask = np.zeros((height, width), dtype=np.uint8)

        # Build edge list with direction from all polygons
        all_edges = []
        for poly in polygons:
            closed_points = list(poly)
            if closed_points[0] != closed_points[-1]:
                closed_points.append(closed_points[0])

            n = len(closed_points) - 1
            for i in range(n):
                p1 = closed_points[i]
                p2 = closed_points[i + 1]
                if p1[1] != p2[1]:  # Skip horizontal edges
                    # Determine direction: +1 if going up, -1 if going down
                    if p1[1] > p2[1]:
                        p1, p2 = p2, p1
                        direction = -1
                    else:
                        direction = 1
                    all_edges.append((p1[0], p1[1], p2[0], p2[1], direction))

        # For each scanline
        for y in range(height):
            screen_y = y + min_y + 0.5  # Center of pixel

            # Find intersections with direction
            intersections = []
            for x1, y1, x2, y2, direction in all_edges:
                if y1 <= screen_y < y2:
                    t = (screen_y - y1) / (y2 - y1)
                    x_intersect = x1 + t * (x2 - x1)
                    intersections.append((x_intersect, direction))

            # Sort by x
            intersections.sort(key=lambda p: p[0])

            # Fill using winding count (nonzero rule)
            winding = 0
            prev_x = None
            for x_int, direction in intersections:
                if winding != 0 and prev_x is not None:
                    x_start = max(0, int(prev_x - min_x))
                    x_end = min(width, int(x_int - min_x))
                    if x_start < x_end:
                        mask[y, x_start:x_end] = 255
                winding += direction
                prev_x = x_int

        # Apply fill using mask (memory-optimized: no full-size temp)
        mask_img = Image.fromarray(mask, "L")

        if fill_ref and fill_ref.startswith("url("):
            # Gradient fill
            match = fill_ref[4:-1]
            if match.startswith("#"):
                match = match[1:]
            if match in ctx.gradients:
                gradient = ctx.gradients[match]
                grad_img = self._create_gradient_for_mask(
                    ctx, gradient, width, height, bbox, min_x, min_y,
                    style.fill_opacity * style.opacity,
                    element_transform
                )
                self._composite_gradient_masked(ctx, grad_img, mask_img, min_x, min_y)
        elif fill:
            # Solid fill
            fill_img = Image.new("RGBA", (width, height), fill[:3] + (255,))
            self._composite_masked_fill(ctx, fill_img, mask_img, min_x, min_y, fill[3])

    def _create_gradient_for_mask(self, ctx: "RenderContext",
                                   gradient, width: int, height: int,
                                   bbox: tuple[float, float, float, float],
                                   offset_x: int, offset_y: int,
                                   opacity: float,
                                   element_transform: Transform = None) -> Image.Image:
        """Create gradient image for use with mask."""
        if isinstance(gradient, LinearGradient):
            return self._create_linear_gradient_image(
                ctx, gradient, width, height, bbox, offset_x, offset_y, opacity,
                element_transform
            )
        else:
            return self._create_radial_gradient_image(
                ctx, gradient, width, height, bbox, offset_x, offset_y, opacity,
                element_transform
            )

    def _fill_polygon_evenodd(self, ctx: "RenderContext",
                              points: list[tuple[float, float]],
                              fill: tuple[int, int, int, int]):
        """Fill a polygon using the even-odd rule."""
        if len(points) < 3:
            return

        # Get bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = int(min(xs)), int(max(xs)) + 1
        min_y, max_y = int(min(ys)), int(max(ys)) + 1

        # Clip to image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(ctx.image.width, max_x)
        max_y = min(ctx.image.height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return

        width = max_x - min_x
        height = max_y - min_y

        # Use Rust implementation if available (much faster)
        if HAS_RUST:
            mask = vectorstag_rust.fill_polygon_evenodd(points, width, height, min_x, min_y)
        else:
            # Create mask using scanline algorithm with even-odd rule
            mask = np.zeros((height, width), dtype=np.uint8)

            # Close the polygon
            closed_points = list(points)
            if closed_points[0] != closed_points[-1]:
                closed_points.append(closed_points[0])

            n = len(closed_points) - 1

            # For each scanline
            for y in range(height):
                screen_y = y + min_y + 0.5  # Center of pixel

                # Find intersections with all edges
                intersections = []

                for i in range(n):
                    p1 = closed_points[i]
                    p2 = closed_points[i + 1]

                    # Check if edge crosses this scanline
                    if (p1[1] <= screen_y < p2[1]) or (p2[1] <= screen_y < p1[1]):
                        # Compute x intersection
                        if abs(p2[1] - p1[1]) > 1e-10:
                            t = (screen_y - p1[1]) / (p2[1] - p1[1])
                            x_intersect = p1[0] + t * (p2[0] - p1[0])
                            intersections.append(x_intersect)

                # Sort intersections
                intersections.sort()

                # Fill between pairs (even-odd rule)
                for i in range(0, len(intersections) - 1, 2):
                    x_start = max(0, int(intersections[i] - min_x))
                    x_end = min(width, int(intersections[i + 1] - min_x) + 1)
                    mask[y, x_start:x_end] = 255

        # Apply fill using mask (memory-optimized: no full-size temp)
        mask_img = Image.fromarray(mask, "L")
        fill_img = Image.new("RGBA", (width, height), fill[:3] + (255,))
        self._composite_masked_fill(ctx, fill_img, mask_img, min_x, min_y, fill[3])

    def _fill_polygon_with_gradient(self, ctx: "RenderContext",
                                    points: list[tuple[float, float]],
                                    gradient: Union[LinearGradient, RadialGradient],
                                    bbox: tuple[float, float, float, float],
                                    opacity: float,
                                    fill_rule: str = "nonzero",
                                    element_transform: Transform = None):
        """Fill a polygon with a gradient."""
        if not points or len(points) < 3:
            return

        # Get bounds of transformed points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = int(min(xs)), int(max(xs)) + 1
        min_y, max_y = int(min(ys)), int(max(ys)) + 1

        if min_x >= max_x or min_y >= max_y:
            return

        # Clip to image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(ctx.image.width, max_x)
        max_y = min(ctx.image.height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return

        # Create gradient image for the bounding box
        grad_width = max_x - min_x
        grad_height = max_y - min_y

        if isinstance(gradient, LinearGradient):
            grad_img = self._create_linear_gradient_image(
                ctx, gradient, grad_width, grad_height, bbox, min_x, min_y, opacity,
                element_transform
            )
        else:
            grad_img = self._create_radial_gradient_image(
                ctx, gradient, grad_width, grad_height, bbox, min_x, min_y, opacity,
                element_transform
            )

        # Create mask from polygon with fill-rule support
        if fill_rule == "evenodd":
            mask_crop = self._create_evenodd_mask(points, min_x, min_y, grad_width, grad_height)
        else:
            # Create cropped-size mask directly (memory-optimized)
            mask_crop = Image.new("L", (grad_width, grad_height), 0)
            mask_draw = ImageDraw.Draw(mask_crop)
            # Offset points to local coordinates
            local_points = [(x - min_x, y - min_y) for x, y in points]
            mask_draw.polygon(local_points, fill=255)

        # Apply gradient with mask (memory-optimized: no full-size temp)
        self._composite_gradient_masked(ctx, grad_img, mask_crop, min_x, min_y)

    def _create_evenodd_mask(self, points: list[tuple[float, float]],
                             min_x: int, min_y: int, width: int, height: int) -> Image.Image:
        """Create a mask using even-odd fill rule."""
        mask = np.zeros((height, width), dtype=np.uint8)

        # Close the polygon
        closed_points = list(points)
        if closed_points[0] != closed_points[-1]:
            closed_points.append(closed_points[0])

        n = len(closed_points) - 1

        # For each scanline
        for y in range(height):
            screen_y = y + min_y + 0.5

            intersections = []
            for i in range(n):
                p1 = closed_points[i]
                p2 = closed_points[i + 1]

                if (p1[1] <= screen_y < p2[1]) or (p2[1] <= screen_y < p1[1]):
                    if abs(p2[1] - p1[1]) > 1e-10:
                        t = (screen_y - p1[1]) / (p2[1] - p1[1])
                        x_intersect = p1[0] + t * (p2[0] - p1[0])
                        intersections.append(x_intersect)

            intersections.sort()

            for i in range(0, len(intersections) - 1, 2):
                x_start = max(0, int(intersections[i] - min_x))
                x_end = min(width, int(intersections[i + 1] - min_x) + 1)
                mask[y, x_start:x_end] = 255

        return Image.fromarray(mask, "L")

    def _create_linear_gradient_image(self, ctx: "RenderContext",
                                      gradient: LinearGradient,
                                      width: int, height: int,
                                      bbox: tuple[float, float, float, float],
                                      offset_x: int, offset_y: int,
                                      opacity: float,
                                      element_transform: Transform = None) -> Image.Image:
        """Create an image filled with a linear gradient (memory-optimized)."""
        if not gradient.stops:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Get gradient vector in gradient space
        if gradient.units == "objectBoundingBox":
            bx, by, bw, bh = bbox
            x1 = bx + gradient.x1 * bw
            y1 = by + gradient.y1 * bh
            x2 = bx + gradient.x2 * bw
            y2 = by + gradient.y2 * bh
        else:
            x1, y1 = gradient.x1, gradient.y1
            x2, y2 = gradient.x2, gradient.y2

        # Apply gradient transform if present
        if gradient.transform:
            x1, y1 = gradient.transform.apply(x1, y1)
            x2, y2 = gradient.transform.apply(x2, y2)

        # For userSpaceOnUse, apply element transform then base transform
        if gradient.units == "userSpaceOnUse" and element_transform:
            x1, y1 = element_transform.apply(x1, y1)
            x2, y2 = element_transform.apply(x2, y2)
            x1, y1 = ctx.base_transform.apply(x1, y1)
            x2, y2 = ctx.base_transform.apply(x2, y2)
        else:
            x1, y1 = ctx.base_transform.apply(x1, y1)
            x2, y2 = ctx.base_transform.apply(x2, y2)

        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length == 0:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))

        dx /= length
        dy /= length

        # Memory-optimized: use float32 and reuse arrays
        # Create single coordinate array, compute t in-place
        t = np.empty((height, width), dtype=np.float32)
        y_base = np.arange(height, dtype=np.float32) + offset_y
        x_base = np.arange(width, dtype=np.float32) + offset_x

        # Compute t row by row to minimize peak memory
        for row in range(height):
            wy = y_base[row]
            t[row, :] = ((x_base - x1) * dx + (wy - y1) * dy) / length

        # Apply spreadMethod in-place
        spread_method = getattr(gradient, 'spread_method', 'pad')
        if spread_method == "repeat":
            np.remainder(t, 1.0, out=t)
        elif spread_method == "reflect":
            np.remainder(t, 2.0, out=t)
            mask = t > 1.0
            t[mask] = 2.0 - t[mask]
        else:  # pad
            np.clip(t, 0, 1, out=t)

        # Vectorized color interpolation
        pixels = self._interpolate_gradient_colors_vectorized(gradient.stops, t, opacity)
        return Image.fromarray(pixels, "RGBA")

    def _create_radial_gradient_image(self, ctx: "RenderContext",
                                      gradient: RadialGradient,
                                      width: int, height: int,
                                      bbox: tuple[float, float, float, float],
                                      offset_x: int, offset_y: int,
                                      opacity: float,
                                      element_transform: Transform = None) -> Image.Image:
        """Create an image filled with a radial gradient (memory-optimized)."""
        if not gradient.stops:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Get gradient parameters in gradient space
        if gradient.units == "objectBoundingBox":
            bx, by, bw, bh = bbox
            cx = bx + gradient.cx * bw
            cy = by + gradient.cy * bh
            r = gradient.r * max(bw, bh)
        else:
            cx, cy = gradient.cx, gradient.cy
            r = gradient.r

        if r == 0:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Build combined transform and compute inverse
        combined_transform = ctx.base_transform
        if gradient.units == "userSpaceOnUse" and element_transform:
            combined_transform = ctx.base_transform.multiply(element_transform)
        if gradient.transform:
            combined_transform = combined_transform.multiply(gradient.transform)

        det = combined_transform.a * combined_transform.d - combined_transform.b * combined_transform.c
        if abs(det) < 1e-10:
            det = 1e-10

        inv_a = float(combined_transform.d / det)
        inv_b = float(-combined_transform.c / det)
        inv_c = float(-combined_transform.b / det)
        inv_d = float(combined_transform.a / det)
        inv_e = float((combined_transform.c * combined_transform.f - combined_transform.d * combined_transform.e) / det)
        inv_f = float((combined_transform.b * combined_transform.e - combined_transform.a * combined_transform.f) / det)

        # Memory-optimized: use float32 and compute row by row
        t = np.empty((height, width), dtype=np.float32)
        x_base = np.arange(width, dtype=np.float32) + offset_x

        for row in range(height):
            wy = float(row + offset_y)
            # Inverse transform to gradient space
            gx = inv_a * x_base + (inv_b * wy + inv_e)
            gy = inv_c * x_base + (inv_d * wy + inv_f)
            # Distance from center, normalized
            t[row, :] = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) / r

        # Apply spreadMethod in-place
        spread_method = getattr(gradient, 'spread_method', 'pad')
        if spread_method == "repeat":
            np.remainder(t, 1.0, out=t)
        elif spread_method == "reflect":
            np.remainder(t, 2.0, out=t)
            mask = t > 1.0
            t[mask] = 2.0 - t[mask]
        else:  # pad
            np.clip(t, 0, 1, out=t)

        # Vectorized color interpolation
        pixels = self._interpolate_gradient_colors_vectorized(gradient.stops, t, opacity)
        return Image.fromarray(pixels, "RGBA")

    def _interpolate_gradient_colors_vectorized(self, stops: list[GradientStop],
                                                  t: np.ndarray, opacity: float) -> np.ndarray:
        """Memory-optimized vectorized color interpolation for gradient images."""
        height, width = t.shape
        pixels = np.empty((height, width, 4), dtype=np.uint8)

        if not stops:
            pixels.fill(0)
            return pixels

        # Build arrays of stop offsets and colors
        offsets = np.array([s.offset for s in stops], dtype=np.float32)
        colors = np.array([s.color for s in stops], dtype=np.float32)

        # Use searchsorted to find which stop segment each pixel belongs to
        # This is more memory efficient than creating masks for each segment
        indices = np.searchsorted(offsets, t, side='right') - 1
        np.clip(indices, 0, len(stops) - 2, out=indices)

        # Pre-compute ratios for all pixels
        # ratio = (t - offset[i]) / (offset[i+1] - offset[i])
        lower_offsets = offsets[indices]
        upper_offsets = offsets[indices + 1]
        denom = upper_offsets - lower_offsets

        # Avoid division by zero - where denom is tiny, use 0 ratio
        safe_denom = np.where(denom > 1e-10, denom, 1.0)
        ratio = np.clip((t - lower_offsets) / safe_denom, 0, 1)
        ratio = np.where(denom > 1e-10, ratio, 0.0).astype(np.float32)

        # Interpolate each color channel
        for c in range(4):
            lower_colors = colors[indices, c]
            upper_colors = colors[indices + 1, c]
            interp = lower_colors + ratio * (upper_colors - lower_colors)
            if c == 3:  # Alpha channel
                pixels[:, :, c] = (interp * opacity).astype(np.uint8)
            else:
                pixels[:, :, c] = interp.astype(np.uint8)

        # Clean up large temporaries
        del indices, lower_offsets, upper_offsets, denom, safe_denom, ratio

        return pixels

    def _interpolate_gradient_color(self, stops: list[GradientStop],
                                    t: float) -> tuple[int, int, int, int]:
        """Interpolate color at position t along gradient."""
        if not stops:
            return (0, 0, 0, 0)

        if t <= stops[0].offset:
            return stops[0].color

        if t >= stops[-1].offset:
            return stops[-1].color

        # Find surrounding stops
        for i in range(len(stops) - 1):
            if stops[i].offset <= t <= stops[i + 1].offset:
                s1, s2 = stops[i], stops[i + 1]
                if s2.offset == s1.offset:
                    return s1.color

                # Interpolate
                ratio = (t - s1.offset) / (s2.offset - s1.offset)
                return (
                    int(s1.color[0] + ratio * (s2.color[0] - s1.color[0])),
                    int(s1.color[1] + ratio * (s2.color[1] - s1.color[1])),
                    int(s1.color[2] + ratio * (s2.color[2] - s1.color[2])),
                    int(s1.color[3] + ratio * (s2.color[3] - s1.color[3]))
                )

        return stops[-1].color

    def _composite_masked_fill(self, ctx: "RenderContext", fill_img: Image.Image,
                               mask_img: Image.Image, dest_x: int, dest_y: int,
                               fill_alpha: int = 255):
        """Composite a masked fill without creating a full-size temp image.

        Args:
            ctx: Render context
            fill_img: The fill image (same size as mask)
            mask_img: The mask image (L mode)
            dest_x, dest_y: Destination coordinates on ctx.image
            fill_alpha: Alpha value of the fill (0-255)
        """
        # Apply mask to fill's alpha channel
        if fill_alpha < 255:
            # Scale mask by fill alpha for semi-transparent fills
            scaled_mask = mask_img.point(lambda x: x * fill_alpha // 255)
            fill_img.putalpha(scaled_mask)
        else:
            fill_img.putalpha(mask_img)

        # Composite directly at destination without full-size temp
        ctx.image.alpha_composite(fill_img, (dest_x, dest_y))

    def _composite_gradient_masked(self, ctx: "RenderContext", grad_img: Image.Image,
                                    mask_img: Image.Image, dest_x: int, dest_y: int):
        """Composite a gradient with mask without creating a full-size temp image."""
        # Apply mask to gradient's alpha channel
        grad_r, grad_g, grad_b, grad_a = grad_img.split()
        masked_alpha = ImageChops.multiply(grad_a, mask_img)
        grad_masked = Image.merge("RGBA", (grad_r, grad_g, grad_b, masked_alpha))

        # Composite directly at destination
        ctx.image.alpha_composite(grad_masked, (dest_x, dest_y))

    # Font mapping for common font families
    FONT_PATHS = {
        # Serif fonts
        "serif": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/System/Library/Fonts/Times.ttc",
            "times.ttf",
        ],
        "times": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/System/Library/Fonts/Times.ttc",
        ],
        "times new roman": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/System/Library/Fonts/Times.ttc",
        ],
        # Sans-serif fonts
        "sans-serif": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "arial.ttf",
        ],
        "arial": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "arial.ttf",
        ],
        "helvetica": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ],
        # Monospace fonts
        "monospace": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/System/Library/Fonts/Courier.ttc",
            "cour.ttf",
        ],
        "courier": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Courier.ttc",
        ],
    }

    def _get_font(self, font_family: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Get a font for the given family and size."""
        # Normalize font family name
        family_lower = font_family.lower().strip()

        # Try exact match first, then fallback to sans-serif
        font_paths = self.FONT_PATHS.get(family_lower, self.FONT_PATHS["sans-serif"])

        for path in font_paths:
            try:
                return ImageFont.truetype(path, font_size)
            except (OSError, IOError):
                continue

        # Final fallback to any available DejaVu font
        fallbacks = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
        for path in fallbacks:
            try:
                return ImageFont.truetype(path, font_size)
            except (OSError, IOError):
                continue

        # Last resort - default font (will be small)
        return ImageFont.load_default()

    def _render_text(self, ctx: "RenderContext", text: TextElement):
        """Render text element."""
        if not text.text:
            return

        transform = ctx.base_transform.multiply(text.transform)
        x, y = transform.apply(text.x, text.y)

        fill = self._get_fill_color(ctx, text.style)
        if not fill:
            fill = (0, 0, 0, 255)

        draw = ImageDraw.Draw(ctx.image, "RGBA")

        # Calculate font size with transform scaling
        font_size = max(1, int(text.font_size * self._get_scale(ctx, text.transform)))
        font = self._get_font(text.font_family, font_size)

        # Map SVG text-anchor to PIL anchor
        # SVG: start=left, middle=center, end=right
        # PIL anchor: first char is horizontal (l/m/r), second is vertical (a/m/s/d/b)
        # "ls" = left baseline, "ms" = middle baseline, "rs" = right baseline
        anchor_map = {"start": "ls", "middle": "ms", "end": "rs"}
        anchor = anchor_map.get(text.text_anchor, "ls")

        draw.text((x, y), text.text, fill=fill, font=font, anchor=anchor)

    def _get_scale(self, ctx: "RenderContext", element_transform: Transform) -> float:
        """Get the effective scale factor."""
        full = ctx.base_transform.multiply(element_transform)
        # Approximate scale from matrix
        return math.sqrt(abs(full.a * full.d - full.b * full.c))

    def _compute_bbox(self, points: list[tuple[float, float]]
                      ) -> tuple[float, float, float, float]:
        """Compute bounding box of points."""
        if not points:
            return (0, 0, 0, 0)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _compute_path_bbox(self, commands: list[tuple]
                           ) -> tuple[float, float, float, float]:
        """Compute bounding box from path commands."""
        points = []
        for cmd in commands:
            if cmd[0] == 'M':
                points.append((cmd[1], cmd[2]))
            elif cmd[0] == 'L':
                points.append((cmd[1], cmd[2]))
            elif cmd[0] == 'C':
                points.append((cmd[5], cmd[6]))
            elif cmd[0] == 'Q':
                points.append((cmd[3], cmd[4]))

        return self._compute_bbox(points)


class RenderContext:
    """Context for rendering operations."""

    def __init__(self, image: Image.Image,
                 gradients: dict[str, Union[LinearGradient, RadialGradient]],
                 base_transform: Transform,
                 clip_paths: dict[str, ClipPath] = None,
                 filters: dict = None):
        self.image = image
        self.gradients = gradients
        self.base_transform = base_transform
        self.clip_paths = clip_paths or {}
        self.filters = filters or {}
