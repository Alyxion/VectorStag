"""SVG Renderer - Render parsed SVG to PIL Images."""

import math
from typing import Optional, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .parser import (
    SVGParser, SVGDocument, SVGElement, Transform, Style,
    RectElement, CircleElement, EllipseElement, LineElement,
    PolylineElement, PolygonElement, PathElement, GroupElement,
    TextElement, LinearGradient, RadialGradient, GradientStop,
    FILL_NOT_SET
)


class SVGRenderer:
    """Render SVG documents to PIL Images."""

    def __init__(self, scale: float = 1.0, background: Optional[tuple[int, int, int, int]] = None):
        """
        Initialize renderer.

        Args:
            scale: Scale factor for rendering
            background: Background color (RGBA). Default is white.
        """
        self.scale = scale
        self.background = background or (255, 255, 255, 255)
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

        # Create image
        image = Image.new("RGBA", (out_width, out_height), self.background)

        # Calculate scaling transform
        scale_x = out_width / src_w if src_w else 1
        scale_y = out_height / src_h if src_h else 1
        scale = min(scale_x, scale_y)

        # Center the content
        offset_x = (out_width - src_w * scale) / 2 - src_x * scale
        offset_y = (out_height - src_h * scale) / 2 - src_y * scale

        transform = Transform.translate(offset_x, offset_y).multiply(
            Transform.scale(scale)
        )

        # Create render context
        ctx = RenderContext(image, doc.gradients, transform)

        # Render all elements
        for element in doc.elements:
            self._render_element(ctx, element)

        return image

    def _render_element(self, ctx: "RenderContext", element: SVGElement):
        """Render a single element."""
        if isinstance(element, GroupElement):
            for child in element.children:
                self._render_element(ctx, child)
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
                                          (x, y, w, h))
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

        # Generate circle points
        n_points = max(32, int(circle.r * 2))
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

        # Generate ellipse points
        n_points = max(32, int(max(ellipse.rx, ellipse.ry) * 2))
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

        stroke = self._get_stroke_color(ctx, line.style)
        if stroke:
            draw = ImageDraw.Draw(ctx.image, "RGBA")
            width = max(1, int(line.style.stroke_width * self._get_scale(ctx, line.transform)))
            draw.line(points, fill=stroke, width=width)

    def _render_polyline(self, ctx: "RenderContext", polyline: PolylineElement):
        """Render a polyline."""
        if len(polyline.points) < 2:
            return

        points = self._transform_points(ctx, polyline.transform, polyline.points)

        # Stroke only (no fill for polyline)
        stroke = self._get_stroke_color(ctx, polyline.style)
        if stroke:
            draw = ImageDraw.Draw(ctx.image, "RGBA")
            width = max(1, int(polyline.style.stroke_width * self._get_scale(ctx, polyline.transform)))
            draw.line(points, fill=stroke, width=width)

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

        # Render each polygon
        for polygon_points in polygons:
            if len(polygon_points) < 2:
                continue

            points = self._transform_points(ctx, path.transform, polygon_points)

            # Fill
            fill = self._get_fill_color(ctx, path.style)
            fill_ref = path.style.fill if isinstance(path.style.fill, str) else None

            if fill and len(points) >= 3:
                self._fill_polygon_with_gradient_check(
                    ctx, points, path.style, path.transform, bbox, fill, fill_ref
                )
            elif fill_ref and fill_ref.startswith("url(") and len(points) >= 3:
                self._fill_polygon_with_gradient_check(
                    ctx, points, path.style, path.transform, bbox, None, fill_ref
                )

            # Stroke
            stroke = self._get_stroke_color(ctx, path.style)
            if stroke and len(points) >= 2:
                draw = ImageDraw.Draw(ctx.image, "RGBA")
                width = max(1, int(path.style.stroke_width * self._get_scale(ctx, path.transform)))
                # Draw as lines
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i + 1]], fill=stroke, width=width)

    def _path_to_polygons(self, commands: list[tuple]) -> list[list[tuple[float, float]]]:
        """Convert path commands to a list of polygons."""
        polygons = []
        current_polygon = []
        current_x, current_y = 0.0, 0.0

        for cmd in commands:
            cmd_type = cmd[0]

            if cmd_type == 'M':
                if current_polygon:
                    polygons.append(current_polygon)
                current_polygon = [(cmd[1], cmd[2])]
                current_x, current_y = cmd[1], cmd[2]

            elif cmd_type == 'L':
                current_polygon.append((cmd[1], cmd[2]))
                current_x, current_y = cmd[1], cmd[2]

            elif cmd_type == 'C':
                # Cubic bezier - sample it
                x1, y1, x2, y2, x, y = cmd[1:]
                bezier_points = self._sample_cubic_bezier(
                    current_x, current_y, x1, y1, x2, y2, x, y
                )
                current_polygon.extend(bezier_points)
                current_x, current_y = x, y

            elif cmd_type == 'Q':
                # Quadratic bezier - sample it
                x1, y1, x, y = cmd[1:]
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
                    current_x, current_y = current_polygon[0]
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
        points = []
        for i in range(1, n_samples + 1):
            t = i / n_samples
            mt = 1 - t

            x = mt * mt * x0 + 2 * mt * t * x1 + t * t * x2
            y = mt * mt * y0 + 2 * mt * t * y1 + t * t * y2
            points.append((x, y))

        return points

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

        # Stroke
        stroke = self._get_stroke_color(ctx, style)
        if stroke and len(points) >= 2:
            draw = ImageDraw.Draw(ctx.image, "RGBA")
            width = max(1, int(style.stroke_width * self._get_scale(ctx, element_transform)))

            # Close the polygon for stroke
            closed_points = points + [points[0]]
            for i in range(len(closed_points) - 1):
                draw.line([closed_points[i], closed_points[i + 1]], fill=stroke, width=width)

    def _fill_polygon_with_gradient_check(self, ctx: "RenderContext",
                                          points: list[tuple[float, float]],
                                          style: Style,
                                          element_transform: Transform,
                                          bbox: tuple[float, float, float, float],
                                          fill: Optional[tuple[int, int, int, int]],
                                          fill_ref: Optional[str]):
        """Fill a polygon, handling gradients if needed."""
        if fill_ref and fill_ref.startswith("url("):
            # Extract gradient ID
            match = fill_ref[4:-1]  # Remove "url(" and ")"
            if match.startswith("#"):
                match = match[1:]

            if match in ctx.gradients:
                gradient = ctx.gradients[match]
                self._fill_polygon_with_gradient(ctx, points, gradient, bbox, style.opacity)
                return

        # Simple fill
        if fill and len(points) >= 3:
            draw = ImageDraw.Draw(ctx.image, "RGBA")
            draw.polygon(points, fill=fill)

    def _fill_polygon_with_gradient(self, ctx: "RenderContext",
                                    points: list[tuple[float, float]],
                                    gradient: Union[LinearGradient, RadialGradient],
                                    bbox: tuple[float, float, float, float],
                                    opacity: float):
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
                ctx, gradient, grad_width, grad_height, bbox, min_x, min_y, opacity
            )
        else:
            grad_img = self._create_radial_gradient_image(
                ctx, gradient, grad_width, grad_height, bbox, min_x, min_y, opacity
            )

        # Create mask from polygon
        mask = Image.new("L", (ctx.image.width, ctx.image.height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(points, fill=255)

        # Crop mask to gradient bounds
        mask_crop = mask.crop((min_x, min_y, max_x, max_y))

        # Apply gradient with mask
        ctx.image.paste(grad_img, (min_x, min_y), mask_crop)

    def _create_linear_gradient_image(self, ctx: "RenderContext",
                                      gradient: LinearGradient,
                                      width: int, height: int,
                                      bbox: tuple[float, float, float, float],
                                      offset_x: int, offset_y: int,
                                      opacity: float) -> Image.Image:
        """Create an image filled with a linear gradient."""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        if not gradient.stops:
            return img

        # Get gradient vector
        if gradient.units == "objectBoundingBox":
            bx, by, bw, bh = bbox
            # For objectBoundingBox, use the transformed bbox
            x1 = bx + gradient.x1 * bw
            y1 = by + gradient.y1 * bh
            x2 = bx + gradient.x2 * bw
            y2 = by + gradient.y2 * bh
            # Apply base transform to get screen coords
            x1, y1 = ctx.base_transform.apply(x1, y1)
            x2, y2 = ctx.base_transform.apply(x2, y2)
        else:
            # For userSpaceOnUse, coordinates are in SVG space, transform them
            x1, y1 = ctx.base_transform.apply(gradient.x1, gradient.y1)
            x2, y2 = ctx.base_transform.apply(gradient.x2, gradient.y2)

        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length == 0:
            return img

        dx /= length
        dy /= length

        # Create gradient
        pixels = np.zeros((height, width, 4), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Screen coordinates
                wx = x + offset_x
                wy = y + offset_y

                # Project onto gradient line
                t = ((wx - x1) * dx + (wy - y1) * dy) / length
                t = max(0, min(1, t))

                # Get color at t
                color = self._interpolate_gradient_color(gradient.stops, t)
                pixels[y, x] = [color[0], color[1], color[2],
                                int(color[3] * opacity)]

        return Image.fromarray(pixels, "RGBA")

    def _create_radial_gradient_image(self, ctx: "RenderContext",
                                      gradient: RadialGradient,
                                      width: int, height: int,
                                      bbox: tuple[float, float, float, float],
                                      offset_x: int, offset_y: int,
                                      opacity: float) -> Image.Image:
        """Create an image filled with a radial gradient."""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        if not gradient.stops:
            return img

        # Get gradient parameters and transform to screen coords
        if gradient.units == "objectBoundingBox":
            bx, by, bw, bh = bbox
            cx = bx + gradient.cx * bw
            cy = by + gradient.cy * bh
            r = gradient.r * max(bw, bh)
            fx = bx + (gradient.fx if gradient.fx is not None else gradient.cx) * bw
            fy = by + (gradient.fy if gradient.fy is not None else gradient.cy) * bh
            # Transform to screen coords
            cx, cy = ctx.base_transform.apply(cx, cy)
            fx, fy = ctx.base_transform.apply(fx, fy)
            # Scale radius
            scale = self._get_scale(ctx, Transform.identity())
            r = r * scale
        else:
            # For userSpaceOnUse, transform coordinates
            cx, cy = ctx.base_transform.apply(gradient.cx, gradient.cy)
            r = gradient.r * self._get_scale(ctx, Transform.identity())
            fx = gradient.fx if gradient.fx is not None else gradient.cx
            fy = gradient.fy if gradient.fy is not None else gradient.cy
            fx, fy = ctx.base_transform.apply(fx, fy)

        if r == 0:
            return img

        # Create gradient
        pixels = np.zeros((height, width, 4), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                wx = x + offset_x
                wy = y + offset_y

                # Distance from center
                d = math.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
                t = d / r
                t = max(0, min(1, t))

                color = self._interpolate_gradient_color(gradient.stops, t)
                pixels[y, x] = [color[0], color[1], color[2],
                                int(color[3] * opacity)]

        return Image.fromarray(pixels, "RGBA")

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

        # Try to use a font
        font_size = int(text.font_size * self._get_scale(ctx, text.transform))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        draw.text((x, y), text.text, fill=fill, font=font, anchor="ls")

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
                 base_transform: Transform):
        self.image = image
        self.gradients = gradients
        self.base_transform = base_transform
