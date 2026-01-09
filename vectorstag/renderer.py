"""SVG Renderer - Render parsed SVG to PIL Images."""

import math
from typing import Optional, Union, List, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter
import numpy as np

# Rust extension is required for performance
import vectorstag_rust

from .svg_parser import SVGParser
from .core.transforms import Transform
from .parser.elements import (
    SVGDocument, SVGElement, RectElement, CircleElement, EllipseElement,
    LineElement, PolylineElement, PolygonElement, PathElement, GroupElement,
    TextElement, ImageElement, ClipPath, Mask,
)
from .parser.styles import Style, FILL_NOT_SET
from .parser.gradients import LinearGradient, RadialGradient, GradientStop, Pattern
from .parser.filters import (
    Filter, FilterPrimitive,
    FeGaussianBlur, FeOffset, FeFlood, FeBlend, FeComposite, FeMerge, FeMergeNode,
    FeColorMatrix, FeComponentTransfer, FeMorphology, FeConvolveMatrix,
    FeTurbulence, FeDisplacementMap, FeImage, FeTile,
    FeDiffuseLighting, FeSpecularLighting, FeDropShadow,
    FeDistantLight, FePointLight, FeSpotLight,
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
        self._image_registry: dict[str, Image.Image] = {}

    def register_image(self, name: str, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Register an in-memory image for use in SVG.

        Args:
            name: Unique name for the image
            image: PIL Image or numpy array (RGBA)

        Returns:
            Reference string to use in SVG href attribute ('memory:name')

        Example:
            >>> renderer = SVGRenderer()
            >>> renderer.register_image("photo", pil_image)
            >>> # In SVG: <image href="memory:photo" width="100" height="100"/>
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        self._image_registry[name] = image
        return f"memory:{name}"

    def clear_images(self) -> None:
        """Clear all registered images from the registry."""
        self._image_registry.clear()

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

        # Create numpy array directly (skip PIL Image.new for Rust path)
        # Create numpy array with background color directly
        image_arr = np.zeros((render_height, render_width, 4), dtype=np.uint8)
        if self.background != (0, 0, 0, 0):
            image_arr[:, :] = self.background
        image = None  # We'll create PIL image at the end if needed
        # Create render context with numpy array as primary (if Rust available)
        ctx = RenderContext(image, doc.gradients, transform, doc.clip_paths, doc.masks, doc.filters, doc.patterns, doc.elements_by_id, doc.path_data_by_id, viewport_width=src_w, viewport_height=src_h)
        ctx.image_arr = image_arr

        # Render all elements
        for element in doc.elements:
            self._render_element(ctx, element)

        # Get the rendered array (either from ctx or convert from PIL)
        if ctx.image_arr is not None:
            img_arr = ctx.image_arr
        else:
            img_arr = np.array(image)

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

            # Only apply clipping if we have a valid rectangle that's smaller than full size
            needs_clip = (clip_x2 > clip_x1 and clip_y2 > clip_y1 and
                         (clip_x1 > 0 or clip_y1 > 0 or
                          clip_x2 < render_width or clip_y2 < render_height))

            if needs_clip:
                # Set letterbox areas to background color
                img_arr[:clip_y1, :] = self.background  # Top
                img_arr[clip_y2:, :] = self.background  # Bottom
                img_arr[:, :clip_x1] = self.background  # Left
                img_arr[:, clip_x2:] = self.background  # Right

        # Downscale for anti-aliasing effect
        if aa > 1:
            # Use Rust box filter resize (faster for 4x downscale)
            if not img_arr.flags['C_CONTIGUOUS']:
                img_arr = np.ascontiguousarray(img_arr)
            resized_arr = vectorstag_rust.resize_rgba(img_arr, out_width, out_height)
            return Image.fromarray(resized_arr, "RGBA")
        # No resize needed - convert to PIL at the end
        return Image.fromarray(img_arr, "RGBA")

    def _render_element(self, ctx: "RenderContext", element: SVGElement, depth: int = 0):
        """Render a single element."""
        # Prevent infinite recursion
        if depth > self.MAX_RENDER_DEPTH:
            return

        # Skip elements with display: none
        if element.style.display == "none":
            return

        # Skip elements with visibility: hidden (but still process children)
        # For non-group elements, visibility:hidden means don't render
        if element.style.visibility == "hidden" and not isinstance(element, GroupElement):
            return

        # Check if element has a clip path
        if element.clip_path_id and element.clip_path_id in ctx.clip_paths:
            self._render_element_with_clip(ctx, element, depth)
            return

        # Check if element has a mask
        if element.mask_id and element.mask_id in ctx.masks:
            self._render_element_with_mask(ctx, element, depth)
            return

        # Apply filter if present
        if element.style.filter_id:
            if element.style.filter_id in ctx.filters:
                self._render_element_with_filter(ctx, element, depth)
            # If filter reference is invalid (doesn't exist), element is not rendered
            return

        if isinstance(element, GroupElement):
            # Handle group opacity: render to temp buffer if opacity < 1
            group_opacity = element.style.opacity
            if group_opacity < 1.0:
                # Render children to temporary context
                temp_ctx = ctx.create_child_context()
                for child in element.children:
                    self._render_element(temp_ctx, child, depth + 1)
                # Composite temp context onto main context with group opacity
                if temp_ctx.image_arr is not None:
                    # Using numpy arrays - multiply alpha by group opacity
                    temp_arr = temp_ctx.image_arr.copy()
                    temp_arr[:, :, 3] = (temp_arr[:, :, 3] * group_opacity).astype(np.uint8)
                    vectorstag_rust.alpha_composite_inplace(ctx.image_arr, temp_arr, 0, 0)
                elif temp_ctx.image is not None:
                    # Using PIL images
                    temp_img = temp_ctx.image
                    if temp_img.mode == 'RGBA':
                        r, g, b, a = temp_img.split()
                        a = a.point(lambda x: int(x * group_opacity))
                        temp_img = Image.merge('RGBA', (r, g, b, a))
                    ctx.image = Image.alpha_composite(ctx.image, temp_img)
            else:
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
        elif isinstance(element, ImageElement):
            self._render_image(ctx, element)

    def _render_element_with_clip(self, ctx: "RenderContext", element: SVGElement, depth: int = 0):
        """Render an element with a clip path applied."""
        clip_path = ctx.clip_paths[element.clip_path_id]

        # Get element bounding box for objectBoundingBox units (needed for cache key)
        elem_bbox = self._get_element_bbox(element, ctx.base_transform)

        # Create cache key: (clip_path_id, transform_tuple, units, bbox_tuple)
        t = element.transform
        transform_tuple = (t.a, t.b, t.c, t.d, t.e, t.f) if t else (1, 0, 0, 1, 0, 0)
        if clip_path.units == "objectBoundingBox" and elem_bbox:
            cache_key = (element.clip_path_id, transform_tuple, "obb", elem_bbox)
        else:
            cache_key = (element.clip_path_id, transform_tuple, "usu", None)

        # Check cache for pre-computed mask
        if cache_key in ctx.clip_mask_cache:
            mask_arr = ctx.clip_mask_cache[cache_key]
        else:
            # Create clip mask and cache it
            mask = self._create_clip_mask(ctx, clip_path, element.transform, elem_bbox)
            mask_arr = np.array(mask)  # PIL arrays are already C-contiguous
            ctx.clip_mask_cache[cache_key] = mask_arr

        # Create a temporary image for the element
        temp_image = Image.new("RGBA", ctx.image_size, (0, 0, 0, 0))
        temp_ctx = RenderContext(temp_image, ctx.gradients, ctx.base_transform, ctx.clip_paths, ctx.masks, ctx.filters, ctx.patterns, ctx.elements_by_id, ctx.path_elements)

        # Temporarily remove clip path to render to temp image
        old_clip_path_id = element.clip_path_id
        element.clip_path_id = None

        # Render the element (which will handle filter if present)
        self._render_element(temp_ctx, element, depth + 1)

        # Restore clip path
        element.clip_path_id = old_clip_path_id

        # Apply the mask and composite onto main image
        if ctx.image_arr is not None:
            # Use Rust function for combined mask + composite (fast path)
            temp_arr = np.array(temp_image)  # PIL arrays are already C-contiguous
            vectorstag_rust.apply_mask_and_composite(ctx.image_arr, temp_arr, mask_arr, 0, 0)
        else:
            # Fallback: PIL-based mask application
            mask = Image.fromarray(mask_arr)
            temp_image.putalpha(ImageChops.multiply(temp_image.split()[3], mask))
            self._alpha_composite(ctx, temp_image, 0, 0)

    def _render_element_with_mask(self, ctx: "RenderContext", element: SVGElement, depth: int = 0):
        """Render an element with an SVG mask applied."""
        mask_def = ctx.masks[element.mask_id]

        # Create a temporary image for the element
        temp_image = Image.new("RGBA", ctx.image_size, (0, 0, 0, 0))
        temp_ctx = RenderContext(temp_image, ctx.gradients, ctx.base_transform, ctx.clip_paths, ctx.masks, ctx.filters, ctx.patterns, ctx.elements_by_id, ctx.path_elements)

        # Temporarily remove mask to render to temp image
        old_mask_id = element.mask_id
        element.mask_id = None

        # Render the element
        self._render_element(temp_ctx, element, depth + 1)

        # Restore mask
        element.mask_id = old_mask_id

        # Calculate mask bounds in screen coordinates
        # Get element bounding box for objectBoundingBox units
        elem_bbox = self._get_element_bbox(element, ctx.base_transform)

        if mask_def.mask_units == "objectBoundingBox" and elem_bbox:
            bbox_x, bbox_y, bbox_w, bbox_h = elem_bbox
            # Mask x, y, width, height are fractions of bounding box
            mask_x = bbox_x + mask_def.x * bbox_w
            mask_y = bbox_y + mask_def.y * bbox_h
            mask_w = mask_def.width * bbox_w
            mask_h = mask_def.height * bbox_h
        else:
            # userSpaceOnUse - transform mask coordinates
            combined = ctx.base_transform.multiply(element.transform)
            p1 = combined.apply(mask_def.x, mask_def.y)
            p2 = combined.apply(mask_def.x + mask_def.width, mask_def.y + mask_def.height)
            mask_x, mask_y = min(p1[0], p2[0]), min(p1[1], p2[1])
            mask_w, mask_h = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])

        # Render mask content to a temporary image
        mask_image = Image.new("RGBA", ctx.image_size, (0, 0, 0, 0))

        # Set up transform for mask content
        if mask_def.mask_content_units == "objectBoundingBox" and elem_bbox:
            # Scale content to bounding box
            bbox_x, bbox_y, bbox_w, bbox_h = elem_bbox
            mask_transform = ctx.base_transform.multiply(
                Transform.translate(bbox_x, bbox_y).multiply(
                    Transform.scale(bbox_w, bbox_h)
                )
            )
        else:
            # userSpaceOnUse - use base transform
            mask_transform = ctx.base_transform

        mask_ctx = RenderContext(mask_image, ctx.gradients, mask_transform, ctx.clip_paths, ctx.masks, ctx.filters, ctx.patterns, ctx.elements_by_id, ctx.path_elements)

        # Render mask elements
        for mask_elem in mask_def.elements:
            self._render_element(mask_ctx, mask_elem, depth + 1)

        # Convert mask to luminance
        # SVG spec: mask luminance = 0.2126*R + 0.7152*G + 0.0722*B, multiplied by alpha
        mask_arr = np.array(mask_image, dtype=np.float32)
        luminance = (0.2126 * mask_arr[:, :, 0] +
                     0.7152 * mask_arr[:, :, 1] +
                     0.0722 * mask_arr[:, :, 2])
        # Apply mask's own alpha
        luminance = luminance * (mask_arr[:, :, 3] / 255.0)
        luminance = np.clip(luminance, 0, 255).astype(np.uint8)

        mask_lum = Image.fromarray(luminance, mode="L")

        # Apply the luminance mask to element's alpha channel
        temp_image.putalpha(ImageChops.multiply(temp_image.split()[3], mask_lum))
        self._alpha_composite(ctx, temp_image, 0, 0)

    def _render_element_with_filter(self, ctx: "RenderContext", element: SVGElement, depth: int = 0):
        """Render an element with SVG filter primitives applied."""
        filter_def = ctx.filters[element.style.filter_id]

        # If filter has no primitives, output is transparent (no rendering)
        if not filter_def.primitives:
            return

        # Get element bounding box in screen coordinates
        elem_bbox = self._get_element_bbox(element, ctx.base_transform)

        # Get viewport dimensions for percentage calculations
        vp_w = getattr(ctx, 'viewport_width', None) or ctx.image_width
        vp_h = getattr(ctx, 'viewport_height', None) or ctx.image_height

        # Calculate filter region coordinates based on filterUnits
        if filter_def.filter_units == "userSpaceOnUse":
            # For userSpaceOnUse, percentages are relative to viewport
            fx = filter_def.x * vp_w if getattr(filter_def, 'x_pct', False) else filter_def.x
            fy = filter_def.y * vp_h if getattr(filter_def, 'y_pct', False) else filter_def.y
            fw = filter_def.width * vp_w if getattr(filter_def, 'width_pct', False) else filter_def.width
            fh = filter_def.height * vp_h if getattr(filter_def, 'height_pct', False) else filter_def.height

            # Transform to screen coordinates
            fx1, fy1 = ctx.base_transform.apply(fx, fy)
            fx2, fy2 = ctx.base_transform.apply(fx + fw, fy + fh)
            filter_bbox = (fx1, fy1, fx2 - fx1, fy2 - fy1)
        else:
            # objectBoundingBox - percentages relative to element bbox
            filter_bbox = None

        # If element has no content (empty bbox), use filter region if available
        if elem_bbox is None or (elem_bbox[2] <= 0 and elem_bbox[3] <= 0):
            if filter_bbox is not None and filter_bbox[2] > 0 and filter_bbox[3] > 0:
                elem_bbox = filter_bbox
            else:
                return

        # Calculate filter region
        combined = ctx.base_transform.multiply(element.transform)
        scale = math.sqrt(abs(combined.a * combined.d - combined.b * combined.c))

        # Calculate padding for filters that expand (blur, morphology, etc.)
        max_padding = 50  # Default padding
        for prim in filter_def.primitives:
            if isinstance(prim, FeGaussianBlur):
                max_padding = max(max_padding, int((prim.std_deviation_x + prim.std_deviation_y) * scale * 3) + 5)
            elif isinstance(prim, FeMorphology):
                max_padding = max(max_padding, int((prim.radius_x + prim.radius_y) * scale) + 5)
            elif isinstance(prim, FeDropShadow):
                max_padding = max(max_padding, int((prim.std_deviation_x + prim.std_deviation_y) * scale * 3 + abs(prim.dx) + abs(prim.dy)) + 5)

        # Determine filter region - use filter_bbox for userSpaceOnUse, elem_bbox for objectBoundingBox
        if filter_def.filter_units == "userSpaceOnUse" and filter_bbox is not None:
            # Use the explicit filter region
            ex, ey, ew, eh = filter_bbox
        else:
            # Use element bbox (possibly expanded by filter percentages for objectBoundingBox)
            ex, ey, ew, eh = elem_bbox
            # Apply filter region as percentages of element bbox
            ex = ex + filter_def.x * ew
            ey = ey + filter_def.y * eh
            ew = filter_def.width * ew
            eh = filter_def.height * eh

        region_x = max(0, int(ex) - max_padding)
        region_y = max(0, int(ey) - max_padding)
        region_x2 = min(ctx.image_width, int(ex + ew) + max_padding)
        region_y2 = min(ctx.image_height, int(ey + eh) + max_padding)
        region_w = region_x2 - region_x
        region_h = region_y2 - region_y
        use_region = region_w * region_h < ctx.image_width * ctx.image_height * 0.5

        if use_region and region_w > 0 and region_h > 0:
            temp_image = Image.new("RGBA", (region_w, region_h), (0, 0, 0, 0))
            offset_transform = Transform(1, 0, 0, 1, -region_x, -region_y)
            adjusted_base = offset_transform.multiply(ctx.base_transform)
            temp_ctx = RenderContext(temp_image, ctx.gradients, adjusted_base, ctx.clip_paths, ctx.masks, ctx.filters, ctx.patterns, ctx.elements_by_id, ctx.path_elements)
        else:
            region_x, region_y = 0, 0
            region_w, region_h = ctx.image_width, ctx.image_height
            temp_image = Image.new("RGBA", ctx.image_size, (0, 0, 0, 0))
            temp_ctx = RenderContext(temp_image, ctx.gradients, ctx.base_transform, ctx.clip_paths, ctx.masks, ctx.filters, ctx.patterns, ctx.elements_by_id, ctx.path_elements)

        # Render the element without filter to get SourceGraphic
        # Per SVG spec, SourceGraphic should NOT include element opacity
        # Opacity is applied AFTER the filter
        old_filter_id = element.style.filter_id
        old_opacity = element.style.opacity
        element.style.filter_id = None
        element.style.opacity = 1.0

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

        element.style.filter_id = old_filter_id
        element.style.opacity = old_opacity

        # Execute filter chain (pass temp_ctx for feImage element references)
        source_graphic = np.array(temp_image, dtype=np.uint8)
        result = self._execute_filter_chain_with_merge(filter_def, source_graphic, region_w, region_h, scale, temp_ctx)

        temp_image = Image.fromarray(result, mode='RGBA')

        # Apply filter region clipping if needed
        if elem_bbox:
            ebx, eby, ebw, ebh = elem_bbox
            if filter_def.filter_units == "objectBoundingBox":
                fx = ebx + filter_def.x * ebw
                fy = eby + filter_def.y * ebh
                fw = filter_def.width * ebw
                fh = filter_def.height * ebh
            else:
                # userSpaceOnUse - convert percentages to viewport coordinates
                user_x = filter_def.x * vp_w if getattr(filter_def, 'x_pct', False) else filter_def.x
                user_y = filter_def.y * vp_h if getattr(filter_def, 'y_pct', False) else filter_def.y
                user_w = filter_def.width * vp_w if getattr(filter_def, 'width_pct', False) else filter_def.width
                user_h = filter_def.height * vp_h if getattr(filter_def, 'height_pct', False) else filter_def.height
                fx, fy = ctx.base_transform.apply(user_x, user_y)
                fx2, fy2 = ctx.base_transform.apply(user_x + user_w, user_y + user_h)
                fw, fh = fx2 - fx, fy2 - fy

            filter_mask = Image.new("L", temp_image.size, 0)
            filter_draw = ImageDraw.Draw(filter_mask)
            filter_draw.rectangle([int(fx - region_x), int(fy - region_y),
                                   int(fx + fw - region_x), int(fy + fh - region_y)], fill=255)
            temp_image.putalpha(ImageChops.multiply(temp_image.split()[3], filter_mask))

        # Apply element opacity to the filter result (per SVG spec, opacity is applied after filter)
        if old_opacity < 1.0:
            alpha = temp_image.split()[3]
            # Multiply alpha by opacity
            alpha = alpha.point(lambda x: int(x * old_opacity))
            temp_image.putalpha(alpha)

        self._alpha_composite(ctx, temp_image, region_x, region_y)

    def _execute_filter_chain(self, filter_def: Filter, source_graphic: np.ndarray,
                               width: int, height: int, scale: float) -> np.ndarray:
        """Execute the filter primitive chain using Rust acceleration."""
        # Named buffers for filter chain
        buffers = {
            "SourceGraphic": source_graphic,
            "SourceAlpha": self._get_source_alpha(source_graphic),
        }
        last_result = source_graphic

        for prim in filter_def.primitives:
            # If input1 is None, use last_result; otherwise look up in buffers
            if prim.input1 is None:
                in1 = last_result
            else:
                in1 = buffers.get(prim.input1, last_result)

            in2_name = getattr(prim, 'input2', None)
            in2 = buffers.get(in2_name, source_graphic) if in2_name else None

            # Execute the primitive
            result = self._execute_filter_primitive(prim, in1, in2, width, height, scale)

            # Store result
            if prim.result:
                buffers[prim.result] = result
            last_result = result

        return last_result

    def _get_source_alpha(self, src: np.ndarray) -> np.ndarray:
        """Get SourceAlpha - just the alpha channel as grayscale."""
        return vectorstag_rust.get_source_alpha(src)
        result = np.zeros_like(src)
        result[:, :, 3] = src[:, :, 3]
        return result

    def _apply_subregion(self, result: np.ndarray, prim: FilterPrimitive,
                          elem_bbox: tuple, scale: float) -> np.ndarray:
        """Apply primitive subregion - pixels outside are transparent."""
        # If no subregion is specified, return as-is
        if prim.x is None and prim.y is None and prim.width is None and prim.height is None:
            return result

        h, w = result.shape[:2]
        # Default subregion to full filter region
        x = prim.x if prim.x is not None else 0
        y = prim.y if prim.y is not None else 0
        sr_w = prim.width if prim.width is not None else w / scale
        sr_h = prim.height if prim.height is not None else h / scale

        # Convert to pixel coordinates (subregion is in element space)
        # For objectBoundingBox units, coordinates are relative to element bbox
        x1 = int(x * scale)
        y1 = int(y * scale)
        x2 = int((x + sr_w) * scale)
        y2 = int((y + sr_h) * scale)

        # Clamp to buffer bounds
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        # Create output with only the subregion visible
        output = np.zeros_like(result)
        if x2 > x1 and y2 > y1:
            output[y1:y2, x1:x2] = result[y1:y2, x1:x2]
        return output

    def _execute_filter_primitive(self, prim: FilterPrimitive, in1: np.ndarray,
                                    in2: Optional[np.ndarray], width: int, height: int,
                                    scale: float, render_ctx: "RenderContext" = None,
                                    elem_bbox: tuple = None, filter_def: Filter = None) -> np.ndarray:
        """Execute a single filter primitive."""
        result = None

        # Check if primitiveUnits=objectBoundingBox (length values are fractions of bbox)
        use_bbox_units = filter_def is not None and filter_def.primitive_units == "objectBoundingBox"

        if isinstance(prim, FeGaussianBlur):
            if use_bbox_units:
                # stdDeviation as fraction of bbox - use average of width/height
                std_x = prim.std_deviation_x * width
                std_y = prim.std_deviation_y * height
            else:
                std_x = prim.std_deviation_x * scale
                std_y = prim.std_deviation_y * scale
            if std_x >= 0.5 or std_y >= 0.5:
                return vectorstag_rust.fe_gaussian_blur(in1, std_x, std_y)
            return in1

        elif isinstance(prim, FeOffset):
            if use_bbox_units:
                # dx/dy as fractions of bbox
                dx = int(prim.dx * width)
                dy = int(prim.dy * height)
            else:
                dx = int(prim.dx * scale)
                dy = int(prim.dy * scale)
            return vectorstag_rust.fe_offset(in1, dx, dy)
            result = np.zeros_like(in1)
            h, w = in1.shape[:2]
            for y in range(h):
                sy = y - dy
                if sy < 0 or sy >= h: continue
                for x in range(w):
                    sx = x - dx
                    if sx < 0 or sx >= w: continue
                    result[y, x] = in1[sy, sx]
            return result

        elif isinstance(prim, FeFlood):
            return vectorstag_rust.fe_flood(width, height, *prim.flood_color)
            result = np.zeros((height, width, 4), dtype=np.uint8)
            result[:, :] = prim.flood_color
            return result

        elif isinstance(prim, FeBlend):
            if in2 is None:
                in2 = in1
            mode_map = {"normal": 0, "multiply": 1, "screen": 2, "darken": 3, "lighten": 4,
                       "overlay": 5, "color-dodge": 6, "color-burn": 7, "hard-light": 8,
                       "soft-light": 9, "difference": 10, "exclusion": 11,
                       "hue": 12, "saturation": 13, "color": 14, "luminosity": 15}
            mode = mode_map.get(prim.mode, 0)
            return vectorstag_rust.fe_blend(in1, in2, mode)

        elif isinstance(prim, FeComposite):
            if in2 is None:
                in2 = in1
            op_map = {"over": 0, "in": 1, "out": 2, "atop": 3, "xor": 4, "arithmetic": 5}
            op = op_map.get(prim.operator, 0)
            return vectorstag_rust.fe_composite(in1, in2, op, prim.k1, prim.k2, prim.k3, prim.k4)

        elif isinstance(prim, FeMerge):
            if not prim.nodes:
                return in1
            # Collect layer arrays
            layers = []
            # We need access to the buffers dict which we don't have here
            # For now, just return in1 - this will be handled in the chain executor
            return in1

        elif isinstance(prim, FeColorMatrix):
            type_map = {"matrix": 0, "saturate": 1, "hueRotate": 2, "luminanceToAlpha": 3}
            mt = type_map.get(prim.type, 0)
            # Validate saturate value - must be in [0, 1] range
            # Out of range values are undefined behavior, resvg returns transparent
            if prim.type == "saturate" and prim.values:
                sat_val = prim.values[0]
                if sat_val < 0.0 or sat_val > 1.0:
                    return np.zeros_like(in1)
            # For matrix type with invalid values (empty or wrong count), pass through source
            if prim.type == "matrix" and len(prim.values) != 20:
                return in1  # Pass through source for invalid matrix
            return vectorstag_rust.fe_color_matrix(in1, mt, prim.values)

        elif isinstance(prim, FeComponentTransfer):
            def make_func_tuple(f):
                type_map = {"identity": 0, "table": 1, "discrete": 2, "linear": 3, "gamma": 4}
                return (type_map.get(f.type, 0), f.table_values, f.slope, f.intercept,
                        f.amplitude, f.exponent, f.offset)
            return vectorstag_rust.fe_component_transfer(
                in1, make_func_tuple(prim.func_r), make_func_tuple(prim.func_g),
                make_func_tuple(prim.func_b), make_func_tuple(prim.func_a))

        elif isinstance(prim, FeMorphology):
            if use_bbox_units:
                rx = prim.radius_x * width
                ry = prim.radius_y * height
            else:
                rx = prim.radius_x * scale
                ry = prim.radius_y * scale
            h, w = in1.shape[:2]
            # For erode with huge radius (>= image dimension), result is transparent
            # because the kernel extends into transparent padding
            if prim.operator == "erode" and (rx >= w or ry >= h):
                return np.zeros_like(in1)
            op = 0 if prim.operator == "erode" else 1
            return vectorstag_rust.fe_morphology(in1, op, rx, ry)

        elif isinstance(prim, FeConvolveMatrix):
            divisor = prim.divisor if prim.divisor is not None else sum(prim.kernel_matrix) or 1.0
            # Default target to center, clamp to valid range [0, order-1]
            target_x = prim.target_x if prim.target_x is not None else prim.order_x // 2
            target_y = prim.target_y if prim.target_y is not None else prim.order_y // 2
            # Clamp to valid range to prevent errors with negative or out-of-range values
            target_x = max(0, min(target_x, prim.order_x - 1))
            target_y = max(0, min(target_y, prim.order_y - 1))
            edge_map = {"duplicate": 0, "wrap": 1, "none": 2}
            edge_mode = edge_map.get(prim.edge_mode, 0)
            return vectorstag_rust.fe_convolve_matrix(
                in1, prim.order_x, prim.order_y, prim.kernel_matrix, divisor,
                prim.bias, target_x, target_y, edge_mode, prim.preserve_alpha)

        elif isinstance(prim, FeTurbulence):
            noise_type = 0 if prim.type == "turbulence" else 1
            stitch = prim.stitch_tiles == "stitch"
            return vectorstag_rust.fe_turbulence(
                width, height, prim.base_frequency_x, prim.base_frequency_y,
                prim.num_octaves, prim.seed, noise_type, stitch)

        elif isinstance(prim, FeDisplacementMap):
            if in2 is None:
                in2 = in1
            ch_map = {"R": 0, "G": 1, "B": 2, "A": 3}
            x_ch = ch_map.get(prim.x_channel_selector, 3)
            y_ch = ch_map.get(prim.y_channel_selector, 3)
            return vectorstag_rust.fe_displacement_map(in1, in2, prim.scale * scale, x_ch, y_ch)

        elif isinstance(prim, FeTile):
            return vectorstag_rust.fe_tile(in1, width, height)

        elif isinstance(prim, FeDiffuseLighting):
            light = prim.light_source
            if light is None:
                # No light source = no lighting effect (transparent output)
                return np.zeros_like(in1)
            if isinstance(light, FeDistantLight):
                lt, az, el = 0, light.azimuth, light.elevation
                lx, ly, lz, px, py, pz, se, lca = 0, 0, 0, 0, 0, 0, 1, 180
            elif isinstance(light, FePointLight):
                lt, az, el = 1, 0, 0
                lx, ly, lz = light.x * scale, light.y * scale, light.z * scale
                px, py, pz, se, lca = 0, 0, 0, 1, 180
            elif isinstance(light, FeSpotLight):
                lt, az, el = 2, 0, 0
                lx, ly, lz = light.x * scale, light.y * scale, light.z * scale
                px, py, pz = light.points_at_x * scale, light.points_at_y * scale, light.points_at_z * scale
                se = light.specular_exponent
                lca = light.limiting_cone_angle if light.limiting_cone_angle else 180
            else:
                # Unknown light type - no effect
                return np.zeros_like(in1)
            return vectorstag_rust.fe_diffuse_lighting(
                in1, prim.surface_scale * scale, prim.diffuse_constant, prim.lighting_color,
                lt, az, el, lx, ly, lz, px, py, pz, se, lca)

        elif isinstance(prim, FeSpecularLighting):
            # specularExponent must be in [1, 128] range per SVG spec
            # Values outside this range produce no output (transparent)
            if prim.specular_exponent < 1.0 or prim.specular_exponent > 128.0:
                return np.zeros_like(in1)
            light = prim.light_source
            if light is None:
                # No light source = no lighting effect (transparent output)
                return np.zeros_like(in1)
            if isinstance(light, FeDistantLight):
                lt, az, el = 0, light.azimuth, light.elevation
                lx, ly, lz, px, py, pz, se, lca = 0, 0, 0, 0, 0, 0, 1, 180
            elif isinstance(light, FePointLight):
                lt, az, el = 1, 0, 0
                lx, ly, lz = light.x * scale, light.y * scale, light.z * scale
                px, py, pz, se, lca = 0, 0, 0, 1, 180
            elif isinstance(light, FeSpotLight):
                lt, az, el = 2, 0, 0
                lx, ly, lz = light.x * scale, light.y * scale, light.z * scale
                px, py, pz = light.points_at_x * scale, light.points_at_y * scale, light.points_at_z * scale
                se = light.specular_exponent
                lca = light.limiting_cone_angle if light.limiting_cone_angle else 180
            else:
                # Unknown light type - no effect
                return np.zeros_like(in1)
            return vectorstag_rust.fe_specular_lighting(
                in1, prim.surface_scale * scale, prim.specular_constant, prim.specular_exponent,
                prim.lighting_color, lt, az, el, lx, ly, lz, px, py, pz, se, lca)

        elif isinstance(prim, FeDropShadow):
            return vectorstag_rust.fe_drop_shadow(
                in1, prim.dx * scale, prim.dy * scale,
                prim.std_deviation_x * scale, prim.std_deviation_y * scale,
                *prim.flood_color)

        elif isinstance(prim, FeImage):
            # Load image from href (data URL or element reference)
            href = prim.href.strip()
            if href.startswith('data:'):
                # Data URL - parse and decode
                try:
                    import base64
                    from io import BytesIO

                    # Parse data URL format: data:[<mediatype>][;base64],<data>
                    header, data = href.split(',', 1)
                    if ';base64' in header:
                        img_data = base64.b64decode(data)
                        img = Image.open(BytesIO(img_data)).convert('RGBA')

                        # Resize to fit filter region if needed
                        target_w, target_h = width, height
                        if img.size != (target_w, target_h):
                            # Apply preserveAspectRatio
                            if 'none' in prim.preserveAspectRatio.lower():
                                img = img.resize((target_w, target_h), Image.LANCZOS)
                            else:
                                # Preserve aspect ratio (xMidYMid by default)
                                img_ratio = img.width / img.height
                                target_ratio = target_w / target_h
                                if img_ratio > target_ratio:
                                    new_w = target_w
                                    new_h = int(target_w / img_ratio)
                                else:
                                    new_h = target_h
                                    new_w = int(target_h * img_ratio)
                                img = img.resize((new_w, new_h), Image.LANCZOS)

                                # Center the image
                                result = np.zeros((target_h, target_w, 4), dtype=np.uint8)
                                x_off = (target_w - new_w) // 2
                                y_off = (target_h - new_h) // 2
                                img_arr = np.array(img)
                                result[y_off:y_off+new_h, x_off:x_off+new_w] = img_arr
                                return result

                        return np.array(img)
                except Exception:
                    pass
            elif href.startswith('#') and render_ctx is not None:
                # Element reference - render the referenced element
                elem_id = href[1:]  # Remove the # prefix
                if elem_id in render_ctx.elements_by_id:
                    # Check for recursive reference - if we're rendering for feImage,
                    # don't allow the referenced element to use the same filter
                    ref_elem = render_ctx.elements_by_id[elem_id]

                    # Track which elements are being rendered for feImage to prevent loops
                    if not hasattr(self, '_feimage_rendering'):
                        self._feimage_rendering = set()

                    if elem_id in self._feimage_rendering:
                        # Self-recursive reference - return transparent
                        return np.zeros((height, width, 4), dtype=np.uint8)

                    self._feimage_rendering.add(elem_id)
                    try:
                        # Create a temporary image for rendering the referenced element
                        temp_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                        temp_ctx = RenderContext(temp_image, render_ctx.gradients,
                                                render_ctx.base_transform, render_ctx.clip_paths,
                                                render_ctx.masks, render_ctx.filters,
                                                render_ctx.patterns, render_ctx.elements_by_id,
                                                render_ctx.path_elements)

                        # Temporarily disable filter on the element to prevent recursion
                        old_filter_id = None
                        if hasattr(ref_elem, 'style') and hasattr(ref_elem.style, 'filter_id'):
                            old_filter_id = ref_elem.style.filter_id
                            ref_elem.style.filter_id = None

                        # Render the referenced element
                        self._render_element(temp_ctx, ref_elem, depth=0)

                        # Restore filter
                        if old_filter_id is not None:
                            ref_elem.style.filter_id = old_filter_id

                        return np.array(temp_image, dtype=np.uint8)
                    except Exception:
                        pass
                    finally:
                        self._feimage_rendering.discard(elem_id)

            return in1

        return in1

    def _execute_filter_chain_with_merge(self, filter_def: Filter, source_graphic: np.ndarray,
                                          width: int, height: int, scale: float,
                                          render_ctx: "RenderContext" = None) -> np.ndarray:
        """Execute filter chain with proper feMerge support and subregion handling."""
        # Apply color space conversion if needed (default is linearRGB)
        use_linear = filter_def.color_interpolation_filters == "linearRGB"
        if use_linear:
            source_graphic = vectorstag_rust.srgb_to_linear(source_graphic)

        buffers = {
            "SourceGraphic": source_graphic,
            "SourceAlpha": self._get_source_alpha(source_graphic),
        }
        last_result = source_graphic

        for prim in filter_def.primitives:
            # If input1 is None, use last_result; otherwise look up in buffers
            if prim.input1 is None:
                in1 = last_result
            else:
                in1 = buffers.get(prim.input1, last_result)

            # Calculate primitive subregion
            subregion = self._calculate_primitive_subregion(prim, filter_def, width, height, scale)

            if isinstance(prim, FeMerge):
                # Handle merge specially - collect all node inputs
                # Empty feMerge produces transparent output
                if not prim.nodes:
                    result = np.zeros_like(source_graphic)
                else:
                    layers = []
                    for node in prim.nodes:
                        node_in = buffers.get(node.input1, last_result)
                        layers.append(node_in)
                    if layers:
                        result = vectorstag_rust.fe_merge(layers)
                    else:
                        result = np.zeros_like(source_graphic)
            else:
                in2_name = getattr(prim, 'input2', None)
                in2 = buffers.get(in2_name, source_graphic) if in2_name else None
                result = self._execute_filter_primitive_with_subregion(
                    prim, in1, in2, width, height, scale, subregion, render_ctx, filter_def=filter_def
                )

            if prim.result:
                buffers[prim.result] = result
            last_result = result

        # Convert back to sRGB if we were working in linear space
        if use_linear:
            last_result = vectorstag_rust.linear_to_srgb(last_result)

        return last_result

    def _calculate_primitive_subregion(self, prim: FilterPrimitive, filter_def: Filter,
                                        width: int, height: int, scale: float) -> tuple:
        """Calculate the subregion for a filter primitive in pixel coordinates.

        Returns (x, y, w, h, clip_x, clip_y) where:
        - x, y, w, h: the primitive's logical subregion (can be negative/outside bounds)
        - clip_x, clip_y, clip_w, clip_h: the clipped region that fits in the output
        """
        # Default subregion is the entire filter region
        x, y, w, h = 0, 0, width, height

        # Check if primitive has explicit subregion
        has_subregion = (prim.x is not None or prim.y is not None or
                         prim.width is not None or prim.height is not None)

        if has_subregion:
            if filter_def.primitive_units == "objectBoundingBox":
                # Values are fractions (0-1) of the filter region
                x = int((prim.x or 0) * width)
                y = int((prim.y or 0) * height)
                w = int((prim.width if prim.width is not None else 1.0) * width)
                h = int((prim.height if prim.height is not None else 1.0) * height)
            else:
                # userSpaceOnUse - values are in user coordinates, scale them
                x = int((prim.x or 0) * scale)
                y = int((prim.y or 0) * scale)
                w = int((prim.width if prim.width is not None else width / scale) * scale)
                h = int((prim.height if prim.height is not None else height / scale) * scale)

        # Calculate clipped region (the part that's visible in the output)
        clip_x = max(0, x)
        clip_y = max(0, y)
        clip_x2 = min(width, x + w)
        clip_y2 = min(height, y + h)
        clip_w = max(0, clip_x2 - clip_x)
        clip_h = max(0, clip_y2 - clip_y)

        # Return both the logical subregion and the clipped region
        return (x, y, w, h, clip_x, clip_y, clip_w, clip_h)

    def _execute_filter_primitive_with_subregion(self, prim: FilterPrimitive, in1: np.ndarray,
                                                   in2: Optional[np.ndarray], width: int, height: int,
                                                   scale: float, subregion: tuple,
                                                   render_ctx: "RenderContext" = None,
                                                   filter_def: Filter = None) -> np.ndarray:
        """Execute a filter primitive with subregion handling."""
        # Unpack subregion: logical (x, y, w, h) and clipped (clip_x, clip_y, clip_w, clip_h)
        log_x, log_y, log_w, log_h, clip_x, clip_y, clip_w, clip_h = subregion

        # Special handling for primitives that need subregion support
        if isinstance(prim, FeFlood):
            # Create transparent output, fill only the clipped subregion
            result = np.zeros((height, width, 4), dtype=np.uint8)
            if clip_w > 0 and clip_h > 0:
                result[clip_y:clip_y+clip_h, clip_x:clip_x+clip_w] = prim.flood_color
            return result

        elif isinstance(prim, FeTile):
            # feTile tiles the entire input buffer (including transparent areas)
            # to fill the output region. The tile is the entire input buffer.
            # Use the entire input as the tile source
            tile_src = in1
            th, tw = tile_src.shape[:2]
            if th > 0 and tw > 0:
                # If subregion is specified, tile only to that region
                if clip_w < width or clip_h < height or clip_x > 0 or clip_y > 0:
                    # Tile to fill the clipped region
                    # Account for the offset from the logical origin
                    offset_x = clip_x % tw
                    offset_y = clip_y % th

                    tile_w = clip_w + offset_x
                    tile_h = clip_h + offset_y
                    tiled = vectorstag_rust.fe_tile(tile_src, tile_w, tile_h)

                    result = np.zeros((height, width, 4), dtype=np.uint8)
                    result[clip_y:clip_y+clip_h, clip_x:clip_x+clip_w] = tiled[offset_y:offset_y+clip_h, offset_x:offset_x+clip_w]
                    return result
                else:
                    # Tile to fill the entire filter region
                    return vectorstag_rust.fe_tile(tile_src, width, height)
            return in1

        elif isinstance(prim, FeTurbulence):
            # Generate noise for the full region, then mask to subregion
            noise_type = 0 if prim.type == "turbulence" else 1
            stitch = prim.stitch_tiles == "stitch"
            full_noise = vectorstag_rust.fe_turbulence(
                width, height, prim.base_frequency_x, prim.base_frequency_y,
                prim.num_octaves, prim.seed, noise_type, stitch)
            # If subregion is specified and not full size, mask it
            if clip_w < width or clip_h < height or clip_x > 0 or clip_y > 0:
                result = np.zeros((height, width, 4), dtype=np.uint8)
                result[clip_y:clip_y+clip_h, clip_x:clip_x+clip_w] = full_noise[clip_y:clip_y+clip_h, clip_x:clip_x+clip_w]
                return result
            return full_noise
            return np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)

        # For other primitives, execute normally then apply subregion mask
        result = self._execute_filter_primitive(prim, in1, in2, width, height, scale, render_ctx, filter_def=filter_def)

        # Apply subregion mask - only show output within the subregion
        log_x, log_y, log_w, log_h, clip_x, clip_y, clip_w, clip_h = subregion
        has_subregion = (prim.x is not None or prim.y is not None or
                         prim.width is not None or prim.height is not None)
        if has_subregion and (clip_w < width or clip_h < height or clip_x > 0 or clip_y > 0):
            masked = np.zeros_like(result)
            if clip_w > 0 and clip_h > 0:
                masked[clip_y:clip_y+clip_h, clip_x:clip_x+clip_w] = result[clip_y:clip_y+clip_h, clip_x:clip_x+clip_w]
            return masked
        return result

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
        elif isinstance(element, GroupElement):
            # For groups, compute union of all children bboxes
            all_xs, all_ys = [], []
            combined = transform.multiply(element.transform)
            for child in element.children:
                child_bbox = self._get_element_bbox(child, combined)
                if child_bbox:
                    x, y, w, h = child_bbox
                    all_xs.extend([x, x + w])
                    all_ys.extend([y, y + h])
            if all_xs and all_ys:
                return (min(all_xs), min(all_ys), max(all_xs) - min(all_xs), max(all_ys) - min(all_ys))
            return None
        elif isinstance(element, PathElement):
            # For paths, compute bbox from commands
            combined = transform.multiply(element.transform)
            xs, ys = [], []
            for cmd in element.commands:
                if isinstance(cmd, tuple) and len(cmd) >= 3:
                    # Commands are tuples: ('M', x, y, ...), ('C', x, y, x1, y1, x2, y2), etc.
                    cmd_type = cmd[0]
                    if cmd_type in ('M', 'L', 'T'):
                        tx, ty = combined.apply(cmd[1], cmd[2])
                        xs.append(tx)
                        ys.append(ty)
                    elif cmd_type in ('C', 'S', 'Q'):
                        # Cubic/quadratic curves: include all control points for bbox
                        for i in range(1, len(cmd) - 1, 2):
                            if i + 1 < len(cmd):
                                tx, ty = combined.apply(cmd[i], cmd[i + 1])
                                xs.append(tx)
                                ys.append(ty)
                    elif cmd_type == 'A':
                        # Arc: at minimum include endpoint
                        if len(cmd) >= 8:
                            tx, ty = combined.apply(cmd[6], cmd[7])
                            xs.append(tx)
                            ys.append(ty)
                    elif cmd_type == 'H':
                        # Horizontal line: only x changes
                        if xs:  # Need previous y
                            tx, _ = combined.apply(cmd[1], 0)
                            xs.append(tx)
                    elif cmd_type == 'V':
                        # Vertical line: only y changes
                        if ys:  # Need previous x
                            _, ty = combined.apply(0, cmd[1])
                            ys.append(ty)
                elif hasattr(cmd, 'x'):
                    # Object-style commands (fallback)
                    tx, ty = combined.apply(cmd.x, cmd.y)
                    xs.append(tx)
                    ys.append(ty)
                    if hasattr(cmd, 'x1'):
                        tx, ty = combined.apply(cmd.x1, cmd.y1)
                        xs.append(tx)
                        ys.append(ty)
                    if hasattr(cmd, 'x2'):
                        tx, ty = combined.apply(cmd.x2, cmd.y2)
                        xs.append(tx)
                        ys.append(ty)
            if xs and ys:
                return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            return None
        elif isinstance(element, (PolygonElement, PolylineElement)):
            combined = transform.multiply(element.transform)
            if element.points:
                transformed = [combined.apply(x, y) for x, y in element.points]
                xs = [p[0] for p in transformed]
                ys = [p[1] for p in transformed]
                return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            return None
        elif isinstance(element, LineElement):
            combined = transform.multiply(element.transform)
            p1 = combined.apply(element.x1, element.y1)
            p2 = combined.apply(element.x2, element.y2)
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
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
                          element_transform: Transform, elem_bbox: tuple = None) -> Image.Image:
        """Create a mask image from a clip path."""
        mask = Image.new("L", ctx.image_size, 0)

        # Determine the transform based on clipPathUnits
        use_bbox_units = clip_path.units == "objectBoundingBox" and elem_bbox is not None

        for clip_elem in clip_path.elements:
            if use_bbox_units:
                # For objectBoundingBox, clip coordinates (0-1) should be scaled to element bbox
                bbox_x, bbox_y, bbox_w, bbox_h = elem_bbox
                # Create transform: translate to bbox origin, then scale by bbox size
                bbox_transform = Transform.translate(bbox_x, bbox_y).multiply(
                    Transform.scale(bbox_w, bbox_h)
                )
                full_transform = bbox_transform.multiply(clip_elem.transform)
            else:
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
            nested_mask = self._create_clip_mask(ctx, nested_clip, element_transform, elem_bbox)
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
                self._stroke_open_polygon(ctx, points, stroke, half_width,
                                          style.stroke_linecap)
            return

        int_width = max(1, int(width))

        # Use temp image for semi-transparent strokes or when ctx.image is None (Rust path)
        if stroke[3] < 255 or ctx.image is None:
            temp = Image.new("RGBA", ctx.image_size, (0, 0, 0, 0))
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
            self._alpha_composite(ctx, temp, 0, 0)

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
        stroke_opacity = style.stroke_opacity * style.opacity

        # Fill the stroke polygon with gradient
        self._fill_polygon_with_gradient_check(ctx, stroke_polygon, style, element_transform,
                                               screen_bbox, None, stroke_ref, opacity=stroke_opacity)

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

        # Draw the stroke polygon using fast path when available
        self._fill_polygon_with_rule(ctx, stroke_polygon, stroke, "nonzero")

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

        # Routing strategy:
        # - Round joins: ALWAYS use Rust outline approach (handles arcs correctly, no gaps)
        # - Convex shapes: Use Rust outline approach (efficient, gap-free)
        # - Non-convex shapes with miter joins: Use segmented approach to avoid self-intersection
        # - Bevel joins: Use segmented approach for correct perpendicular offsets
        if linejoin == "round":
            # Round joins work best with Rust outline approach (no gaps)
            self._stroke_closed_polygon_outline(ctx, points, stroke, half_width, miterlimit, linejoin)
        elif has_reflex or linejoin == "bevel":
            self._stroke_closed_polygon_segmented(ctx, points, stroke, half_width, miterlimit, linejoin)
        else:
            self._stroke_closed_polygon_outline(ctx, points, stroke, half_width, miterlimit, linejoin)

    def _stroke_closed_polygon_outline(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                                        stroke: Tuple[int, int, int, int], half_width: float,
                                        miterlimit: float = 4.0, linejoin: str = "miter"):
        """Render closed polygon stroke using outer/inner outline approach."""
        n = len(points)
        if n < 3:
            return

        # Get bounding box for stroke area
        # For miter joins, need extra space for miter extensions (up to miterlimit * half_width)
        # For round joins, need half_width * 1.5 for the round corners
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if linejoin == "round":
            padding = half_width * 1.5
        elif linejoin == "miter":
            padding = miterlimit * half_width + 2
        else:
            padding = half_width + 2
        min_x = max(0, int(min(xs) - padding))
        min_y = max(0, int(min(ys) - padding))
        max_x = min(ctx.image_width, int(max(xs) + padding))
        max_y = min(ctx.image_height, int(max(ys) + padding))

        if min_x >= max_x or min_y >= max_y:
            return

        width = max_x - min_x
        height = max_y - min_y

        # Use Rust implementation if available
        # Add small expansion to half_width for better sub-pixel edge coverage
        stroke_expand = 0.3
        mask = vectorstag_rust.render_stroke_closed_polygon(
            points, half_width + stroke_expand, miterlimit, width, height, min_x, min_y, linejoin
        )
        mask_img = Image.fromarray(mask, "L")

        # Round joins are now handled in Rust by drawing circles at corners

        # Apply stroke color with mask
        fill_img = Image.new("RGBA", (width, height), stroke[:3] + (255,))
        self._composite_masked_fill(ctx, fill_img, mask_img, min_x, min_y, stroke[3])
    def _stroke_closed_polygon_segmented(self, ctx: "RenderContext", points: List[Tuple[float, float]],
                                          stroke: Tuple[int, int, int, int], half_width: float,
                                          miterlimit: float = 4.0, linejoin: str = "miter"):
        """Render closed polygon stroke using segment-based approach.

        Used for non-convex shapes where the outline approach would self-intersect.
        """
        n = len(points)

        temp = Image.new("RGBA", ctx.image_size, (0, 0, 0, 0))
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

        # Fill corners with miter triangles (only for miter/bevel joins, not round)
        if linejoin != "round":
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

                # For miter calculation, extend outer_p1 forward (d1) and outer_p2 backward (-d2)
                neg_d2 = (-d2[0], -d2[1])

                if cross > 0:
                    # Left turn - outer/convex side is RIGHT (-perp)
                    miter_pt = self._line_intersection(right_p1, d1, right_p2, neg_d2)
                    if miter_pt:
                        miter_dist = math.sqrt((miter_pt[0] - p_curr[0])**2 + (miter_pt[1] - p_curr[1])**2)
                        if miter_dist <= miterlimit * half_width:
                            tri = [right_p1, miter_pt, right_p2]
                            draw.polygon(tri, fill=stroke)
                else:
                    # Right turn - outer/convex side is LEFT (+perp)
                    miter_pt = self._line_intersection(left_p1, d1, left_p2, neg_d2)
                    if miter_pt:
                        miter_dist = math.sqrt((miter_pt[0] - p_curr[0])**2 + (miter_pt[1] - p_curr[1])**2)
                        if miter_dist <= miterlimit * half_width:
                            tri = [left_p1, miter_pt, left_p2]
                            draw.polygon(tri, fill=stroke)

        # Draw round joins at corners if linejoin is "round"
        # Use pie slices (arcs) instead of full circles to avoid over-filling
        if linejoin == "round":
            for i in range(n):
                p_prev = points[(i - 1) % n]
                p_curr = points[i]
                p_next = points[(i + 1) % n]

                # Direction vectors for adjacent edges
                d1 = self._normalize(self._subtract(p_curr, p_prev))
                d2 = self._normalize(self._subtract(p_next, p_curr))

                # Cross product to determine turn direction
                cross = d1[0] * d2[1] - d1[1] * d2[0]

                # Only draw round join on outside of corner (convex turn)
                # For a closed polygon, outside corners have cross product with sign
                # matching the polygon's winding direction
                if abs(cross) < 0.001:
                    continue  # Collinear edges, no join needed

                # Perpendicular directions (pointing outward from stroke)
                perp1 = (-d1[1], d1[0])  # Perpendicular to incoming edge
                perp2 = (-d2[1], d2[0])  # Perpendicular to outgoing edge

                # Calculate angles for the arc
                # perp1/perp2 point in the "left" direction relative to edge direction
                # For the round join, we need the arc on the OUTSIDE of the corner
                # PIL angles: 0=right, 90=down, 180=left, 270=up (counterclockwise)
                angle1 = math.degrees(math.atan2(perp1[1], perp1[0]))
                angle2 = math.degrees(math.atan2(perp2[1], perp2[0]))

                x, y = p_curr
                bbox = [x - half_width, y - half_width, x + half_width, y + half_width]

                # For right turn (cross < 0), outside is where we draw the arc
                # The arc connects the outer ends of the two edge strokes
                if cross < 0:
                    # Right turn - arc goes from perp2 angle to perp1 angle (counterclockwise)
                    draw.pieslice(bbox, angle2, angle1, fill=stroke)
                else:
                    # Left turn - arc goes from perp1 angle to perp2 angle (counterclockwise)
                    draw.pieslice(bbox, angle1, angle2, fill=stroke)

        self._alpha_composite(ctx, temp, 0, 0)

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

    def _generate_corner_arc(self, center: Tuple[float, float],
                             perp1: Tuple[float, float], perp2: Tuple[float, float],
                             radius: float, sign: int) -> List[Tuple[float, float]]:
        """Generate arc points for round join between two edge directions.

        Args:
            center: Corner point (path vertex)
            perp1: Perpendicular of first edge (pointing left)
            perp2: Perpendicular of second edge (pointing left)
            radius: Stroke half-width
            sign: 1 for left side (use perp), -1 for right side (use -perp)

        Returns:
            List of arc points from edge1 endpoint to edge2 endpoint
        """
        # Start and end directions for the arc
        start_dir = (perp1[0] * sign, perp1[1] * sign)
        end_dir = (perp2[0] * sign, perp2[1] * sign)

        # Compute angle between start and end
        start_angle = math.atan2(start_dir[1], start_dir[0])
        end_angle = math.atan2(end_dir[1], end_dir[0])

        # Compute angle difference (taking shorter arc)
        angle_diff = end_angle - start_angle
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Number of arc segments based on angle
        n_segments = max(2, int(abs(angle_diff) * 6 / math.pi))

        arc_points = []
        for j in range(n_segments + 1):
            t = j / n_segments
            angle = start_angle + t * angle_diff
            px = center[0] + radius * math.cos(angle)
            py = center[1] + radius * math.sin(angle)
            arc_points.append((px, py))

        return arc_points

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
                                          fill_ref: Optional[str],
                                          opacity: Optional[float] = None):
        """Fill a polygon, handling gradients and patterns if needed.

        Args:
            opacity: Override opacity (for strokes, pass stroke_opacity * element_opacity)
        """
        fallback_fill = fill  # Keep original fill as fallback
        # Use passed opacity or default to fill_opacity * element_opacity
        if opacity is None:
            opacity = style.fill_opacity * style.opacity

        if fill_ref and fill_ref.startswith("url("):
            # Extract paint server ID - handle fallback colors like "url(#id) rgb(0,0,0)"
            end_paren = fill_ref.find(")")
            if end_paren != -1:
                match = fill_ref[4:end_paren]  # Remove "url(" and extract up to ")"
                # Check for fallback color after the url()
                fallback_str = fill_ref[end_paren + 1:].strip()
                if fallback_str:
                    fallback_fill = self._parse_fallback_color(fallback_str, style)
            else:
                match = fill_ref[4:]
            match = match.strip()
            # Strip quotes if present (SVG 2 allows quoted URLs)
            if (match.startswith("'") and match.endswith("'")) or (match.startswith('"') and match.endswith('"')):
                match = match[1:-1]
            if match.startswith("#"):
                match = match[1:]

            if match in ctx.gradients:
                gradient = ctx.gradients[match]
                # Check if gradient has stops - empty gradient should use fallback
                if gradient.stops:
                    self._fill_polygon_with_gradient(ctx, points, gradient, bbox,
                                                     opacity,
                                                     style.fill_rule,
                                                     element_transform)
                    return
                # Empty gradient - fall through to use fallback color

            if match in ctx.patterns:
                pattern = ctx.patterns[match]
                self._fill_polygon_with_pattern(ctx, points, pattern, bbox,
                                                opacity,
                                                style.fill_rule,
                                                element_transform)
                return

            # Gradient/pattern not found - use fallback color
            fill = fallback_fill

        # Simple fill with fill-rule support
        if fill and len(points) >= 3:
            self._fill_polygon_with_rule(ctx, points, fill, style.fill_rule)

    def _parse_fallback_color(self, color_str: str, style: Style) -> Optional[tuple[int, int, int, int]]:
        """Parse a fallback color string."""
        color_str = color_str.strip()
        if not color_str or color_str == "none":
            return None

        # Handle currentColor
        if color_str == "currentColor":
            # currentColor inherits from parent - use style's fill as approximation
            if isinstance(style.fill, tuple):
                return style.fill
            return (0, 0, 0, 255)  # Default black

        # Try to parse as a color
        color = self._parse_color(color_str)
        if color:
            r, g, b = color
            a = int(255 * style.fill_opacity * style.opacity)
            return (r, g, b, a)

        return None

    def _parse_color(self, color_str: str) -> Optional[tuple[int, int, int]]:
        """Parse a color string to RGB tuple."""
        color_str = color_str.strip().lower()

        # Named colors
        named_colors = {
            "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
            "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
            "cyan": (0, 255, 255), "magenta": (255, 0, 255), "gray": (128, 128, 128),
            "grey": (128, 128, 128), "lime": (0, 255, 0), "maroon": (128, 0, 0),
            "navy": (0, 0, 128), "olive": (128, 128, 0), "purple": (128, 0, 128),
            "teal": (0, 128, 128), "silver": (192, 192, 192), "fuchsia": (255, 0, 255),
            "aqua": (0, 255, 255), "orange": (255, 165, 0), "pink": (255, 192, 203),
            "brown": (165, 42, 42), "gold": (255, 215, 0), "coral": (255, 127, 80),
            "crimson": (220, 20, 60), "darkblue": (0, 0, 139), "darkgreen": (0, 100, 0),
            "darkred": (139, 0, 0), "lightblue": (173, 216, 230), "lightgreen": (144, 238, 144),
        }
        if color_str in named_colors:
            return named_colors[color_str]

        # Hex colors
        if color_str.startswith("#"):
            hex_str = color_str[1:]
            if len(hex_str) == 3:
                r = int(hex_str[0] * 2, 16)
                g = int(hex_str[1] * 2, 16)
                b = int(hex_str[2] * 2, 16)
                return (r, g, b)
            elif len(hex_str) == 6:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b)

        # rgb() and rgba()
        import re
        rgb_match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
        if rgb_match:
            return (int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3)))

        rgba_match = re.match(r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\d.]+\s*\)', color_str)
        if rgba_match:
            return (int(rgba_match.group(1)), int(rgba_match.group(2)), int(rgba_match.group(3)))

        return None

    def _fill_polygon_with_rule(self, ctx: "RenderContext",
                                points: list[tuple[float, float]],
                                fill: tuple[int, int, int, int],
                                fill_rule: str):
        """Fill a polygon with the specified fill rule."""
        # Fast path: use Rust to render directly to numpy array with AA
        if ctx.image_arr is not None:
            fill_rule_code = 1 if fill_rule == "evenodd" else 0
            # Use anti-aliased version for better edge quality
            vectorstag_rust.fill_polygon_aa_to_array(
                ctx.image_arr, points,
                fill[0], fill[1], fill[2], fill[3],
                fill_rule_code
            )
            return

        # Fallback: PIL-based rendering
        if fill_rule == "evenodd":
            self._fill_polygon_evenodd(ctx, points, fill)
        else:
            # Check if polygon is self-intersecting (stars, complex shapes)
            if self._is_self_intersecting(points):
                self._fill_polygon_nonzero_color(ctx, points, fill)
            else:
                # Simple non-intersecting polygon
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                min_x, max_x = max(0, int(min(xs))), min(ctx.image_width, int(max(xs)) + 1)
                min_y, max_y = max(0, int(min(ys))), min(ctx.image_height, int(max(ys)) + 1)
                if min_x < max_x and min_y < max_y:
                    width, height = max_x - min_x, max_y - min_y
                    if fill[3] < 255 or ctx.image is None:
                        temp = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(temp, "RGBA")
                        local_points = [(x - min_x, y - min_y) for x, y in points]
                        draw.polygon(local_points, fill=fill)
                        self._alpha_composite(ctx, temp, min_x, min_y)
                    else:
                        draw = ImageDraw.Draw(ctx.image, "RGBA")
                        draw.polygon(points, fill=fill)

    def _is_self_intersecting(self, points: list[tuple[float, float]]) -> bool:
        """Check if a polygon has self-intersecting edges (optimized)."""
        # Use Rust implementation if available (140x faster)
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

        width, height = ctx.image_size

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
        mask_arr = vectorstag_rust.fill_polygon_nonzero(
            points, crop_width, crop_height, min_x, min_y
        )
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
        max_x = min(ctx.image_width, max_x)
        max_y = min(ctx.image_height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return

        width = max_x - min_x
        height = max_y - min_y

        # Use Rust implementation if available (much faster)
        mask = vectorstag_rust.fill_multi_polygon_evenodd(polygons, width, height, min_x, min_y)
        # Apply fill using mask (memory-optimized: no full-size temp)
        mask_img = Image.fromarray(mask, "L")

        if fill_ref and fill_ref.startswith("url("):
            # Gradient fill - extract URL reference
            end_paren = fill_ref.find(")")
            if end_paren != -1:
                match = fill_ref[4:end_paren]
            else:
                match = fill_ref[4:-1]
            match = match.strip()
            # Strip quotes if present (SVG 2 allows quoted URLs)
            if (match.startswith("'") and match.endswith("'")) or (match.startswith('"') and match.endswith('"')):
                match = match[1:-1]
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
        max_x = min(ctx.image_width, max_x)
        max_y = min(ctx.image_height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return

        width = max_x - min_x
        height = max_y - min_y

        # Use Rust implementation if available (much faster)
        mask = vectorstag_rust.fill_multi_polygon_nonzero(polygons, width, height, min_x, min_y)
        # Apply fill using mask (memory-optimized: no full-size temp)
        mask_img = Image.fromarray(mask, "L")

        if fill_ref and fill_ref.startswith("url("):
            # Gradient fill - still needs mask-based approach
            end_paren = fill_ref.find(")")
            if end_paren != -1:
                match = fill_ref[4:end_paren]
            else:
                match = fill_ref[4:-1]
            match = match.strip()
            # Strip quotes if present (SVG 2 allows quoted URLs)
            if (match.startswith("'") and match.endswith("'")) or (match.startswith('"') and match.endswith('"')):
                match = match[1:-1]
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
            # Solid fill - use Rust direct rendering if available
            if ctx.image_arr is not None:
                vectorstag_rust.fill_multi_polygon_aa_to_array(
                    ctx.image_arr, polygons,
                    fill[0], fill[1], fill[2], fill[3],
                    0  # nonzero fill rule
                )
            else:
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
        max_x = min(ctx.image_width, max_x)
        max_y = min(ctx.image_height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return

        width = max_x - min_x
        height = max_y - min_y

        # Use Rust implementation if available (much faster)
        mask = vectorstag_rust.fill_polygon_evenodd(points, width, height, min_x, min_y)
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
        max_x = min(ctx.image_width, max_x)
        max_y = min(ctx.image_height, max_y)

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
            mask_arr = vectorstag_rust.fill_polygon_evenodd(points, grad_width, grad_height, min_x, min_y)
        else:
            mask_arr = vectorstag_rust.fill_polygon_nonzero(points, grad_width, grad_height, min_x, min_y)

        # Apply gradient with mask (memory-optimized: no full-size temp)
        self._composite_gradient_masked(ctx, grad_img, mask_arr, min_x, min_y)

    def _fill_polygon_with_pattern(self, ctx: "RenderContext",
                                   points: list[tuple[float, float]],
                                   pattern: Pattern,
                                   bbox: tuple[float, float, float, float],
                                   opacity: float,
                                   fill_rule: str = "nonzero",
                                   element_transform: Transform = None):
        """Fill a polygon with a tiled pattern."""
        if not points or len(points) < 3:
            return

        if pattern.width <= 0 or pattern.height <= 0:
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
        max_x = min(ctx.image_width, max_x)
        max_y = min(ctx.image_height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return

        fill_width = max_x - min_x
        fill_height = max_y - min_y

        # Calculate pattern tile size in screen coordinates
        bx, by, bw, bh = bbox
        scale_x = ctx.base_transform.a
        scale_y = ctx.base_transform.d

        if pattern.pattern_units == "objectBoundingBox":
            # Pattern dimensions are relative to bbox (0-1 range)
            tile_w = int(pattern.width * bw * scale_x + 0.5)
            tile_h = int(pattern.height * bh * scale_y + 0.5)
            pat_x = bx + pattern.x * bw
            pat_y = by + pattern.y * bh
        else:
            # Pattern dimensions are in user space
            tile_w = int(pattern.width * scale_x + 0.5)
            tile_h = int(pattern.height * scale_y + 0.5)
            pat_x = pattern.x
            pat_y = pattern.y

        if tile_w <= 0 or tile_h <= 0:
            return

        # Render pattern tile
        tile_img = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))

        # Create a sub-context for rendering pattern elements
        if pattern.pattern_content_units == "objectBoundingBox":
            # Content coordinates are relative to bbox
            content_transform = Transform(bw, 0, 0, bh, bx, by)
            content_transform = ctx.base_transform.multiply(content_transform)
        else:
            # Content coordinates are in user space, scale to tile size
            if pattern.viewbox:
                vb_x, vb_y, vb_w, vb_h = pattern.viewbox
                sx = tile_w / vb_w if vb_w > 0 else 1
                sy = tile_h / vb_h if vb_h > 0 else 1
                content_transform = Transform(sx, 0, 0, sy, -vb_x * sx, -vb_y * sy)
            else:
                content_transform = Transform(scale_x, 0, 0, scale_y, 0, 0)

        # Create context without this pattern to prevent recursion
        safe_patterns = {k: v for k, v in ctx.patterns.items() if k != pattern.id}
        tile_ctx = RenderContext(tile_img, ctx.gradients, content_transform,
                                 ctx.clip_paths, ctx.masks, ctx.filters, safe_patterns,
                                 ctx.elements_by_id, ctx.path_elements)

        # Render pattern elements (with recursion limit)
        for elem in pattern.elements:
            self._render_element(tile_ctx, elem, depth=5)

        # Handle patternTransform
        has_transform = pattern.transform is not None

        if has_transform:
            # For transformed patterns, we need to tile in a larger area and transform
            import math
            # Calculate expanded bounds to handle rotation
            diag = math.sqrt(fill_width**2 + fill_height**2)
            expand = int(diag - min(fill_width, fill_height)) // 2 + tile_w + tile_h
            exp_width = fill_width + 2 * expand
            exp_height = fill_height + 2 * expand

            # Create larger tiled pattern
            exp_pattern = Image.new("RGBA", (exp_width, exp_height), (0, 0, 0, 0))

            # Calculate pattern offset in expanded coordinates
            px, py = ctx.base_transform.apply(pat_x, pat_y)
            offset_x = int((min_x - expand - px) % tile_w) if tile_w > 0 else 0
            offset_y = int((min_y - expand - py) % tile_h) if tile_h > 0 else 0

            # Tile the pattern in expanded area
            for ty in range(-offset_y, exp_height, tile_h):
                for tx in range(-offset_x, exp_width, tile_w):
                    exp_pattern.paste(tile_img, (tx, ty), tile_img)

            # Apply patternTransform using PIL
            # Convert Transform to PIL affine (inverse for sampling)
            pt = pattern.transform
            # Apply transform centered on the fill region
            cx, cy = exp_width / 2, exp_height / 2

            # Create affine matrix for PIL (inverse transform for resampling)
            # PIL expects [a, b, c, d, e, f] where new_x = a*x + b*y + c
            det = pt.a * pt.d - pt.b * pt.c
            if abs(det) > 1e-10:
                inv_a = pt.d / det
                inv_b = -pt.b / det
                inv_c = (pt.b * pt.f - pt.d * pt.e) / det
                inv_d = -pt.c / det
                inv_e = pt.a / det
                inv_f = (pt.c * pt.e - pt.a * pt.f) / det

                # Apply transform around center of fill region
                fill_cx = fill_width / 2
                fill_cy = fill_height / 2
                exp_cx = expand + fill_cx
                exp_cy = expand + fill_cy

                # Translate to center, apply inverse, translate back
                affine = (
                    inv_a, inv_b, exp_cx - inv_a * fill_cx - inv_b * fill_cy,
                    inv_d, inv_e, exp_cy - inv_d * fill_cx - inv_e * fill_cy
                )

                pattern_img = exp_pattern.transform(
                    (fill_width, fill_height),
                    Image.AFFINE,
                    affine,
                    resample=Image.BILINEAR
                )
            else:
                # Singular transform - just crop
                pattern_img = exp_pattern.crop((expand, expand, expand + fill_width, expand + fill_height))
        else:
            # No transform - simple tiling
            pattern_img = Image.new("RGBA", (fill_width, fill_height), (0, 0, 0, 0))

            # Calculate pattern offset
            px, py = ctx.base_transform.apply(pat_x, pat_y)
            offset_x = int((min_x - px) % tile_w) if tile_w > 0 else 0
            offset_y = int((min_y - py) % tile_h) if tile_h > 0 else 0

            # Tile the pattern
            for ty in range(-offset_y, fill_height, tile_h):
                for tx in range(-offset_x, fill_width, tile_w):
                    pattern_img.paste(tile_img, (tx, ty), tile_img)

        # Apply opacity if needed
        if opacity < 1.0:
            pattern_arr = np.array(pattern_img)
            pattern_arr[:, :, 3] = (pattern_arr[:, :, 3] * opacity).astype(np.uint8)
            pattern_img = Image.fromarray(pattern_arr, "RGBA")

        # Create mask from polygon
        if fill_rule == "evenodd":
            mask_arr = vectorstag_rust.fill_polygon_evenodd(points, fill_width, fill_height, min_x, min_y)
        else:
            mask_arr = vectorstag_rust.fill_polygon_nonzero(points, fill_width, fill_height, min_x, min_y)
        # Apply pattern with mask
        self._composite_gradient_masked(ctx, pattern_img, mask_arr, min_x, min_y)

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

    def _resolve_gradient_stops(self, ctx: "RenderContext",
                                gradient: Union[LinearGradient, RadialGradient],
                                visited: set = None) -> list:
        """Resolve gradient stops by following href chain.

        Per SVG spec, gradients can inherit stops from referenced gradients
        via xlink:href. This method follows the chain to find the stops.
        """
        if visited is None:
            visited = set()

        # Prevent infinite loops
        if gradient.id in visited:
            return []
        visited.add(gradient.id)

        # If this gradient has stops, return them
        if gradient.stops:
            return gradient.stops

        # Otherwise, follow the href chain
        if gradient.href and gradient.href in ctx.gradients:
            ref_gradient = ctx.gradients[gradient.href]
            return self._resolve_gradient_stops(ctx, ref_gradient, visited)

        return []

    def _create_linear_gradient_image(self, ctx: "RenderContext",
                                      gradient: LinearGradient,
                                      width: int, height: int,
                                      bbox: tuple[float, float, float, float],
                                      offset_x: int, offset_y: int,
                                      opacity: float,
                                      element_transform: Transform = None) -> Image.Image:
        """Create an image filled with a linear gradient (memory-optimized)."""
        # Resolve stops from href chain if needed
        stops = self._resolve_gradient_stops(ctx, gradient)

        # Check for invalid gradient (invalid gradientUnits or no stops)
        if gradient.units == "invalid" or not stops:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Get gradient vector in gradient space
        if gradient.units == "objectBoundingBox":
            bx, by, bw, bh = bbox
            x1 = bx + gradient.x1 * bw
            y1 = by + gradient.y1 * bh
            x2 = bx + gradient.x2 * bw
            y2 = by + gradient.y2 * bh
        else:
            # userSpaceOnUse - convert percentage values to viewport coordinates
            x1, y1 = gradient.x1, gradient.y1
            x2, y2 = gradient.x2, gradient.y2

            # If values were percentages, scale by viewport dimensions
            vp_w = ctx.viewport_width or 100
            vp_h = ctx.viewport_height or 100
            if getattr(gradient, 'x1_pct', False):
                x1 = gradient.x1 * vp_w
            if getattr(gradient, 'y1_pct', False):
                y1 = gradient.y1 * vp_h
            if getattr(gradient, 'x2_pct', False):
                x2 = gradient.x2 * vp_w
            if getattr(gradient, 'y2_pct', False):
                y2 = gradient.y2 * vp_h

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

        # Get spread method
        spread_method = getattr(gradient, 'spread_method', 'pad')
        spread_code = 0  # pad
        if spread_method == "repeat":
            spread_code = 1
        elif spread_method == "reflect":
            spread_code = 2

        # Use Rust implementation if available (much faster)
        offsets = [float(s.offset) for s in stops]
        colors = [tuple(s.color) for s in stops]
        pixels = vectorstag_rust.create_linear_gradient_image(
            width, height, offset_x, offset_y,
            float(x1), float(y1), float(dx), float(dy), float(length),
            offsets, colors, opacity, spread_code
        )
        # Return numpy array directly - avoid PIL conversion
        return pixels

        # Python fallback
        t = np.empty((height, width), dtype=np.float32)
        y_base = np.arange(height, dtype=np.float32) + offset_y
        x_base = np.arange(width, dtype=np.float32) + offset_x

        for row in range(height):
            wy = y_base[row]
            t[row, :] = ((x_base - x1) * dx + (wy - y1) * dy) / length

        if spread_method == "repeat":
            np.remainder(t, 1.0, out=t)
        elif spread_method == "reflect":
            np.remainder(t, 2.0, out=t)
            mask = t > 1.0
            t[mask] = 2.0 - t[mask]
        else:
            np.clip(t, 0, 1, out=t)

        pixels = self._interpolate_gradient_colors_vectorized(stops, t, opacity)
        return Image.fromarray(pixels, "RGBA")

    def _create_radial_gradient_image(self, ctx: "RenderContext",
                                      gradient: RadialGradient,
                                      width: int, height: int,
                                      bbox: tuple[float, float, float, float],
                                      offset_x: int, offset_y: int,
                                      opacity: float,
                                      element_transform: Transform = None) -> Image.Image:
        """Create an image filled with a radial gradient (memory-optimized)."""
        # Resolve stops from href chain if needed
        stops = self._resolve_gradient_stops(ctx, gradient)

        # Check for invalid gradient (invalid gradientUnits or no stops)
        if gradient.units == "invalid" or not stops:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Get gradient parameters in gradient space
        if gradient.units == "objectBoundingBox":
            bx, by, bw, bh = bbox
            cx = bx + gradient.cx * bw
            cy = by + gradient.cy * bh
            r = gradient.r * max(bw, bh)
            # Focal point defaults to center
            fx = bx + (gradient.fx if gradient.fx is not None else gradient.cx) * bw
            fy = by + (gradient.fy if gradient.fy is not None else gradient.cy) * bh
            fr = gradient.fr * max(bw, bh)
        else:
            # userSpaceOnUse - convert percentage values to viewport coordinates
            vp_w = ctx.viewport_width or 100
            vp_h = ctx.viewport_height or 100
            vp_max = max(vp_w, vp_h)

            cx = gradient.cx * vp_w if getattr(gradient, 'cx_pct', False) else gradient.cx
            cy = gradient.cy * vp_h if getattr(gradient, 'cy_pct', False) else gradient.cy
            r = gradient.r * vp_max if getattr(gradient, 'r_pct', False) else gradient.r

            # Focal point defaults to center
            if gradient.fx is not None:
                fx = gradient.fx * vp_w if getattr(gradient, 'fx_pct', False) else gradient.fx
            else:
                fx = cx
            if gradient.fy is not None:
                fy = gradient.fy * vp_h if getattr(gradient, 'fy_pct', False) else gradient.fy
            else:
                fy = cy
            fr = gradient.fr * vp_max if getattr(gradient, 'fr_pct', False) else gradient.fr

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

        # Get spread method
        spread_method = getattr(gradient, 'spread_method', 'pad')
        spread_code = 0  # pad
        if spread_method == "repeat":
            spread_code = 1
        elif spread_method == "reflect":
            spread_code = 2

        # Use Rust implementation if available (much faster)
        offsets = [float(s.offset) for s in stops]
        colors = [tuple(s.color) for s in stops]
        pixels = vectorstag_rust.create_radial_gradient_image(
            width, height, offset_x, offset_y,
            float(cx), float(cy), float(r),
            inv_a, inv_b, inv_c, inv_d, inv_e, inv_f,
            offsets, colors, opacity, spread_code
        )
        # Return numpy array directly - avoid PIL conversion
        return pixels

        # Python fallback
        t = np.empty((height, width), dtype=np.float32)
        x_base = np.arange(width, dtype=np.float32) + offset_x

        # Calculate effective radius for gradient (r - fr)
        effective_r = r - fr
        if effective_r <= 0:
            effective_r = 1e-10  # Avoid division by zero

        for row in range(height):
            wy = float(row + offset_y)
            gx = inv_a * x_base + (inv_b * wy + inv_e)
            gy = inv_c * x_base + (inv_d * wy + inv_f)
            # Distance from focal point, normalized with focal radius
            dist = np.sqrt((gx - fx) ** 2 + (gy - fy) ** 2)
            t[row, :] = (dist - fr) / effective_r

        if spread_method == "repeat":
            np.remainder(t, 1.0, out=t)
        elif spread_method == "reflect":
            np.remainder(t, 2.0, out=t)
            mask = t > 1.0
            t[mask] = 2.0 - t[mask]
        else:
            np.clip(t, 0, 1, out=t)

        pixels = self._interpolate_gradient_colors_vectorized(stops, t, opacity)
        return Image.fromarray(pixels, "RGBA")

    def _interpolate_gradient_colors_vectorized(self, stops: list[GradientStop],
                                                  t: np.ndarray, opacity: float) -> np.ndarray:
        """Memory-optimized vectorized color interpolation for gradient images."""
        height, width = t.shape

        if not stops:
            return np.zeros((height, width, 4), dtype=np.uint8)

        # Use Rust implementation if available (much faster)
        offsets = [float(s.offset) for s in stops]
        colors = [tuple(s.color) for s in stops]
        t_float32 = t.astype(np.float32) if t.dtype != np.float32 else t
        return vectorstag_rust.interpolate_gradient_colors(t_float32, offsets, colors, opacity)
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

    def _alpha_composite(self, ctx: "RenderContext", src_img: Image.Image,
                         dest_x: int, dest_y: int):
        """Alpha composite src onto ctx.image or ctx.image_arr at destination coordinates."""
        if ctx.image is not None:
            # Use PIL's native compositing
            ctx.image.alpha_composite(src_img, (dest_x, dest_y))
        elif ctx.image_arr is not None:
            # Use Rust for fast compositing onto numpy array
            src_arr = np.array(src_img)  # PIL arrays are already C-contiguous
            vectorstag_rust.alpha_composite_inplace(ctx.image_arr, src_arr, dest_x, dest_y)

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
        self._alpha_composite(ctx, fill_img, dest_x, dest_y)

    def _composite_gradient_masked(self, ctx: "RenderContext", grad_img,
                                    mask_img, dest_x: int, dest_y: int):
        """Composite a gradient with mask without creating a full-size temp image.

        Args:
            grad_img: Gradient image (can be PIL Image or numpy array)
            mask_img: Mask image (can be PIL Image 'L' mode or numpy array)
        """
        # Convert to numpy if needed
        grad_arr = grad_img if isinstance(grad_img, np.ndarray) else np.array(grad_img)
        mask_arr = mask_img if isinstance(mask_img, np.ndarray) else np.array(mask_img)

        # Multiply alpha channel by mask (both are 0-255)
        grad_arr[:, :, 3] = (grad_arr[:, :, 3].astype(np.uint16) * mask_arr // 255).astype(np.uint8)

        # Composite directly at destination - avoid PIL conversion when possible
        if ctx.image is None:
            # Direct Rust compositing onto numpy array
            grad_arr = np.ascontiguousarray(grad_arr)
            vectorstag_rust.alpha_composite_inplace(ctx.image_arr, grad_arr, dest_x, dest_y)
        else:
            grad_masked = Image.fromarray(grad_arr, "RGBA")
            self._alpha_composite(ctx, grad_masked, dest_x, dest_y)

    # Font mapping for common font families
    FONT_PATHS = {
        # Serif fonts
        "serif": [
            "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/System/Library/Fonts/Times.ttc",
            "times.ttf",
        ],
        "times": [
            "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/System/Library/Fonts/Times.ttc",
        ],
        "times new roman": [
            "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/System/Library/Fonts/Times.ttc",
        ],
        # Sans-serif fonts
        "sans-serif": [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "arial.ttf",
        ],
        "arial": [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "arial.ttf",
        ],
        "helvetica": [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ],
        # Monospace fonts
        "monospace": [
            "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/System/Library/Fonts/Courier.ttc",
            "cour.ttf",
        ],
        "courier": [
            "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
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

        # Final fallback to any available font
        fallbacks = [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
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

        fill = self._get_fill_color(ctx, text.style)
        if not fill:
            fill = (0, 0, 0, 255)

        # Calculate font size with transform scaling
        font_size = max(1, int(text.font_size * self._get_scale(ctx, text.transform)))
        font = self._get_font(text.font_family, font_size)

        # Check for textPath
        if text.text_path_href or text.text_path_data:
            self._render_text_on_path(ctx, text, transform, fill, font)
            return

        x, y = transform.apply(text.x, text.y)

        # Map SVG text-anchor to PIL anchor
        # SVG: start=left, middle=center, end=right
        # PIL anchor: first char is horizontal (l/m/r), second is vertical (a/m/s/d/b)
        # "ls" = left baseline, "ms" = middle baseline, "rs" = right baseline
        anchor_map = {"start": "ls", "middle": "ms", "end": "rs"}
        anchor = anchor_map.get(text.text_anchor, "ls")

        if ctx.image is None:
            # Create temp image for text rendering when using Rust path
            temp = Image.new("RGBA", ctx.image_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(temp, "RGBA")
            draw.text((x, y), text.text, fill=fill, font=font, anchor=anchor)
            self._alpha_composite(ctx, temp, 0, 0)
        else:
            draw = ImageDraw.Draw(ctx.image, "RGBA")
            draw.text((x, y), text.text, fill=fill, font=font, anchor=anchor)

    def _render_text_on_path(self, ctx: "RenderContext", text: TextElement,
                             transform: Transform, fill: tuple, font):
        """Render text along a path (textPath element)."""
        import math

        # Calculate font size with transform scaling
        font_size = max(1, int(text.font_size * self._get_scale(ctx, text.transform)))

        # Get path data - either from reference or direct attribute
        path_data = text.text_path_data
        if not path_data and text.text_path_href:
            # Look up path element from document
            if hasattr(ctx, 'path_elements') and text.text_path_href in ctx.path_elements:
                path_data = ctx.path_elements[text.text_path_href]

        if not path_data:
            return

        # Parse path and get points with cumulative distances
        points = self._sample_path_points(path_data, transform)
        if not points or len(points) < 2:
            return

        # Calculate total path length
        total_length = points[-1][2]  # (x, y, cumulative_distance)
        if total_length <= 0:
            return

        # Calculate starting offset
        start_offset = text.text_path_start_offset
        if start_offset < 1:  # Treat as percentage
            start_offset = start_offset * total_length

        # Create temp image for text rendering
        temp = Image.new("RGBA", ctx.image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp, "RGBA")

        # Render each character along the path
        current_dist = start_offset
        for char in text.text:
            if char.isspace():
                # Get space width
                try:
                    char_width = font.getlength(" ")
                except:
                    char_width = font_size * 0.3
                current_dist += char_width
                continue

            # Get character width
            try:
                char_width = font.getlength(char)
            except:
                char_width = font_size * 0.6

            # Find position and angle at current distance (center of character)
            char_center_dist = current_dist + char_width / 2
            pos_angle = self._get_point_on_path(points, char_center_dist)
            if pos_angle is None:
                break

            px, py, angle = pos_angle

            # Create rotated character image
            # Use "ms" anchor: middle horizontally, baseline vertically (baseline on path)
            char_img = Image.new("RGBA", (int(char_width * 2) + 10, int(font_size * 2) + 10), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img, "RGBA")
            char_draw.text((char_img.width // 2, char_img.height // 2), char, fill=fill, font=font, anchor="ms")

            # Rotate character
            rotated = char_img.rotate(-math.degrees(angle), expand=True, resample=Image.BICUBIC)

            # Calculate paste position (centered on path point)
            paste_x = int(px - rotated.width // 2)
            paste_y = int(py - rotated.height // 2)

            # Composite onto temp image
            temp.paste(rotated, (paste_x, paste_y), rotated)

            current_dist += char_width

        self._alpha_composite(ctx, temp, 0, 0)

    def _sample_path_points(self, path_data: str, transform: Transform, num_samples: int = 500) -> list:
        """Sample points along a path with cumulative distances."""
        import math

        # Parse path commands
        commands = self._parse_path_commands(path_data)
        if not commands:
            return []

        points = []
        current_x, current_y = 0, 0
        start_x, start_y = 0, 0
        cumulative_dist = 0

        for cmd, args in commands:
            if cmd == 'M':
                current_x, current_y = args[0], args[1]
                start_x, start_y = current_x, current_y
                tx, ty = transform.apply(current_x, current_y)
                points.append((tx, ty, cumulative_dist))
            elif cmd == 'm':
                current_x += args[0]
                current_y += args[1]
                start_x, start_y = current_x, current_y
                tx, ty = transform.apply(current_x, current_y)
                points.append((tx, ty, cumulative_dist))
            elif cmd == 'L':
                for i in range(0, len(args), 2):
                    next_x, next_y = args[i], args[i+1]
                    # Sample line with distances in render coordinates
                    prev_tx, prev_ty = transform.apply(current_x, current_y)
                    next_tx, next_ty = transform.apply(next_x, next_y)
                    total_dist = math.sqrt((next_tx - prev_tx)**2 + (next_ty - prev_ty)**2)
                    steps = max(2, int(total_dist / 5))
                    for j in range(1, steps + 1):
                        t = j / steps
                        px = current_x + t * (next_x - current_x)
                        py = current_y + t * (next_y - current_y)
                        tx, ty = transform.apply(px, py)
                        step_dist = math.sqrt((tx - prev_tx)**2 + (ty - prev_ty)**2)
                        cumulative_dist += step_dist
                        points.append((tx, ty, cumulative_dist))
                        prev_tx, prev_ty = tx, ty
                    current_x, current_y = next_x, next_y
            elif cmd == 'l':
                for i in range(0, len(args), 2):
                    next_x = current_x + args[i]
                    next_y = current_y + args[i+1]
                    prev_tx, prev_ty = transform.apply(current_x, current_y)
                    next_tx, next_ty = transform.apply(next_x, next_y)
                    total_dist = math.sqrt((next_tx - prev_tx)**2 + (next_ty - prev_ty)**2)
                    steps = max(2, int(total_dist / 5))
                    for j in range(1, steps + 1):
                        t = j / steps
                        px = current_x + t * (next_x - current_x)
                        py = current_y + t * (next_y - current_y)
                        tx, ty = transform.apply(px, py)
                        step_dist = math.sqrt((tx - prev_tx)**2 + (ty - prev_ty)**2)
                        cumulative_dist += step_dist
                        points.append((tx, ty, cumulative_dist))
                        prev_tx, prev_ty = tx, ty
                    current_x, current_y = next_x, next_y
            elif cmd == 'C':
                # Cubic bezier
                for i in range(0, len(args), 6):
                    x1, y1 = args[i], args[i+1]
                    x2, y2 = args[i+2], args[i+3]
                    x3, y3 = args[i+4], args[i+5]
                    # Sample bezier curve
                    steps = 20
                    prev_tx, prev_ty = transform.apply(current_x, current_y)
                    for j in range(1, steps + 1):
                        t = j / steps
                        u = 1 - t
                        px = u*u*u*current_x + 3*u*u*t*x1 + 3*u*t*t*x2 + t*t*t*x3
                        py = u*u*u*current_y + 3*u*u*t*y1 + 3*u*t*t*y2 + t*t*t*y3
                        tx, ty = transform.apply(px, py)
                        # Compute distance in render coordinates
                        step_dist = math.sqrt((tx - prev_tx)**2 + (ty - prev_ty)**2)
                        cumulative_dist += step_dist
                        points.append((tx, ty, cumulative_dist))
                        prev_tx, prev_ty = tx, ty
                    current_x, current_y = x3, y3
            elif cmd == 'c':
                # Relative cubic bezier
                for i in range(0, len(args), 6):
                    x1 = current_x + args[i]
                    y1 = current_y + args[i+1]
                    x2 = current_x + args[i+2]
                    y2 = current_y + args[i+3]
                    x3 = current_x + args[i+4]
                    y3 = current_y + args[i+5]
                    steps = 20
                    prev_tx, prev_ty = transform.apply(current_x, current_y)
                    for j in range(1, steps + 1):
                        t = j / steps
                        u = 1 - t
                        px = u*u*u*current_x + 3*u*u*t*x1 + 3*u*t*t*x2 + t*t*t*x3
                        py = u*u*u*current_y + 3*u*u*t*y1 + 3*u*t*t*y2 + t*t*t*y3
                        tx, ty = transform.apply(px, py)
                        step_dist = math.sqrt((tx - prev_tx)**2 + (ty - prev_ty)**2)
                        cumulative_dist += step_dist
                        points.append((tx, ty, cumulative_dist))
                        prev_tx, prev_ty = tx, ty
                    current_x, current_y = x3, y3
            elif cmd == 'Z' or cmd == 'z':
                if (current_x, current_y) != (start_x, start_y):
                    prev_tx, prev_ty = transform.apply(current_x, current_y)
                    tx, ty = transform.apply(start_x, start_y)
                    dist = math.sqrt((tx - prev_tx)**2 + (ty - prev_ty)**2)
                    cumulative_dist += dist
                    points.append((tx, ty, cumulative_dist))
                current_x, current_y = start_x, start_y

        return points

    def _get_point_on_path(self, points: list, distance: float) -> tuple:
        """Get position and angle at a given distance along the path."""
        import math

        if not points or distance < 0:
            return None

        # Find the segment containing this distance
        for i in range(1, len(points)):
            if points[i][2] >= distance:
                # Interpolate between points[i-1] and points[i]
                prev_x, prev_y, prev_dist = points[i-1]
                curr_x, curr_y, curr_dist = points[i]

                if curr_dist == prev_dist:
                    t = 0
                else:
                    t = (distance - prev_dist) / (curr_dist - prev_dist)

                px = prev_x + t * (curr_x - prev_x)
                py = prev_y + t * (curr_y - prev_y)

                # Calculate angle
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                angle = math.atan2(dy, dx)

                return (px, py, angle)

        # Beyond path end
        return None

    def _parse_path_commands(self, path_data: str) -> list:
        """Parse SVG path data into commands and arguments."""
        import re

        commands = []
        # Match command letter followed by numbers
        pattern = r'([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)'

        for match in re.finditer(pattern, path_data):
            cmd = match.group(1)
            args_str = match.group(2).strip()

            if args_str:
                # Parse numbers (handle negative numbers and scientific notation)
                nums = re.findall(r'-?\d*\.?\d+(?:[eE][+-]?\d+)?', args_str)
                args = [float(n) for n in nums]
            else:
                args = []

            commands.append((cmd, args))

        return commands

    def _render_image(self, ctx: "RenderContext", img_elem: ImageElement):
        """Render image element (embedded or external images)."""
        import base64
        import io
        import gzip

        href = img_elem.href
        if not href:
            return

        try:
            # Handle in-memory images from registry
            if href.startswith("memory:"):
                name = href[7:]  # Remove "memory:" prefix
                if name in self._image_registry:
                    img = self._image_registry[name].copy()
                else:
                    return  # Image not found in registry

            # Handle data URLs (embedded images)
            elif href.startswith("data:"):
                # Parse data URL: data:[<mediatype>][;base64],<data>
                # Remove newlines and whitespace that may be in the data URL
                href = href.replace("\n", "").replace("\r", "").replace(" ", "")

                # Extract MIME type
                mime_type = ""
                if href.startswith("data:") and ("," in href or ";" in href):
                    end = href.find(",")
                    if end == -1:
                        end = len(href)
                    header = href[5:end]
                    if ";base64" in header:
                        mime_type = header.split(";")[0]
                    else:
                        mime_type = header

                if ";base64," in href:
                    data_start = href.index(";base64,") + 8
                    data = base64.b64decode(href[data_start:])
                elif "," in href:
                    # Non-base64 data URL (rare)
                    data_start = href.index(",") + 1
                    data = href[data_start:].encode()
                else:
                    return

                # Handle SVG data URLs
                is_svg = False
                is_svgz = False

                # Check for gzip magic bytes (0x1f 0x8b)
                if len(data) >= 2 and data[0] == 0x1f and data[1] == 0x8b:
                    is_svgz = True
                elif mime_type in ("image/svg+xml", "image/svg+xml;charset=utf-8", ""):
                    # Try to detect SVG by content
                    try:
                        svg_content = data.decode('utf-8')
                        if '<svg' in svg_content or '<?xml' in svg_content:
                            is_svg = True
                    except:
                        pass

                if is_svgz:
                    # Decompress gzipped SVG
                    try:
                        svg_content = gzip.decompress(data).decode('utf-8')
                        self._render_embedded_svg_direct(ctx, img_elem, svg_content)
                        return
                    except:
                        pass
                elif is_svg:
                    svg_content = data.decode('utf-8')
                    self._render_embedded_svg_direct(ctx, img_elem, svg_content)
                    return
                # Handle SVGZ (gzipped SVG) by MIME type
                elif mime_type == "image/svg+xml-compressed":
                    svg_content = gzip.decompress(data).decode('utf-8')
                    self._render_embedded_svg_direct(ctx, img_elem, svg_content)
                    return
                else:
                    # Raster image (PNG, JPEG, GIF, etc.)
                    img = Image.open(io.BytesIO(data)).convert("RGBA")
            else:
                # External file reference - resolve relative path if base_path is available
                import os

                if img_elem.base_path:
                    # Resolve relative path against base path
                    if not os.path.isabs(href):
                        href = os.path.normpath(os.path.join(img_elem.base_path, href))

                    if os.path.exists(href):
                        # Determine file type by extension
                        ext = os.path.splitext(href)[1].lower()
                        if ext in ('.svg', '.svgz'):
                            # External SVG/SVGZ file
                            try:
                                if ext == '.svgz':
                                    with gzip.open(href, 'rt', encoding='utf-8') as f:
                                        svg_content = f.read()
                                else:
                                    with open(href, 'r', encoding='utf-8') as f:
                                        svg_content = f.read()
                                self._render_embedded_svg_direct(ctx, img_elem, svg_content)
                                return
                            except:
                                pass
                        else:
                            # Raster image (PNG, JPEG, GIF, etc.)
                            try:
                                img = Image.open(href).convert("RGBA")
                            except:
                                return
                    else:
                        return
                else:
                    return

            # Get target dimensions
            transform = ctx.base_transform.multiply(img_elem.transform)

            # Calculate position and size
            x = img_elem.x
            y = img_elem.y

            # Handle missing width/height with aspect ratio preservation
            # SVG spec: if only one dimension is specified, calculate the other
            # to preserve the image's intrinsic aspect ratio
            intrinsic_w = img.width
            intrinsic_h = img.height
            has_width = img_elem.width > 0
            has_height = img_elem.height > 0

            if has_width and has_height:
                # Both specified - use them directly
                width = img_elem.width
                height = img_elem.height
            elif has_width and not has_height:
                # Only width specified - calculate height from aspect ratio
                width = img_elem.width
                if intrinsic_w > 0:
                    height = width * (intrinsic_h / intrinsic_w)
                else:
                    height = intrinsic_h
            elif has_height and not has_width:
                # Only height specified - calculate width from aspect ratio
                height = img_elem.height
                if intrinsic_h > 0:
                    width = height * (intrinsic_w / intrinsic_h)
                else:
                    width = intrinsic_w
            else:
                # Neither specified - use intrinsic dimensions
                width = intrinsic_w
                height = intrinsic_h

            # Handle preserveAspectRatio
            par = img_elem.preserveAspectRatio
            is_slice = "slice" in par

            # Remember original viewport for slice clipping
            viewport_x = x
            viewport_y = y
            viewport_w = width
            viewport_h = height

            if par != "none":
                # Calculate aspect-ratio-preserving fit
                src_aspect = img.width / img.height if img.height > 0 else 1
                dst_aspect = width / height if height > 0 else 1

                if "meet" in par:
                    # Scale to fit within bounds (may have letterboxing)
                    if src_aspect > dst_aspect:
                        new_width = width
                        new_height = width / src_aspect
                    else:
                        new_height = height
                        new_width = height * src_aspect
                elif is_slice:
                    # Scale to cover bounds (may crop)
                    if src_aspect > dst_aspect:
                        new_height = height
                        new_width = height * src_aspect
                    else:
                        new_width = width
                        new_height = width / src_aspect
                else:
                    new_width = width
                    new_height = height

                # Handle alignment (xMidYMid is default)
                x_offset = 0
                y_offset = 0
                if "xMid" in par:
                    x_offset = (width - new_width) / 2
                elif "xMax" in par:
                    x_offset = width - new_width
                if "YMid" in par:
                    y_offset = (height - new_height) / 2
                elif "YMax" in par:
                    y_offset = height - new_height

                x += x_offset
                y += y_offset
                width = new_width
                height = new_height

            # Apply transform to get final coordinates
            x1, y1 = transform.apply(x, y)
            x2, y2 = transform.apply(x + width, y + height)

            # Calculate final dimensions
            final_width = int(abs(x2 - x1))
            final_height = int(abs(y2 - y1))
            final_x = int(min(x1, x2))
            final_y = int(min(y1, y2))

            if final_width <= 0 or final_height <= 0:
                return

            # Resize image to target dimensions
            resized = img.resize((final_width, final_height), Image.LANCZOS)

            # For slice mode, crop to viewport bounds
            if is_slice:
                # Transform viewport corners
                vp_x1, vp_y1 = transform.apply(viewport_x, viewport_y)
                vp_x2, vp_y2 = transform.apply(viewport_x + viewport_w, viewport_y + viewport_h)
                vp_final_x = int(min(vp_x1, vp_x2))
                vp_final_y = int(min(vp_y1, vp_y2))
                vp_final_w = int(abs(vp_x2 - vp_x1))
                vp_final_h = int(abs(vp_y2 - vp_y1))

                # Calculate crop region within the resized image
                crop_left = max(0, vp_final_x - final_x)
                crop_top = max(0, vp_final_y - final_y)
                crop_right = min(final_width, crop_left + vp_final_w)
                crop_bottom = min(final_height, crop_top + vp_final_h)

                if crop_right > crop_left and crop_bottom > crop_top:
                    resized = resized.crop((crop_left, crop_top, crop_right, crop_bottom))
                    final_x = vp_final_x
                    final_y = vp_final_y

            # Apply opacity if needed
            if img_elem.style.opacity < 1.0:
                alpha = resized.getchannel("A")
                alpha = alpha.point(lambda a: int(a * img_elem.style.opacity))
                resized.putalpha(alpha)

            # Composite onto canvas
            self._alpha_composite(ctx, resized, final_x, final_y)

        except Exception:
            # Silently fail for invalid images
            pass

    def _render_embedded_svg_direct(self, ctx: "RenderContext", img_elem: ImageElement, svg_content: str):
        """Render embedded SVG content directly to the context.

        Args:
            ctx: Current render context
            img_elem: The image element containing the SVG
            svg_content: SVG content as string
        """
        try:
            # Get target dimensions
            transform = ctx.base_transform.multiply(img_elem.transform)

            # Calculate target size
            x = img_elem.x
            y = img_elem.y
            width = img_elem.width if img_elem.width > 0 else 100
            height = img_elem.height if img_elem.height > 0 else 100

            # Apply transform to get final position and size
            x1, y1 = transform.apply(x, y)
            x2, y2 = transform.apply(x + width, y + height)
            final_width = max(1, int(abs(x2 - x1)))
            final_height = max(1, int(abs(y2 - y1)))
            final_x = int(min(x1, x2))
            final_y = int(min(y1, y2))

            # Create a new renderer for the embedded SVG
            # Use the same antialias setting but transparent background
            embedded_renderer = SVGRenderer(
                background=(0, 0, 0, 0),
                antialias=self.antialias
            )

            # Parse the embedded SVG to get its intrinsic dimensions
            doc = embedded_renderer.parser.parse(svg_content)
            if doc is None:
                return

            # Get intrinsic dimensions from the embedded SVG
            if doc.viewBox and len(doc.viewBox) >= 4:
                svg_w = doc.viewBox[2] if doc.viewBox[2] > 0 else 100
                svg_h = doc.viewBox[3] if doc.viewBox[3] > 0 else 100
            else:
                svg_w = doc.width if doc.width > 0 else 100
                svg_h = doc.height if doc.height > 0 else 100

            # Handle preserveAspectRatio for embedded SVG
            par = img_elem.preserveAspectRatio
            is_slice = "slice" in par

            render_w = final_width
            render_h = final_height
            crop_x = 0
            crop_y = 0

            if par != "none" and (is_slice or "meet" in par):
                src_aspect = svg_w / svg_h if svg_h > 0 else 1
                dst_aspect = final_width / final_height if final_height > 0 else 1

                if "meet" in par:
                    # Scale to fit within bounds
                    if src_aspect > dst_aspect:
                        render_w = final_width
                        render_h = int(final_width / src_aspect)
                    else:
                        render_h = final_height
                        render_w = int(final_height * src_aspect)
                elif is_slice:
                    # Scale to cover bounds
                    if src_aspect > dst_aspect:
                        render_h = final_height
                        render_w = int(final_height * src_aspect)
                    else:
                        render_w = final_width
                        render_h = int(final_width / src_aspect)

            render_w = max(1, render_w)
            render_h = max(1, render_h)

            # Render at calculated dimensions
            img = embedded_renderer.render_document(doc, render_w, render_h)
            if img is None:
                return

            img = img.convert("RGBA")

            # Handle alignment and cropping
            if par != "none":
                x_offset = 0
                y_offset = 0
                if "xMid" in par:
                    x_offset = (final_width - render_w) // 2
                elif "xMax" in par:
                    x_offset = final_width - render_w
                if "YMid" in par:
                    y_offset = (final_height - render_h) // 2
                elif "YMax" in par:
                    y_offset = final_height - render_h

                if is_slice:
                    # Crop to viewport
                    crop_x = -x_offset if x_offset < 0 else 0
                    crop_y = -y_offset if y_offset < 0 else 0
                    crop_right = min(render_w, crop_x + final_width)
                    crop_bottom = min(render_h, crop_y + final_height)
                    if crop_right > crop_x and crop_bottom > crop_y:
                        img = img.crop((crop_x, crop_y, crop_right, crop_bottom))
                else:
                    # For meet, adjust final position
                    final_x += x_offset
                    final_y += y_offset

            # Apply opacity if needed
            if img_elem.style.opacity < 1.0:
                alpha = img.getchannel("A")
                alpha = alpha.point(lambda a: int(a * img_elem.style.opacity))
                img.putalpha(alpha)

            # Composite onto canvas
            self._alpha_composite(ctx, img, final_x, final_y)
        except Exception:
            pass

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
                 masks: dict[str, Mask] = None,
                 filters: dict = None,
                 patterns: dict[str, Pattern] = None,
                 elements_by_id: dict = None,
                 path_elements: dict = None,
                 viewport_width: float = None,
                 viewport_height: float = None):
        self.image = image
        self.image_arr = None  # Optional numpy array for Rust compositing
        self.gradients = gradients
        self.patterns = patterns or {}
        self.base_transform = base_transform
        self.clip_paths = clip_paths or {}
        self.masks = masks or {}
        self.filters = filters or {}
        self.elements_by_id = elements_by_id or {}
        self.path_elements = path_elements or {}
        # Viewport dimensions for userSpaceOnUse gradient percentages
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        # Cache for clip masks: key = (clip_path_id, transform_tuple, bbox_tuple)
        self.clip_mask_cache: dict[tuple, np.ndarray] = {}

    @property
    def image_width(self) -> int:
        """Get image width from either PIL image or numpy array."""
        if self.image_arr is not None:
            return self.image_arr.shape[1]
        return self.image.width

    @property
    def image_height(self) -> int:
        """Get image height from either PIL image or numpy array."""
        if self.image_arr is not None:
            return self.image_arr.shape[0]
        return self.image.height

    @property
    def image_size(self) -> tuple[int, int]:
        """Get image (width, height) from either PIL image or numpy array."""
        if self.image_arr is not None:
            return (self.image_arr.shape[1], self.image_arr.shape[0])
        return self.image.size

    def create_child_context(self) -> "RenderContext":
        """Create a child context with transparent background for group opacity."""
        import numpy as np

        # Create context matching parent's rendering mode (numpy array or PIL image)
        if self.image_arr is not None:
            # Parent uses numpy array - create child with numpy array too
            child_arr = np.zeros_like(self.image_arr)
            child_ctx = RenderContext(
                image=None,
                gradients=self.gradients,
                base_transform=self.base_transform,
                clip_paths=self.clip_paths,
                masks=self.masks,
                filters=self.filters,
                patterns=self.patterns,
                elements_by_id=self.elements_by_id,
                path_elements=self.path_elements,
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height
            )
            child_ctx.image_arr = child_arr
        else:
            # Parent uses PIL image
            child_image = Image.new('RGBA', self.image_size, (0, 0, 0, 0))
            child_ctx = RenderContext(
                image=child_image,
                gradients=self.gradients,
                base_transform=self.base_transform,
                clip_paths=self.clip_paths,
                masks=self.masks,
                filters=self.filters,
                patterns=self.patterns,
                elements_by_id=self.elements_by_id,
                path_elements=self.path_elements,
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height
            )
        # Share clip mask cache
        child_ctx.clip_mask_cache = self.clip_mask_cache
        return child_ctx
