#!/usr/bin/env python3
"""Analyze tiger.svg to understand bbox/stroke differences."""

from vectorstag.parser import SVGParser

# Read tiger SVG
with open('samples/svg/tiger.svg', 'r') as f:
    svg_content = f.read()

parser = SVGParser()

# Temporarily disable stroke expansion to compare
original_compute = parser._compute_element_bbox

stroke_widths_seen = set()

def compute_element_bbox_wrapper(elem):
    """Wrapper to log stroke widths."""
    if hasattr(elem, 'style') and elem.style.stroke:
        stroke_widths_seen.add(elem.style.stroke_width)
    return original_compute(elem)

parser._compute_element_bbox = compute_element_bbox_wrapper

doc = parser.parse(svg_content)
print(f"VectorStag size (with stroke expansion): {doc.width:.2f} x {doc.height:.2f}")
print(f"Stroke widths seen: {sorted(stroke_widths_seen)}")

# Now compute without stroke expansion
def compute_element_bbox_no_stroke(elem):
    """Compute without stroke expansion."""
    if hasattr(elem, 'transform'):
        from vectorstag.parser import GroupElement, RectElement, CircleElement, EllipseElement
        from vectorstag.parser import LineElement, PolygonElement, PolylineElement, PathElement, TextElement

        if isinstance(elem, GroupElement):
            return parser._compute_elements_bbox(elem.children)

        bbox = None
        if isinstance(elem, RectElement):
            bbox = (elem.x, elem.y, elem.x + elem.width, elem.y + elem.height)
        elif isinstance(elem, CircleElement):
            bbox = (elem.cx - elem.r, elem.cy - elem.r,
                    elem.cx + elem.r, elem.cy + elem.r)
        elif isinstance(elem, EllipseElement):
            bbox = (elem.cx - elem.rx, elem.cy - elem.ry,
                    elem.cx + elem.rx, elem.cy + elem.ry)
        elif isinstance(elem, LineElement):
            bbox = (min(elem.x1, elem.x2), min(elem.y1, elem.y2),
                    max(elem.x1, elem.x2), max(elem.y1, elem.y2))
        elif isinstance(elem, (PolygonElement, PolylineElement)):
            if elem.points:
                xs = [p[0] for p in elem.points]
                ys = [p[1] for p in elem.points]
                bbox = (min(xs), min(ys), max(xs), max(ys))
        elif isinstance(elem, PathElement):
            bbox = parser._compute_path_bbox(elem.commands)
        elif isinstance(elem, TextElement):
            text_width = len(elem.text) * elem.font_size * 0.6
            bbox = (elem.x, elem.y - elem.font_size, elem.x + text_width, elem.y)

        if not bbox:
            return None

        corners = [
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3])
        ]

        transformed = [elem.transform.apply(x, y) for x, y in corners]
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]

        # NO stroke expansion
        return (min(xs), min(ys), max(xs), max(ys))
    return None

parser._compute_element_bbox = compute_element_bbox_no_stroke
doc2 = parser.parse(svg_content)
print(f"VectorStag size (without stroke expansion): {doc2.width:.2f} x {doc2.height:.2f}")

print(f"\nresvg size: 510 x 565")
print(f"\nDifference (with stroke): {doc.width - 510:.2f} x {doc.height - 565:.2f}")
print(f"Difference (without stroke): {doc2.width - 510:.2f} x {doc2.height - 565:.2f}")
