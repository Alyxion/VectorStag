#!/usr/bin/env python3
"""Analyze bezier bounding box accuracy."""

import math

def cubic_bezier_extrema(p0, p1, p2, p3):
    """Compute actual extrema of cubic bezier curve.

    For a cubic bezier B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
    The derivative is:
    B'(t) = 3*(1-t)^2*(P1-P0) + 6*(1-t)*t*(P2-P1) + 3*t^2*(P3-P2)

    Extrema occur at t=0, t=1, or where B'(t)=0.
    """
    points = [p0, p3]  # Always include endpoints

    # For each dimension (x, y), find t values where derivative is 0
    for dim in [0, 1]:
        # Coefficients of derivative: at^2 + bt + c = 0
        v0 = p0[dim]
        v1 = p1[dim]
        v2 = p2[dim]
        v3 = p3[dim]

        # B'(t) = 3*(1-t)^2*(v1-v0) + 6*(1-t)*t*(v2-v1) + 3*t^2*(v3-v2)
        # Expanding: 3*(v1-v0) - 6*t*(v1-v0) + 3*t^2*(v1-v0) + 6*t*(v2-v1) - 6*t^2*(v2-v1) + 3*t^2*(v3-v2)
        # = 3*(v1-v0) + t*(-6*(v1-v0) + 6*(v2-v1)) + t^2*(3*(v1-v0) - 6*(v2-v1) + 3*(v3-v2))

        c = 3 * (v1 - v0)
        b = -6 * (v1 - v0) + 6 * (v2 - v1)
        a = 3 * (v1 - v0) - 6 * (v2 - v1) + 3 * (v3 - v2)

        # Solve at^2 + bt + c = 0
        if abs(a) < 1e-10:
            if abs(b) > 1e-10:
                t = -c / b
                if 0 < t < 1:
                    points.append(evaluate_cubic(t, p0, p1, p2, p3))
        else:
            disc = b*b - 4*a*c
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                t1 = (-b + sqrt_disc) / (2*a)
                t2 = (-b - sqrt_disc) / (2*a)
                for t in [t1, t2]:
                    if 0 < t < 1:
                        points.append(evaluate_cubic(t, p0, p1, p2, p3))

    return points

def evaluate_cubic(t, p0, p1, p2, p3):
    """Evaluate cubic bezier at parameter t."""
    mt = 1 - t
    x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
    y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)

# Test case: control points outside curve extent
p0 = (0, 0)
p1 = (0, 100)  # Control point far outside
p2 = (100, 100)  # Control point far outside
p3 = (100, 0)

print("Control points:", p0, p1, p2, p3)
print("Control point bbox: x=[0,100], y=[0,100]")

extrema = cubic_bezier_extrema(p0, p1, p2, p3)
xs = [p[0] for p in extrema]
ys = [p[1] for p in extrema]
print(f"Actual curve bbox: x=[{min(xs):.2f},{max(xs):.2f}], y=[{min(ys):.2f},{max(ys):.2f}]")
print(f"y extent savings: {100 - max(ys):.2f}")

# Now analyze tiger paths to see how much we over-estimate
print("\n--- Tiger Path Analysis ---")

from vectorstag.parser import SVGParser, PathElement

with open('samples/svg/tiger.svg', 'r') as f:
    svg_content = f.read()

parser = SVGParser()

# Override to not parse content, just collect paths
import xml.etree.ElementTree as ET
root = ET.fromstring(svg_content)

def find_paths(elem, level=0):
    """Find all path elements."""
    paths = []
    for child in elem:
        tag = child.tag.split('}')[-1]  # Remove namespace
        if tag == 'path':
            d = child.get('d', '')
            paths.append(d)
        paths.extend(find_paths(child, level+1))
    return paths

all_paths = find_paths(root)
print(f"Found {len(all_paths)} paths")

# Parse paths and compute bbox difference
from vectorstag.path_parser import parse_path

total_control_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]
total_actual_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]

for path_d in all_paths[:10]:  # Sample first 10
    commands = parse_path(path_d)

    control_points = []
    actual_points = []

    current = None
    for cmd in commands:
        if cmd[0] == 'M':
            current = (cmd[1], cmd[2])
            control_points.append(current)
            actual_points.append(current)
        elif cmd[0] == 'L':
            current = (cmd[1], cmd[2])
            control_points.append(current)
            actual_points.append(current)
        elif cmd[0] == 'C':
            p0 = current
            p1 = (cmd[1], cmd[2])
            p2 = (cmd[3], cmd[4])
            p3 = (cmd[5], cmd[6])

            control_points.extend([p1, p2, p3])
            actual_points.extend(cubic_bezier_extrema(p0, p1, p2, p3))
            current = p3
        elif cmd[0] == 'Z':
            pass

# Compute overall bbox reduction with tighter bezier bounds
# Now do full analysis
control_min_x = float('inf')
control_min_y = float('inf')
control_max_x = float('-inf')
control_max_y = float('-inf')

actual_min_x = float('inf')
actual_min_y = float('inf')
actual_max_x = float('-inf')
actual_max_y = float('-inf')

for path_d in all_paths:
    commands = parse_path(path_d)

    current = None
    for cmd in commands:
        if cmd[0] == 'M':
            current = (cmd[1], cmd[2])
            control_min_x = min(control_min_x, current[0])
            control_min_y = min(control_min_y, current[1])
            control_max_x = max(control_max_x, current[0])
            control_max_y = max(control_max_y, current[1])
            actual_min_x = min(actual_min_x, current[0])
            actual_min_y = min(actual_min_y, current[1])
            actual_max_x = max(actual_max_x, current[0])
            actual_max_y = max(actual_max_y, current[1])
        elif cmd[0] == 'L':
            current = (cmd[1], cmd[2])
            control_min_x = min(control_min_x, current[0])
            control_min_y = min(control_min_y, current[1])
            control_max_x = max(control_max_x, current[0])
            control_max_y = max(control_max_y, current[1])
            actual_min_x = min(actual_min_x, current[0])
            actual_min_y = min(actual_min_y, current[1])
            actual_max_x = max(actual_max_x, current[0])
            actual_max_y = max(actual_max_y, current[1])
        elif cmd[0] == 'C':
            p0 = current
            p1 = (cmd[1], cmd[2])
            p2 = (cmd[3], cmd[4])
            p3 = (cmd[5], cmd[6])

            # Control point bbox
            for p in [p1, p2, p3]:
                control_min_x = min(control_min_x, p[0])
                control_min_y = min(control_min_y, p[1])
                control_max_x = max(control_max_x, p[0])
                control_max_y = max(control_max_y, p[1])

            # Actual curve bbox
            extrema = cubic_bezier_extrema(p0, p1, p2, p3)
            for p in extrema:
                actual_min_x = min(actual_min_x, p[0])
                actual_min_y = min(actual_min_y, p[1])
                actual_max_x = max(actual_max_x, p[0])
                actual_max_y = max(actual_max_y, p[1])

            current = p3
        elif cmd[0] == 'Z':
            pass

print(f"\nControl point bbox (current): ({control_min_x:.2f}, {control_min_y:.2f}) to ({control_max_x:.2f}, {control_max_y:.2f})")
print(f"Actual curve bbox (tight): ({actual_min_x:.2f}, {actual_min_y:.2f}) to ({actual_max_x:.2f}, {actual_max_y:.2f})")

# Note: these are in local coords, tiger has translate(200,200) transform
# Apply translate(200,200)
print(f"\nWith translate(200,200):")
print(f"Control: ({control_min_x+200:.2f}, {control_min_y+200:.2f}) to ({control_max_x+200:.2f}, {control_max_y+200:.2f})")
print(f"Actual:  ({actual_min_x+200:.2f}, {actual_min_y+200:.2f}) to ({actual_max_x+200:.2f}, {actual_max_y+200:.2f})")

print(f"\nDocument size from control points: {control_max_x+200:.2f} x {control_max_y+200:.2f}")
print(f"Document size from actual curves:  {actual_max_x+200:.2f} x {actual_max_y+200:.2f}")
print(f"resvg size:                         510.00 x 565.00")
print(f"\nPotential savings: {control_max_x - actual_max_x:.2f} x {control_max_y - actual_max_y:.2f}")
