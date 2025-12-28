"""SVG Path Parser - Parse path 'd' attribute into commands."""

import re
import math
from typing import Iterator


def tokenize_path(d: str) -> Iterator[str]:
    """Tokenize path data into commands and numbers."""
    # Pattern matches: commands, signed numbers (including scientific notation)
    pattern = r"([MmZzLlHhVvCcSsQqTtAa])|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"

    for match in re.finditer(pattern, d):
        cmd = match.group(1)
        num = match.group(2)
        if cmd:
            yield cmd
        elif num:
            yield num


def parse_path(d: str) -> list[tuple]:
    """
    Parse SVG path data into a list of absolute commands.

    Returns list of tuples where first element is command type:
    - ('M', x, y) - moveto
    - ('L', x, y) - lineto
    - ('H', x) - horizontal lineto
    - ('V', y) - vertical lineto
    - ('C', x1, y1, x2, y2, x, y) - cubic bezier
    - ('S', x2, y2, x, y) - smooth cubic bezier
    - ('Q', x1, y1, x, y) - quadratic bezier
    - ('T', x, y) - smooth quadratic bezier
    - ('A', rx, ry, x_rotation, large_arc, sweep, x, y) - arc
    - ('Z',) - closepath
    """
    commands = []
    tokens = list(tokenize_path(d))

    if not tokens:
        return commands

    i = 0
    current_x, current_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0  # Start of current subpath
    last_cmd = None
    last_control = None  # For smooth curves

    def get_nums(count: int) -> list[float]:
        nonlocal i
        nums = []
        while len(nums) < count and i < len(tokens):
            try:
                nums.append(float(tokens[i]))
                i += 1
            except ValueError:
                break
        return nums

    while i < len(tokens):
        token = tokens[i]

        # Check if it's a command
        if token in "MmZzLlHhVvCcSsQqTtAa":
            cmd = token
            i += 1
        else:
            # Implicit command - repeat last command
            if last_cmd in ('M', 'm'):
                cmd = 'L' if last_cmd == 'M' else 'l'
            elif last_cmd:
                cmd = last_cmd
            else:
                i += 1
                continue

        is_relative = cmd.islower()
        cmd_upper = cmd.upper()

        if cmd_upper == 'M':
            # Moveto
            nums = get_nums(2)
            if len(nums) < 2:
                continue

            x, y = nums[0], nums[1]
            if is_relative:
                x += current_x
                y += current_y

            commands.append(('M', x, y))
            current_x, current_y = x, y
            start_x, start_y = x, y
            last_control = None

            # Additional coordinate pairs are treated as lineto
            while True:
                nums = get_nums(2)
                if len(nums) < 2:
                    break
                x, y = nums[0], nums[1]
                if is_relative:
                    x += current_x
                    y += current_y
                commands.append(('L', x, y))
                current_x, current_y = x, y

        elif cmd_upper == 'Z':
            # Closepath
            commands.append(('Z',))
            current_x, current_y = start_x, start_y
            last_control = None

        elif cmd_upper == 'L':
            # Lineto
            while True:
                nums = get_nums(2)
                if len(nums) < 2:
                    break
                x, y = nums[0], nums[1]
                if is_relative:
                    x += current_x
                    y += current_y
                commands.append(('L', x, y))
                current_x, current_y = x, y
            last_control = None

        elif cmd_upper == 'H':
            # Horizontal lineto
            while True:
                nums = get_nums(1)
                if len(nums) < 1:
                    break
                x = nums[0]
                if is_relative:
                    x += current_x
                commands.append(('L', x, current_y))
                current_x = x
            last_control = None

        elif cmd_upper == 'V':
            # Vertical lineto
            while True:
                nums = get_nums(1)
                if len(nums) < 1:
                    break
                y = nums[0]
                if is_relative:
                    y += current_y
                commands.append(('L', current_x, y))
                current_y = y
            last_control = None

        elif cmd_upper == 'C':
            # Cubic bezier
            while True:
                nums = get_nums(6)
                if len(nums) < 6:
                    break
                x1, y1, x2, y2, x, y = nums
                if is_relative:
                    x1 += current_x
                    y1 += current_y
                    x2 += current_x
                    y2 += current_y
                    x += current_x
                    y += current_y
                commands.append(('C', x1, y1, x2, y2, x, y))
                last_control = (x2, y2)
                current_x, current_y = x, y

        elif cmd_upper == 'S':
            # Smooth cubic bezier
            while True:
                nums = get_nums(4)
                if len(nums) < 4:
                    break
                x2, y2, x, y = nums
                if is_relative:
                    x2 += current_x
                    y2 += current_y
                    x += current_x
                    y += current_y

                # Calculate first control point as reflection
                if last_control and last_cmd in ('C', 'c', 'S', 's'):
                    x1 = 2 * current_x - last_control[0]
                    y1 = 2 * current_y - last_control[1]
                else:
                    x1, y1 = current_x, current_y

                commands.append(('C', x1, y1, x2, y2, x, y))
                last_control = (x2, y2)
                current_x, current_y = x, y

        elif cmd_upper == 'Q':
            # Quadratic bezier
            while True:
                nums = get_nums(4)
                if len(nums) < 4:
                    break
                x1, y1, x, y = nums
                if is_relative:
                    x1 += current_x
                    y1 += current_y
                    x += current_x
                    y += current_y
                commands.append(('Q', x1, y1, x, y))
                last_control = (x1, y1)
                current_x, current_y = x, y

        elif cmd_upper == 'T':
            # Smooth quadratic bezier
            while True:
                nums = get_nums(2)
                if len(nums) < 2:
                    break
                x, y = nums
                if is_relative:
                    x += current_x
                    y += current_y

                # Calculate control point as reflection
                if last_control and last_cmd in ('Q', 'q', 'T', 't'):
                    x1 = 2 * current_x - last_control[0]
                    y1 = 2 * current_y - last_control[1]
                else:
                    x1, y1 = current_x, current_y

                commands.append(('Q', x1, y1, x, y))
                last_control = (x1, y1)
                current_x, current_y = x, y

        elif cmd_upper == 'A':
            # Arc
            while True:
                nums = get_nums(7)
                if len(nums) < 7:
                    break
                rx, ry, x_rot, large_arc, sweep, x, y = nums
                if is_relative:
                    x += current_x
                    y += current_y

                # Convert arc to bezier curves
                arc_commands = arc_to_bezier(
                    current_x, current_y,
                    rx, ry, x_rot,
                    int(large_arc), int(sweep),
                    x, y
                )
                commands.extend(arc_commands)
                current_x, current_y = x, y
            last_control = None

        last_cmd = cmd

    return commands


def arc_to_bezier(x1: float, y1: float, rx: float, ry: float,
                  phi: float, large_arc: int, sweep: int,
                  x2: float, y2: float) -> list[tuple]:
    """
    Convert an SVG arc to cubic bezier curves.

    This implements the SVG arc parameterization to endpoint parameterization
    conversion, then approximates the arc with cubic bezier curves.
    """
    commands = []

    # Handle degenerate cases
    if x1 == x2 and y1 == y2:
        return commands

    if rx == 0 or ry == 0:
        return [('L', x2, y2)]

    rx = abs(rx)
    ry = abs(ry)

    # Convert angle to radians
    phi_rad = math.radians(phi)
    cos_phi = math.cos(phi_rad)
    sin_phi = math.sin(phi_rad)

    # Step 1: Compute (x1', y1')
    dx = (x1 - x2) / 2
    dy = (y1 - y2) / 2
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # Correct radii if too small
    lambda_sq = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lambda_sq > 1:
        lambda_val = math.sqrt(lambda_sq)
        rx *= lambda_val
        ry *= lambda_val

    # Step 2: Compute (cx', cy')
    sq = max(0, (rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p) /
             (rx * rx * y1p * y1p + ry * ry * x1p * x1p))
    sq = math.sqrt(sq)

    if large_arc == sweep:
        sq = -sq

    cxp = sq * rx * y1p / ry
    cyp = -sq * ry * x1p / rx

    # Step 3: Compute (cx, cy)
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2

    # Step 4: Compute theta1 and dtheta
    def angle(ux, uy, vx, vy):
        n = math.sqrt(ux * ux + uy * uy) * math.sqrt(vx * vx + vy * vy)
        if n == 0:
            return 0
        c = (ux * vx + uy * vy) / n
        c = max(-1, min(1, c))
        sign = 1 if ux * vy - uy * vx >= 0 else -1
        return sign * math.acos(c)

    theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                   (-x1p - cxp) / rx, (-y1p - cyp) / ry)

    if sweep == 0 and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep == 1 and dtheta < 0:
        dtheta += 2 * math.pi

    # Split arc into segments of at most 90 degrees
    n_segs = max(1, int(math.ceil(abs(dtheta) / (math.pi / 2))))
    d_theta = dtheta / n_segs

    # Approximate each segment with a cubic bezier
    t = theta1
    for _ in range(n_segs):
        t2 = t + d_theta

        # Control point distance
        alpha = math.sin(d_theta) * (math.sqrt(4 + 3 * math.tan(d_theta / 2) ** 2) - 1) / 3

        # Start point
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        x_start = cx + rx * cos_phi * cos_t - ry * sin_phi * sin_t
        y_start = cy + rx * sin_phi * cos_t + ry * cos_phi * sin_t

        # End point
        cos_t2 = math.cos(t2)
        sin_t2 = math.sin(t2)
        x_end = cx + rx * cos_phi * cos_t2 - ry * sin_phi * sin_t2
        y_end = cy + rx * sin_phi * cos_t2 + ry * cos_phi * sin_t2

        # Derivatives
        dx_start = -rx * cos_phi * sin_t - ry * sin_phi * cos_t
        dy_start = -rx * sin_phi * sin_t + ry * cos_phi * cos_t
        dx_end = -rx * cos_phi * sin_t2 - ry * sin_phi * cos_t2
        dy_end = -rx * sin_phi * sin_t2 + ry * cos_phi * cos_t2

        # Control points
        cp1x = x_start + alpha * dx_start
        cp1y = y_start + alpha * dy_start
        cp2x = x_end - alpha * dx_end
        cp2y = y_end - alpha * dy_end

        commands.append(('C', cp1x, cp1y, cp2x, cp2y, x_end, y_end))

        t = t2

    return commands
