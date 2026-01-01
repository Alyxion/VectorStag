"""2D Affine transformation matrix for SVG transforms."""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Transform:
    """Represents a 2D affine transformation matrix.

    Matrix form:
    | a  c  e |
    | b  d  f |
    | 0  0  1 |
    """
    a: float = 1.0  # scale x
    b: float = 0.0  # skew y
    c: float = 0.0  # skew x
    d: float = 1.0  # scale y
    e: float = 0.0  # translate x
    f: float = 0.0  # translate y

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """Apply transformation to a point."""
        return (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f
        )

    def multiply(self, other: "Transform") -> "Transform":
        """Multiply this transform by another (self * other)."""
        return Transform(
            a=self.a * other.a + self.c * other.b,
            b=self.b * other.a + self.d * other.b,
            c=self.a * other.c + self.c * other.d,
            d=self.b * other.c + self.d * other.d,
            e=self.a * other.e + self.c * other.f + self.e,
            f=self.b * other.e + self.d * other.f + self.f
        )

    @classmethod
    def identity(cls) -> "Transform":
        """Create identity transform (no change)."""
        return cls()

    @classmethod
    def translate(cls, tx: float, ty: float = 0) -> "Transform":
        """Create translation transform."""
        return cls(e=tx, f=ty)

    @classmethod
    def scale(cls, sx: float, sy: Optional[float] = None) -> "Transform":
        """Create scaling transform."""
        if sy is None:
            sy = sx
        return cls(a=sx, d=sy)

    @classmethod
    def rotate(cls, angle: float, cx: float = 0, cy: float = 0) -> "Transform":
        """Rotate by angle (degrees) around point (cx, cy)."""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        if cx == 0 and cy == 0:
            return cls(a=cos_a, b=sin_a, c=-sin_a, d=cos_a)
        # Translate to origin, rotate, translate back
        t1 = cls.translate(-cx, -cy)
        r = cls(a=cos_a, b=sin_a, c=-sin_a, d=cos_a)
        t2 = cls.translate(cx, cy)
        return t2.multiply(r.multiply(t1))

    @classmethod
    def skewX(cls, angle: float) -> "Transform":
        """Create horizontal skew transform."""
        return cls(c=math.tan(math.radians(angle)))

    @classmethod
    def skewY(cls, angle: float) -> "Transform":
        """Create vertical skew transform."""
        return cls(b=math.tan(math.radians(angle)))

    @classmethod
    def matrix(cls, a: float, b: float, c: float, d: float, e: float, f: float) -> "Transform":
        """Create transform from matrix values."""
        return cls(a=a, b=b, c=c, d=d, e=e, f=f)
