"""Type definitions for VectorStag."""

from typing import Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

# SVG input types
SVGInput = Union[str, bytes, Path]

# Render target types (PIL Image or numpy array)
RenderTarget = Union["Image.Image", "np.ndarray"]
