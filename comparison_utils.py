"""Shared utilities for generating comparison images."""

from PIL import Image
import numpy as np


def create_diff_image(img1: Image.Image, img2: Image.Image, size: int, threshold: int = 10) -> Image.Image:
    """Create a diff image highlighting differences in magenta."""
    s = (size, size)

    if img1 is None or img2 is None:
        return Image.new("RGB", s, (128, 128, 128))

    # Resize if needed
    if img1.size != s:
        img1 = img1.resize(s, Image.Resampling.LANCZOS)
    if img2.size != s:
        img2 = img2.resize(s, Image.Resampling.LANCZOS)

    # Composite on white
    white = Image.new("RGBA", s, (255, 255, 255, 255))
    img1_comp = Image.alpha_composite(white, img1)
    img2_comp = Image.alpha_composite(white, img2)

    # Convert to arrays
    arr1 = np.array(img1_comp, dtype=np.int16)[:, :, :3]
    arr2 = np.array(img2_comp, dtype=np.int16)[:, :, :3]

    # Compute difference
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff, axis=2)

    # Create output: show original image darkened, with differences in magenta
    base = np.array(img1_comp)[:, :, :3].astype(np.float32) * 0.3

    # Highlight differences in magenta (amplified)
    mask = max_diff > threshold
    diff_amplified = np.clip(max_diff * 3, 0, 255)

    result = base.copy()
    result[mask, 0] = np.clip(base[mask, 0] + diff_amplified[mask], 0, 255)
    result[mask, 1] = base[mask, 1] * 0.3
    result[mask, 2] = np.clip(base[mask, 2] + diff_amplified[mask], 0, 255)

    return Image.fromarray(result.astype(np.uint8), mode="RGB")


def create_comparison_grid(vs_img: Image.Image, resvg_img: Image.Image,
                          cairo_img: Image.Image = None, size: int = 400) -> Image.Image:
    """Create a comparison grid image.

    Layout: VectorStag | resvg | diff (VS vs resvg)
    All cells are size x size.

    Returns:
        RGB image of size (size*3, size)
    """
    white = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    grid = Image.new("RGB", (size * 3, size), (255, 255, 255))

    # Resize images to target size, preserving aspect ratio and centering
    def fit_image(img: Image.Image) -> Image.Image:
        if img is None:
            return Image.new("RGBA", (size, size), (200, 200, 200, 255))

        # Calculate scale to fit within size x size while preserving aspect ratio
        scale = min(size / img.width, size / img.height)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center on canvas
        canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        offset_x = (size - new_w) // 2
        offset_y = (size - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        return canvas

    vs_fitted = fit_image(vs_img)
    resvg_fitted = fit_image(resvg_img)

    # Composite on white and place in grid
    vs_comp = Image.alpha_composite(white, vs_fitted)
    resvg_comp = Image.alpha_composite(white, resvg_fitted)

    grid.paste(vs_comp.convert("RGB"), (0, 0))
    grid.paste(resvg_comp.convert("RGB"), (size, 0))

    # Create diff
    diff_img = create_diff_image(vs_fitted, resvg_fitted, size)
    grid.paste(diff_img, (size * 2, 0))

    return grid


def compute_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity between two images."""
    if img1 is None or img2 is None:
        return 0.0

    size = (max(img1.width, img2.width), max(img1.height, img2.height))
    if img1.size != size:
        img1 = img1.resize(size, Image.Resampling.LANCZOS)
    if img2.size != size:
        img2 = img2.resize(size, Image.Resampling.LANCZOS)

    white = Image.new("RGBA", size, (255, 255, 255, 255))
    img1_comp = Image.alpha_composite(white, img1)
    img2_comp = Image.alpha_composite(white, img2)

    arr1 = np.array(img1_comp, dtype=np.float32)[:, :, :3] / 255.0
    arr2 = np.array(img2_comp, dtype=np.float32)[:, :, :3] / 255.0

    mse = np.mean((arr1 - arr2) ** 2)
    return max(0.0, 1.0 - min(1.0, mse * 4))
