
import sys
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from vectorstag.rust_renderer import RustSVGRenderer

def test_normal():
    svg_path = Path("resvg-test-suite/tests/filters/feBlend/mode=normal.svg")
    if not svg_path.exists():
        print("Test file not found")
        return

    renderer = RustSVGRenderer(background=(0, 0, 0, 0), antialias=4)
    
    # Expected size from viewbox
    w, h = 200, 200
    
    img = renderer.render_file(str(svg_path), w, h)
    if img is None:
        print("Render failed")
        return
        
    img.save("debug_normal.png")
    print("Saved debug_normal.png")
    
    # Analyze center pixel (should be seagreen)
    # Seagreen: #2E8B57 -> (46, 139, 87)
    arr = np.array(img)
    center = arr[100, 100]
    print(f"Center pixel: {center}")
    
    expected = [46, 139, 87, 255]
    diff = np.abs(center - expected)
    print(f"Difference: {diff}")
    
    # Check if we have red (SourceGraphic) leaking
    # Red: (255, 0, 0)
    
    # Check edges (should be within filter region)
    # Filter region x=-10% -> -16px (relative to rect at 20) -> 4px
    # Check pixel at 10, 10 (inside region, outside rect)
    p10 = arr[10, 10]
    print(f"Pixel at 10,10 (in region, outside rect): {p10}")
    # Should be seagreen because feFlood fills region and SourceGraphic is transparent there
    # blend(green, transparent) = green
    
    # Check pixel at 2, 2 (outside region)
    p2 = arr[2, 2]
    print(f"Pixel at 2,2 (outside region): {p2}")
    # Should be transparent (clipped)

if __name__ == "__main__":
    test_normal()
