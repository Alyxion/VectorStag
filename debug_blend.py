
import sys
import os
import numpy as np
from PIL import Image
from vectorstag.rust_renderer import fe_flood, fe_blend

def test_blend():
    width = 100
    height = 100
    
    # Create Red background (SourceGraphic)
    bg = np.zeros((height, width, 4), dtype=np.uint8)
    bg[:, :] = [255, 0, 0, 255]
    
    # Create Seagreen foreground (Flood)
    fg = fe_flood(width, height, 46, 139, 87, 255)
    
    # Blend Normal
    # mode 0 = normal
    result = fe_blend(fg, bg, 0)
    
    # Check center pixel
    center = result[50, 50]
    print(f"Result pixel: {center}")
    
    expected = np.array([46, 139, 87, 255], dtype=np.uint8)
    if np.allclose(center, expected):
        print("PASS: Seagreen covers Red")
    else:
        print("FAIL: Expected Seagreen")

if __name__ == "__main__":
    test_blend()
