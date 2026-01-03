
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.getcwd())

from vectorstag import SVGRenderer
from vectorstag.rust_renderer import RustSVGRenderer

def compute_similarity(img1, img2):
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.LANCZOS)
    
    arr1 = np.array(img1.convert('RGBA'), dtype=np.float32)
    arr2 = np.array(img2.convert('RGBA'), dtype=np.float32)
    
    # Simple similarity check
    diff = np.abs(arr1 - arr2).mean()
    return max(0, 100 * (1 - diff / 255))

def main():
    test_dir = Path("resvg-test-suite/tests/filters")
    if not test_dir.exists():
        print("resvg tests not found")
        return

    renderer = RustSVGRenderer(background=(0, 0, 0, 0), antialias=4)
    
    failures = []
    
    print("Scanning for failures...")
    for svg_path in sorted(test_dir.rglob("*.svg")):
        if "feMorphology" in str(svg_path): continue # Skip slow ones
        
        ref_path = svg_path.with_suffix('.png')
        if not ref_path.exists(): continue
        
        try:
            ref_img = Image.open(ref_path).convert('RGBA')
            w, h = ref_img.size
            
            img = renderer.render_file(str(svg_path), w, h)
            if img is None:
                print(f"FAILED to render: {svg_path.name}")
                continue
                
            sim = compute_similarity(img, ref_img)
            
            if sim < 95.0:
                print(f"{svg_path.name}: {sim:.1f}%")
                failures.append((svg_path, sim))
                if len(failures) >= 20:
                    break
                    
        except Exception as e:
            print(f"Error on {svg_path.name}: {e}")

    print("\nFirst 20 failures analysis:")
    for path, sim in failures:
        # Read the SVG content to see what it's testing
        with open(path, 'r') as f:
            content = f.read()
            tags = []
            if "primitiveUnits=\"objectBoundingBox\"" in content:
                tags.append("primitiveUnits=bbox")
            if "filterUnits=\"userSpaceOnUse\"" in content:
                tags.append("filterUnits=user")
            
            print(f"  {path.name} ({sim:.1f}%) - {', '.join(tags)}")

if __name__ == "__main__":
    main()
