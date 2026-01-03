
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
import time
import traceback

# Add project root to path
sys.path.insert(0, os.getcwd())

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
    
    # Categories to check
    categories = [
        "feColorMatrix",
        "feComponentTransfer", 
        "feComposite",
        "feBlend",
        "feFlood",
        "feGaussianBlur",
        "feMerge",
        "feOffset",
    ]
    
    summary = {}
    
    for cat in categories:
        cat_dir = test_dir / cat
        if not cat_dir.exists():
            continue
            
        print(f"\nChecking {cat}...")
        files = sorted(cat_dir.glob("*.svg"))
        passed = 0
        total = 0
        
        for svg_path in files:
            # Skip known problematic files
            if "huge" in svg_path.name: continue
            
            ref_path = svg_path.with_suffix('.png')
            if not ref_path.exists(): continue
            
            total += 1
            try:
                ref_img = Image.open(ref_path).convert('RGBA')
                w, h = ref_img.size
                
                # Check if we can parse the file first to detect panics/crashes in rendering vs loading
                with open(svg_path, 'r') as f:
                    content = f.read()

                start = time.time()
                img = renderer.render(content, w, h)
                dur = (time.time() - start) * 1000
                
                if img is None:
                    print(f"  FAILED to render: {svg_path.name}")
                    continue
                    
                sim = compute_similarity(img, ref_img)
                
                if sim < 95.0:
                    print(f"  {svg_path.name}: {sim:.1f}% ({dur:.1f}ms)")
                else:
                    passed += 1
                    
            except Exception as e:
                print(f"  ERROR on {svg_path.name}: {e}")
                # traceback.print_exc()
        
        summary[cat] = (passed, total)
        print(f"  {cat}: {passed}/{total} passed")

    print("\nSummary:")
    for cat, (passed, total) in summary.items():
        if total > 0:
            print(f"{cat}: {passed}/{total} ({100*passed/total:.1f}%)")

if __name__ == "__main__":
    main()
