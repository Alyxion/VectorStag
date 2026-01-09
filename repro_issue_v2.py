
import os
import sys
import numpy as np
from PIL import Image

# Ensure we can import vectorstag
sys.path.insert(0, os.getcwd())

try:
    from vectorstag.rust_renderer import RustSVGRenderer
except ImportError:
    print("Could not import vectorstag. Make sure you are in the project root.")
    sys.exit(1)

def render_svg(path, output_path, width=None, height=None):
    print(f"Rendering {path} to {output_path}")
    # Use transparent background to allow bbox diagnosis
    renderer = RustSVGRenderer(background=(0, 0, 0, 0))
    if width and height:
        img = renderer.render_file(path, width=width, height=height)
    else:
        img = renderer.render_file(path)
    
    if img:
        img.save(output_path)
        print(f"Saved {output_path}")
        return img
    else:
        print(f"Failed to render {path}")
        return None

def main():
    base_dir = os.getcwd()
    
    files = [
        {
            "name": "IL",
            "path": "SciStagEssentialData/images/noto/flags/svg/IL.svg",
            "size": 400
        },
        {
            "name": "NP",
            "path": "SciStagEssentialData/images/noto/flags/svg/NP.svg",
            "size": 400
        },
        {
            "name": "Android",
            "path": "samples/svg/android.svg",
            "size": 400
        }
    ]
    
    os.makedirs("repro_output", exist_ok=True)
    
    for item in files:
        full_path = os.path.join(base_dir, item["path"])
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
            
        output_path = os.path.join("repro_output", f"{item['name']}.png")
        render_svg(full_path, output_path, width=item["size"], height=item["size"])

if __name__ == "__main__":
    main()
