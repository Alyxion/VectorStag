
import sys
import os
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from svg_compare import create_comparison_grid
except ImportError:
    # If standard import fails, try to load the module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("svg_compare", "svg_compare.py")
    svg_compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svg_compare)
    create_comparison_grid = svg_compare.create_comparison_grid

def generate_grid(vs_path, ref_path, output_path, size=400):
    print(f"Generating grid for {vs_path}...")
    
    if not os.path.exists(vs_path):
        print(f"Error: VS image not found: {vs_path}")
        return

    # Load VectorStag image
    try:
        vs_img = Image.open(vs_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading VS image {vs_path}: {e}")
        return

    # Load Reference
    resvg_img = None
    if os.path.exists(ref_path):
        try:
            resvg_img = Image.open(ref_path).convert("RGBA")
            # Resize ref if needed to match requested size
            if resvg_img.size != (size, size):
                 resvg_img = resvg_img.resize((size, size), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading reference {ref_path}: {e}")
    else:
        print(f"Warning: Reference not found: {ref_path}")

    # Create Grid
    grid = create_comparison_grid(vs_img, resvg_img, size)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path)
    print(f"Saved grid to {output_path}")

def main():
    tasks = [
        {
            "vs": "repro_output/IL.png",
            "ref": "references/flags/resvg/IL.png",
            "out": "comparison_grids/IL_grid.png",
            "size": 400
        },
        {
            "vs": "repro_output/NP.png",
            "ref": "references/flags/resvg/NP.png",
            "out": "comparison_grids/NP_grid.png",
            "size": 400
        },
        {
            "vs": "repro_output/Android.png",
            "ref": "references/w3c/resvg/android.png",
            "out": "comparison_grids/Android_grid.png",
            "size": 400
        }
    ]

    for task in tasks:
        generate_grid(task["vs"], task["ref"], task["out"], task["size"])

if __name__ == "__main__":
    main()
