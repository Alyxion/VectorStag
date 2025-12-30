#!/usr/bin/env python3
"""Quick test for key filter categories."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
from PIL import Image
import numpy as np
from vectorstag import SVGRenderer

def compute_similarity(img1, img2):
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.LANCZOS)
    arr1 = np.array(img1.convert('RGBA'), dtype=np.float32)
    arr2 = np.array(img2.convert('RGBA'), dtype=np.float32)
    alpha1, alpha2 = arr1[:,:,3], arr2[:,:,3]
    visible = (alpha1 > 0) | (alpha2 > 0)
    if not visible.any():
        return 100.0
    rgb_diff = np.abs(arr1[:,:,:3] - arr2[:,:,:3])
    rgb_diff_masked = np.where(visible[:,:,np.newaxis], rgb_diff, 0)
    alpha_diff = np.abs(alpha1 - alpha2)
    visible_count = visible.sum()
    rgb_mae = rgb_diff_masked.sum() / (visible_count * 3) if visible_count > 0 else 0
    alpha_mae = alpha_diff.mean()
    combined_mae = rgb_mae * 0.75 + alpha_mae * 0.25
    return max(0, 100 * (1 - combined_mae / 255))

renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
filter_dir = Path('resvg-test-suite/tests/filters')

categories = ['feBlend', 'feComposite', 'feMerge', 'feOffset', 'feFlood',
              'feGaussianBlur', 'feColorMatrix', 'feComponentTransfer',
              'feMorphology', 'feConvolveMatrix', 'feTurbulence']

total_scores = []
for category in categories:
    cat_dir = filter_dir / category
    if not cat_dir.exists():
        continue
    tests = list(sorted(cat_dir.glob('*.svg')))
    scores = []
    errors = []
    for svg in tests:
        ref = svg.with_suffix('.png')
        if not ref.exists():
            continue
        try:
            ref_img = Image.open(ref).convert('RGBA')
            vs_img = renderer.render_file(str(svg), ref_img.width, ref_img.height)
            if vs_img:
                score = compute_similarity(vs_img, ref_img)
                scores.append(score)
                total_scores.append(score)
                if score < 80:
                    errors.append((svg.stem, score))
        except Exception as e:
            errors.append((svg.stem, str(e)[:30]))
    if scores:
        print(f'{category}: {np.mean(scores):.1f}% ({len(scores)} tests)')
        for name, score in errors[:3]:
            if isinstance(score, float):
                print(f'  {score:.1f}%: {name}')
            else:
                print(f'  ERROR: {name} - {score}')

print(f'\nOVERALL FILTER AVERAGE: {np.mean(total_scores):.1f}%')
