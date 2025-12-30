#!/usr/bin/env python3
"""Quick filter benchmark."""
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
    diff = np.abs(arr1 - arr2).mean()
    return max(0, 100 * (1 - diff / 255))

renderer = SVGRenderer(background=(0, 0, 0, 0), antialias=4)
filter_dir = Path('resvg-test-suite/tests/filters')

categories = sorted([d.name for d in filter_dir.iterdir() if d.is_dir()])
total_scores = []
total_errors = 0
total_tests = 0

for category in categories:
    cat_dir = filter_dir / category
    tests = list(sorted(cat_dir.glob('*.svg')))[:15]  # Limit tests per category
    scores = []
    errors = []
    for svg in tests:
        ref = svg.with_suffix('.png')
        if not ref.exists():
            continue
        total_tests += 1
        try:
            ref_img = Image.open(ref).convert('RGBA')
            vs_img = renderer.render_file(str(svg), ref_img.width, ref_img.height)
            if vs_img:
                score = compute_similarity(vs_img, ref_img)
                scores.append(score)
                total_scores.append(score)
            else:
                total_errors += 1
                errors.append(f'{svg.stem}: None')
        except Exception as e:
            total_errors += 1
            errors.append(f'{svg.stem}: {str(e)[:30]}')

    if scores:
        avg = np.mean(scores)
        low = len([s for s in scores if s < 80])
        print(f'{category}: {avg:.1f}% ({len(scores)} tests, {low} <80%)')
        if errors:
            for err in errors[:2]:
                print(f'  ERR: {err}')

print(f'\n=== OVERALL ===')
print(f'Tests: {total_tests}, Success: {len(total_scores)}, Errors: {total_errors}')
print(f'Average: {np.mean(total_scores):.1f}%')
print(f'>95%: {len([s for s in total_scores if s >= 95])} ({100*len([s for s in total_scores if s >= 95])/len(total_scores):.0f}%)')
print(f'>90%: {len([s for s in total_scores if s >= 90])} ({100*len([s for s in total_scores if s >= 90])/len(total_scores):.0f}%)')
