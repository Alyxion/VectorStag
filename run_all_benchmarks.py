#!/usr/bin/env python3
"""
CI-style wrapper to run VectorStag comparisons and Big-4 matrix summaries.

Runs per-collection comparisons against references and emits a consolidated
pairwise similarity matrix across VectorStag, resvg, Cairo, and Chrome.

Configuration via env vars:
- VEC_WORKERS: number of workers (default: min(cpu_count(), 8))
- VEC_LIMIT: limit files per collection (default: no limit)
- VEC_SAVE_MATRIX_TOP: save top-N worst grids per collection (optional)
- VEC_SAVE_MATRIX_ALL: if set to "1", save grids for all files (careful: large)
- VEC_SAVE_DIR: override output dir for matrix grids
"""

import os
from pathlib import Path
from multiprocessing import cpu_count

import svg_compare as sc


def main():
    workers = int(os.environ.get("VEC_WORKERS", max(1, min(cpu_count(), 8))))
    limit_env = os.environ.get("VEC_LIMIT")
    limit = int(limit_env) if limit_env else None
    save_top_env = os.environ.get("VEC_SAVE_MATRIX_TOP")
    save_top = int(save_top_env) if save_top_env else None
    save_all = os.environ.get("VEC_SAVE_MATRIX_ALL", "0") == "1"
    save_dir_env = os.environ.get("VEC_SAVE_DIR")
    labels_off = os.environ.get("VEC_LABELS_OFF", "0") == "1"

    collections = sc.get_collections()

    # Aggregated matrix across collections
    agg_sums = {}
    agg_counts = {}

    for name, col in collections.items():
        if not col.svg_dir.exists():
            print(f"Skipping {name}: {col.svg_dir} not found")
            continue

        print("\n" + "=" * 70)
        print(f"COMPARISON: {name.upper()}")
        print("=" * 70)
        results, errors, buckets = sc.compare_collection(col, num_workers=workers, save_grids=False, limit=limit)
        sc.print_summary(results, errors, buckets, name.upper())

        print("\n" + "=" * 70)
        print(f"MATRIX: {name.upper()}")
        print("=" * 70)
        matrix = sc.matrix_collection(col, num_workers=workers, limit=limit)
        sc.print_matrix(matrix, name.upper())

        # Aggregate
        for k, v in matrix["sums"].items():
            agg_sums[k] = agg_sums.get(k, 0.0) + v
        for k, v in matrix["counts"].items():
            agg_counts[k] = agg_counts.get(k, 0) + v

        # Optional grids per collection
        if save_all or save_top:
            per_file = matrix.get("per_file", [])
            if save_all:
                selected = per_file
            else:
                selected = sorted(per_file, key=lambda x: x.get("worst", 1.0))[: max(0, save_top)]

            out_dir = Path(save_dir_env) if save_dir_env else (col.output_dir / "matrix")
            print(f"\nSaving {len(selected)} grids to: {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for item in selected:
                try:
                    sc.build_and_save_grid(
                        item["svg_path"],
                        col.svg_dir,
                        col.ref_dir,
                        col.size,
                        out_dir / f"{item['name']}.png",
                        show_labels=not labels_off,
                    )
                    saved += 1
                except Exception:
                    pass
            print(f"Saved {saved}/{len(selected)} grids")

    # Print consolidated matrix
    print("\n" + "=" * 99)
    print("PAIRWISE SIMILARITY MATRIX - CONSOLIDATED")
    print("=" * 99)
    pairs = [
        ("vectorstag", "resvg"),
        ("vectorstag", "cairo"),
        ("vectorstag", "chrome"),
        ("resvg", "cairo"),
        ("resvg", "chrome"),
        ("cairo", "chrome"),
    ]
    print("\nPair                     |  Avg   |  Count")
    print("-" * 99)
    for p in pairs:
        if p in agg_counts and agg_counts[p] > 0:
            avg = agg_sums[p] / agg_counts[p]
            print(f"{p[0]:>10} vs {p[1]:<10} | {avg:6.1%} | {agg_counts[p]:6}")
        else:
            print(f"{p[0]:>10} vs {p[1]:<10} |   n/a  |      0")


if __name__ == "__main__":
    main()
