#!/usr/bin/env python3
"""
inspect_coco.py

Verification and visualisation tool for the COCO patch dataset.

What it does:
  1. Prints basic statistics (image count, annotation count, avg ann/patch).
  2. Checks that every image file_name listed in the JSON exists on disk
     (relative to --root).
  3. Picks a random patch (or a specific one with --image_id), reads the
     GeoTIFF patch, draws all polygon annotations, and saves a PNG.

Usage:
  python scripts/inspect_coco.py \
      --json /path/to/sentinel-2-asia/output_coco_64/val_coco.json \
      --root /path/to/sentinel-2-asia \
      --out  verification_patch.png
"""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(coco: dict):
    n_images = len(coco["images"])
    n_anns   = len(coco["annotations"])
    print(f"\n{'='*50}")
    print(f"COCO dataset statistics")
    print(f"{'='*50}")
    print(f"  Images      : {n_images}")
    print(f"  Annotations : {n_anns}")
    if n_images:
        print(f"  Avg ann/img : {n_anns / n_images:.2f}")
    print(f"  Categories  : {[c['name'] for c in coco.get('categories', [])]}")


# ---------------------------------------------------------------------------
# File existence check
# ---------------------------------------------------------------------------

def check_files(coco: dict, root: str):
    print(f"\n{'='*50}")
    print(f"File existence check")
    print(f"{'='*50}")
    missing = []
    for img in coco["images"]:
        if not os.path.isfile(os.path.join(root, img["file_name"])):
            missing.append(img["file_name"])

    if missing:
        print(f"  MISSING files ({len(missing)} / {len(coco['images'])}):")
        for p in missing[:20]:
            print(f"    - {p}")
        if len(missing) > 20:
            print(f"    ... and {len(missing)-20} more")
    else:
        print(f"  All {len(coco['images'])} patch files found on disk. OK")
    return missing


# ---------------------------------------------------------------------------
# Patch visualisation
# ---------------------------------------------------------------------------

def visualize_patch(coco: dict, root: str, image_id: int | None, out_path: str):
    """Load one patch, overlay polygon annotations, save PNG."""
    id_to_img = {img["id"]: img for img in coco["images"]}

    if image_id is not None:
        if image_id not in id_to_img:
            print(f"ERROR: image_id {image_id} not found in JSON.")
            sys.exit(1)
    else:
        image_id = random.choice(list(id_to_img.keys()))

    img_rec = id_to_img[image_id]
    print(f"\n{'='*50}")
    print(f"Visualising image_id={image_id}")
    print(f"  file_name  : {img_rec['file_name']}")

    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    print(f"  annotations: {len(anns)}")

    patch_path = os.path.join(root, img_rec["file_name"])
    if not os.path.isfile(patch_path):
        print(f"ERROR: patch file not found: {patch_path}")
        sys.exit(1)

    with rasterio.open(patch_path) as src:
        data = src.read()   # (bands, H, W)

    # Use RGB bands if available; fall back to grayscale
    rgb = data[[2, 1, 0], :, :] if data.shape[0] >= 3 else data[[0, 0, 0], :, :]

    def pct_stretch(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        if hi == lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)

    rgb = np.stack([pct_stretch(rgb[i]) for i in range(3)], axis=-1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(rgb)
    ax.set_title(
        f"image_id={image_id}  |  {len(anns)} annotations\n{img_rec['file_name']}",
        fontsize=8
    )
    ax.axis("off")

    colors = plt.cm.Set1.colors
    for i, ann in enumerate(anns):
        color = colors[i % len(colors)]
        for ring in ann["segmentation"]:
            pts = np.array(ring, dtype=np.float64).reshape(-1, 2)
            ax.add_patch(MplPolygon(
                pts, closed=True,
                linewidth=1.2, edgecolor=color,
                facecolor=(*color[:3], 0.15)
            ))

    ax.legend(
        handles=[mpatches.Patch(facecolor="none", edgecolor="gray", label="polygon")],
        loc="lower right", fontsize=7
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualisation → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and visualise an AI4SmallFarms COCO patch JSON"
    )
    parser.add_argument("--json", required=True,
                        help="Path to the COCO JSON file (e.g. output_coco_64/val_coco.json)")
    parser.add_argument("--root", required=True,
                        help="Root directory where image file_names are relative to "
                             "(e.g. /path/to/sentinel-2-asia)")
    parser.add_argument("--out",  default="verification_patch.png",
                        help="Output PNG path (default: verification_patch.png)")
    parser.add_argument("--image_id", type=int, default=None,
                        help="Specific image id to visualise (random if omitted)")
    args = parser.parse_args()

    print(f"Loading {args.json} ...")
    with open(args.json, encoding="utf-8") as f:
        coco = json.load(f)

    print_statistics(coco)
    check_files(coco, args.root)

    if not coco["images"]:
        print("No images in JSON — nothing to visualise.")
        return

    visualize_patch(coco, args.root, args.image_id, args.out)
    print("\nInspection complete.")


if __name__ == "__main__":
    main()
