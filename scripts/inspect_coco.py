#!/usr/bin/env python3
"""
inspecter_coco.py

Verification and visualization tool for the COCO patch dataset.

What it does:
  1. Prints basic statistics (image count, annotation count, avg ann/patch).
  2. Checks that every image file_name listed in the JSON actually exists
     on disk (relative to DATA_ROOT).
  3. Picks a random patch (or a specific one with --image_id), reads the
     GeoTIFF patch from disk, draws all polygon annotations and their
     bounding boxes, and saves a PNG for visual inspection.

Usage:
  /mnt/DATA/IMANE/p3/bin/python inspecter_coco.py \
      --json /home/imane/DATA/AI4SmallFarms/sentinel-2-asia/output_coco_64/val_coco_64.json \
      --root /home/imane/DATA/AI4SmallFarms/sentinel-2-asia \
      --out  verification_patch.png

Optional:
  --image_id id   # inspect a specific image id instead of a random one
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


# ---------------------------------------------------------------------------
# 1) Statistics
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
# 2) Check that all image files exist on disk
# ---------------------------------------------------------------------------

def check_files(coco: dict, root: str):
    print(f"\n{'='*50}")
    print(f"File existence check")
    print(f"{'='*50}")
    missing = []
    for img in coco["images"]:
        full = os.path.join(root, img["file_name"])
        if not os.path.isfile(full):
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
# 3) Visualize one patch
# ---------------------------------------------------------------------------

def visualize_patch(coco: dict, root: str, image_id: int | None, out_path: str):
    """
    Load a patch GeoTIFF, overlay polygon annotations and bboxes, save PNG.
    """
    # --- select image record ---
    id_to_img = {img["id"]: img for img in coco["images"]}

    if image_id is not None:
        if image_id not in id_to_img:
            print(f"ERROR: image_id {image_id} not found in JSON.")
            sys.exit(1)
    else:
        image_id = random.choice(list(id_to_img.keys()))

    img_rec = id_to_img[image_id]
    print(f"\n{'='*50}")
    print(f"Visualizing image_id={image_id}")
    print(f"  file_name : {img_rec['file_name']}")

    # --- load annotations for this image ---
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    print(f"  annotations: {len(anns)}")

    # --- read patch from disk ---
    patch_path = os.path.join(root, img_rec["file_name"])
    if not os.path.isfile(patch_path):
        print(f"ERROR: patch file not found: {patch_path}")
        sys.exit(1)

    with rasterio.open(patch_path) as src:
        data = src.read()   # shape: (bands, H, W)

    # Display: use bands 3,2,1 (Sentinel-2 RGB) if 4 bands, else first 3
    if data.shape[0] >= 3:
        rgb = data[[2, 1, 0], :, :]   # bands are 0-indexed: B4, B3, B2
    else:
        rgb = data[[0, 0, 0], :, :]

    # Normalise to 0-1 for display (percentile stretch)
    def pct_stretch(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        if hi == lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)

    rgb = np.stack([pct_stretch(rgb[i]) for i in range(3)], axis=-1)  # (H,W,3)

    # --- plot ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(rgb)
    ax.set_title(
        f"image_id={image_id}  |  {len(anns)} annotations\n"
        f"{img_rec['file_name']}",
        fontsize=8
    )
    ax.axis("off")

    colors = plt.cm.Set1.colors   # distinct colors for polygons

    for i, ann in enumerate(anns):
        color = colors[i % len(colors)]

        # --- polygon ---
        for ring in ann["segmentation"]:
            # ring is a flat list [x0,y0, x1,y1, ...]
            pts = np.array(ring, dtype=np.float64).reshape(-1, 2)
            poly_patch = MplPolygon(
                pts, closed=True,
                linewidth=1.2, edgecolor=color,
                facecolor=(*color[:3], 0.15)   # transparent fill
            )
            ax.add_patch(poly_patch)

    # Legend
    legend_handles = [ mpatches.Patch(facecolor="none", edgecolor="gray", label="polygon")    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and visualize an AI4SmallFarms COCO patch JSON"
    )
    parser.add_argument(
        "--json", required=True,
        help="Path to the COCO JSON file (e.g. train_coco.json)"
    )
    parser.add_argument(
        "--root",
        default="/home/imane/DATA/AI4SmallFarms/sentinel-2-asia",
        help="Root directory where image file_names are relative to"
    )
    parser.add_argument(
        "--out", default="verification_patch.png",
        help="Output PNG path for the patch visualization"
    )
    parser.add_argument(
        "--image_id", type=int, default=None,
        help="Specific image id to visualize (random if omitted)"
    )
    args = parser.parse_args()

    # Load JSON
    print(f"Loading {args.json} ...")
    with open(args.json, encoding="utf-8") as f:
        coco = json.load(f)

    # 1) Statistics
    print_statistics(coco)

    # 2) File existence check
    check_files(coco, args.root)

    # 3) Visualization
    if not coco["images"]:
        print("No images in JSON — nothing to visualize.")
        return

    visualize_patch(coco, args.root, args.image_id, args.out)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
