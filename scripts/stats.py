#!/usr/bin/env python3
"""
stats.py

Print basic statistics for a COCO JSON patch dataset:
  - Number of images
  - Number of annotations
  - Average fields per image
  - Average vertices per field
  - Average total vertices per image

Usage:
  python scripts/stats.py --json /path/to/output_coco_64/train_coco.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Print statistics for a COCO JSON patch dataset"
    )
    parser.add_argument("--json", required=True,
                        help="Path to the COCO JSON file (e.g. output_coco_64/train_coco.json)")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    images      = data["images"]
    annotations = data["annotations"]
    n_images    = len(images)
    n_anns      = len(annotations)

    total_verts = 0
    for ann in annotations:
        for seg in ann["segmentation"]:
            total_verts += len(seg) // 2   # flat list [x0,y0, x1,y1, ...]

    avg_fields = n_anns / n_images if n_images else 0
    avg_verts_field = total_verts / n_anns if n_anns else 0
    avg_verts_img   = total_verts / n_images if n_images else 0

    print(f"File             : {args.json}")
    print(f"Training images  : {n_images}")
    print(f"Total annotations: {n_anns}")
    print(f"Avg fields/image : {avg_fields:.1f}")
    print(f"Avg vertices/field: {avg_verts_field:.1f}")
    print(f"Avg vertices/image: {avg_verts_img:.0f}")


if __name__ == "__main__":
    main()
