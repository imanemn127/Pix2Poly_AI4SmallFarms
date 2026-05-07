#!/usr/bin/env python3
"""
coco_to_gpkg.py

Convert COCO JSON pixel-space annotations back to geographic coordinates
and export them as a GeoPackage (.gpkg) for inspection in QGIS or similar
GIS tools.

Each annotation's pixel coordinates are converted to UTM map coordinates
using the per-image top_left, res_x, and res_y metadata stored by
build_coco_dataset.py.

Usage:
  python scripts/coco_to_gpkg.py \
      --json /path/to/output_coco_64/val_coco.json \
      --out  /path/to/val_annotations_geo.gpkg \
      --epsg 32648
"""

import argparse
import json

import geopandas as gpd
from shapely.geometry import Polygon


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO JSON annotations to a geographic GeoPackage"
    )
    parser.add_argument("--json", required=True,
                        help="Path to the COCO JSON file (e.g. output_coco_64/val_coco.json)")
    parser.add_argument("--out",  required=True,
                        help="Output GeoPackage path (e.g. val_annotations_geo.gpkg)")
    parser.add_argument("--epsg", type=int, default=32648,
                        help="EPSG code of the tile CRS "
                             "(default: 32648 = WGS84 / UTM zone 48N). "
                             "Adapt to match the CRS of your Sentinel-2 tiles.")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    rows = []
    for ann in coco["annotations"]:
        img      = images_by_id[ann["image_id"]]
        top_left = img["top_left"]   # [x, y] of the patch's top-left corner in UTM
        res_x    = img["res_x"]
        res_y    = img["res_y"]

        for seg in ann["segmentation"]:
            # Convert flat pixel list [x0,y0, x1,y1, ...] to UTM map coordinates.
            # The image Y-axis points downward, so map_y decreases with row index.
            coords = []
            for i in range(0, len(seg), 2):
                px = seg[i]
                py = seg[i + 1]
                map_x = top_left[0] + px * res_x
                map_y = top_left[1] - py * abs(res_y)
                coords.append((map_x, map_y))

            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty:
                rows.append({
                    "geometry":   poly,
                    "image_id":   ann["image_id"],
                    "feature_id": ann.get("feature_id", ann["id"]),
                })

    gdf = gpd.GeoDataFrame(rows, crs=f"EPSG:{args.epsg}")
    gdf.to_file(args.out, driver="GPKG")
    print(f"{len(gdf)} polygons written to {args.out}")


if __name__ == "__main__":
    main()
