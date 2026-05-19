#!/usr/bin/env python3
"""
build_coco_dataset.py

Build a COCO-format dataset of 32×32 pixel patches from AI4SmallFarms
Sentinel-2 tiles (Asia subset), using reference polygon files (_areas.gpkg).

Output:
  sentinel-2-asia/output_coco_32/train_coco.json
  sentinel-2-asia/output_coco_32/val_coco.json
  sentinel-2-asia/output_coco_32/test_coco.json

Each patch image is stored as a GeoTIFF cropped from the original tile.
File names in the JSON follow the pattern:
  <split>/patches_32/<tile_id>_<row>_<col>.tif

Usage:
  /mnt/DATA/IMANE/p3/bin/python build_coco_dataset.py

Set TEST_LIMIT to a small number (e.g. 2) to process only the first N tiles
per split during development; set to None to process all tiles.
"""

import json
import os
import re
import sys
import argparse

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, Polygon, MultiPolygon
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration 
# ---------------------------------------------------------------------------

DATA_ROOT = "/home/imane/DATA/AI4SmallFarms/sentinel-2-asia"

PATCH_SIZE = 32        # patch size in pixels — 0.32 km × 0.32 km at 10 m/px
STRIDE     = 32        # stride = patch size → contiguous, no overlap
CATEGORY   = {"id": 1, "name": "field"}

# Set to a small integer (e.g. 2) to process only the first N tiles per split.
# Set to None to process every tile.
TEST_LIMIT = None

# Minimum polygon area (px²) to keep — filters out tiny slivers after clipping
MIN_AREA_PX = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_reference_file(tile_name: str, ref_dir: str) -> str | None:
    """
    Return the path to the *_areas.gpkg matching tile_name (e.g. '1_vietnam'),
    or None if not found.
    The reference files are named like '1_vietnam_areas.gpkg'.
    """
    expected = os.path.join(ref_dir, f"{tile_name}_areas.gpkg")
    if os.path.isfile(expected):
        return expected
    return None


def generate_patches(width: int, height: int, patch_size: int, stride: int):
    """
    Yield (col_off, row_off) for every complete patch that fits in the raster.
    Incomplete edge patches (smaller than patch_size) are discarded.
    """
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            yield col, row


def clip_polygons_to_patch(gdf: gpd.GeoDataFrame,
                            src: rasterio.DatasetReader,
                            col_off: int, row_off: int,
                            patch_size: int) -> list:
    """
    Return a list of shapely geometries (in the raster's CRS) that are the
    intersection of each reference polygon with the patch bounding box.

    Parameters
    ----------
    gdf       : GeoDataFrame of reference polygons (already in raster CRS)
    src       : open rasterio dataset
    col_off   : left column of the patch (pixel coords)
    row_off   : top row of the patch (pixel coords)
    patch_size: side length in pixels
    """
    transform = src.transform
    x_min = transform.c + col_off * transform.a
    y_max = transform.f + row_off * transform.e          # e is negative
    x_max = transform.c + (col_off + patch_size) * transform.a
    y_min = transform.f + (row_off + patch_size) * transform.e

    patch_box = box(x_min, y_min, x_max, y_max)

    mask = gdf.geometry.intersects(patch_box)
    clipped = []
    for geom in gdf.loc[mask, "geometry"]:
        inter = geom.intersection(patch_box)
        if inter.is_empty:
            continue
        if isinstance(inter, Polygon):
            clipped.append(inter)
        elif isinstance(inter, MultiPolygon):
            clipped.extend(inter.geoms)
    return clipped


def polygon_to_pixel_coords(polygon: Polygon,
                             src: rasterio.DatasetReader,
                             col_off: int, row_off: int) -> list[float]:
    """
    Convert a Shapely Polygon (map CRS) into a flat list of pixel coordinates
    [x0,y0, x1,y1, ...] in the LOCAL frame of the patch (origin = top-left).

    Only the exterior ring is returned (holes are ignored, fields have none).
    """
    transform = src.transform

    def map_to_local_px(x_map: float, y_map: float):
        col_global = (x_map - transform.c) / transform.a
        row_global = (y_map - transform.f) / transform.e
        return col_global - col_off, row_global - row_off   # local (x, y)

    coords = []
    for x_map, y_map in polygon.exterior.coords:
        lx, ly = map_to_local_px(x_map, y_map)
        coords.extend([lx, ly])

    # COCO convention: ring is closed and shapely returns a closed one
    return coords


def bbox_and_area_from_flat(flat: list[float]):
    """
    Compute COCO bbox [x, y, width, height] and polygon area (px²) from a
    flat coordinate list [x0,y0, x1,y1, ...].
    """
    if len(flat) < 6:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    pts = np.array(flat, dtype=np.float32).reshape(-1, 2)

    # Remove duplicate closing point if present
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    xs, ys = pts[:, 0], pts[:, 1]
    x_min, y_min = float(xs.min()), float(ys.min())
    w = float(xs.max()) - x_min
    h = float(ys.max()) - y_min

    # Shoelace formula
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = float(abs(area) / 2.0)

    return [x_min, y_min, w, h], area


def save_patch_image(src: rasterio.DatasetReader,
                     col_off: int, row_off: int, patch_size: int,
                     out_path: str):
    """
    Crop a patch from the open rasterio dataset and save it as a GeoTIFF.
    Only the RGB bands (B4, B3, B2) are saved, producing a 3-band image.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    window = Window(col_off, row_off, patch_size, patch_size)

    # Read only the visible bands: Red, Green, Blue
    data = src.read([4, 3, 2], window=window)   # shape (3, H, W)

    new_transform = src.window_transform(window)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=patch_size, width=patch_size,
        count=3,                   
        dtype=src.dtypes[0],
        crs=src.crs,
        transform=new_transform,
    ) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------

def process_tile(tile_path: str,
                 ref_path: str,
                 split: str,
                 tile_id: str,
                 patches_out_dir: str,
                 patch_size: int = PATCH_SIZE,
                 stride: int = STRIDE,
                 min_area: float = MIN_AREA_PX) -> list:
    """
    Process one tile: generate all patches, clip polygons, build COCO records.

    Returns a list of image record dicts, each containing pre-computed
    polygon data for build_coco_json() to consume.
    """
    image_records = []

    with rasterio.open(tile_path) as src:
        width, height = src.width, src.height
        transform = src.transform

        # Pixel resolution (always positive, as in P3 NY format)
        res_x = float(abs(transform.a))
        res_y = float(abs(transform.e))

        # Load and reproject reference polygons to raster CRS if needed
        gdf = gpd.read_file(ref_path)
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        gdf = gdf.copy()
        gdf.sindex  # build spatial index

        patch_list = list(generate_patches(width, height, patch_size, stride))

        for col_off, row_off in tqdm(patch_list,
                                     desc=f"  {tile_id}",
                                     leave=False,
                                     unit="patch"):
            # Relative file path stored in COCO (relative to DATA_ROOT)
            rel_path = os.path.join(
                split, "patches_32",
                f"{tile_id}_{row_off:05d}_{col_off:05d}.tif"
            )
            abs_path = os.path.join(DATA_ROOT, rel_path)

            # Geographic coordinates of the patch top-left corner
            # rasterio affine: (x, y) = transform * (col, row)
            tl_x = float(transform.c + col_off * transform.a)
            tl_y = float(transform.f + row_off * transform.e)
            top_left = [tl_x, tl_y]

            clipped = clip_polygons_to_patch(
                gdf, src, col_off, row_off, patch_size
            )

            seg_list, area_list, bbox_list = [], [], []
            for poly in clipped:
                flat = polygon_to_pixel_coords(poly, src, col_off, row_off)
                bbox, area = bbox_and_area_from_flat(flat)
                if area < min_area:
                    continue
                seg_list.append(flat)
                area_list.append(area)
                bbox_list.append(bbox)

            # Save patch image to disk
            save_patch_image(src, col_off, row_off, patch_size, abs_path)

            image_records.append({
                "file_name":  rel_path,
                "image_path": rel_path,   # same as file_name; required by p3_coco.py
                "width":      patch_size,
                "height":     patch_size,
                "res_x":      res_x,
                "res_y":      res_y,
                "top_left":   top_left,
                # polygon data — consumed by build_coco_json, not written to JSON
                "_seg_list":  seg_list,
                "_area_list": area_list,
                "_bbox_list": bbox_list,
            })

    return image_records


# ---------------------------------------------------------------------------
# COCO JSON builder
# ---------------------------------------------------------------------------

def build_coco_json(all_image_records: list,
                    category: dict,
                    image_id_start: int = 1,
                    ann_id_start: int = 1) -> dict:
    """
    Convert a flat list of image records into a COCO JSON dict whose structure
    matches the P3 NY dataset format expected by p3_coco.py / P3Dataset.

    Field order in each dict mirrors the NY JSON exactly (minus lidar_path).
    """
    coco = {
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "AI4SmallFarms Asia — COCO patches for Pix2Poly",
            "contributor": "",
            "url": "",
            "date_created": "",
        },
        "categories": [category],
        "images": [],
        "annotations": [],
    }

    ann_id = ann_id_start
    for img_id, rec in enumerate(all_image_records, start=image_id_start):
        # Image entry — key order matches P3 NY (id, file_name, image_path,
        # width, height, res_x, res_y, top_left).  lidar_path is intentionally
        # omitted since this dataset has no LiDAR.
        coco["images"].append({
            "id":         img_id,
            "file_name":  rec["file_name"],
            "image_path": rec["image_path"],
            "width":      rec["width"],
            "height":     rec["height"],
            "res_x":      round(rec["res_x"], 6),
            "res_y":      round(rec["res_y"], 6),
            "top_left":   [round(rec["top_left"][0], 6),
                           round(rec["top_left"][1], 6)],
        })

        for feature_id, (flat, area, bbox) in enumerate(zip(
                rec["_seg_list"], rec["_area_list"], rec["_bbox_list"])):
            coco["annotations"].append({
                "feature_id":   feature_id,   # local index 0,1,2… per image
                "id":           ann_id,
                "image_id":     img_id,
                "segmentation": [flat],
                "area":         round(area, 4),
                "bbox":         [round(v, 4) for v in bbox],
                "category_id":  category["id"],
            })
            ann_id += 1

    return coco


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    
    parser = argparse.ArgumentParser(description="Build COCO dataset")
    parser.add_argument("--split", choices=["train", "validate", "test", "all"], default="all",
                        help="Which split to process")
    args = parser.parse_args()

    ref_dir = os.path.join(DATA_ROOT, "reference")

    all_splits = {
    "train": os.path.join(DATA_ROOT, "train", "images"),
    "validate":   os.path.join(DATA_ROOT, "validate", "images"),
    "test":  os.path.join(DATA_ROOT, "test", "images"),
    }
    if args.split == "all":
        splits = all_splits
    else:
        splits = {args.split: all_splits[args.split]}

    # Output JSON names per split
    out_dir = os.path.join(DATA_ROOT, "output_coco_32")
    os.makedirs(out_dir, exist_ok=True)
    out_json = {
    "train":    os.path.join(out_dir, "train_coco.json"),
    "validate": os.path.join(out_dir, "val_coco.json"),
    "test":     os.path.join(out_dir, "test_coco.json"),
    }

    for split, images_dir in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing split: {split.upper()}  ({images_dir})")
        print(f"{'='*60}")

        tile_files = sorted(
            f for f in os.listdir(images_dir) if f.endswith(".tif")
        )

        if TEST_LIMIT is not None:
            tile_files = tile_files[:TEST_LIMIT]
            print(f"  [TEST_LIMIT={TEST_LIMIT}] Processing only: {tile_files}")

        all_records = []

        for tile_file in tile_files:
            tile_id  = tile_file.replace(".tif", "")
            tile_path = os.path.join(images_dir, tile_file)
            ref_path  = find_reference_file(tile_id, ref_dir)

            if ref_path is None:
                print(f"  WARNING: no reference file for {tile_id}, skipping.")
                continue

            print(f"\n  Tile: {tile_id}")
            print(f"    image  : {tile_path}")
            print(f"    ref    : {ref_path}")

            patches_out_dir = os.path.join(DATA_ROOT, split, "patches_32")
            records = process_tile(
                tile_path=tile_path,
                ref_path=ref_path,
                split=split,
                tile_id=tile_id,
                patches_out_dir=patches_out_dir,
            )
            print(f"    → {len(records)} patches generated")
            all_records.extend(records)

        coco = build_coco_json(all_records, CATEGORY)
        with open(out_json[split], "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        n_images = len(coco["images"])
        n_anns   = len(coco["annotations"])
        print(f"\n  Saved: {out_json[split]}")
        print(f"  Total images     : {n_images}")
        print(f"  Total annotations: {n_anns}")
        pct = n_anns / n_images if n_images else 0
        print(f"  Avg ann / patch  : {pct:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
