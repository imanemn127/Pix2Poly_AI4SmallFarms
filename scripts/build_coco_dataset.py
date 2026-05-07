#!/usr/bin/env python3
"""
build_coco_dataset.py

Build a COCO-format dataset of 64×64 pixel patches from AI4SmallFarms
Sentinel-2 tiles (Asia subset), using reference polygon files (_areas.gpkg).

Output (written under DATA_ROOT/output_coco_64/):
  train_coco.json
  val_coco.json
  test_coco.json

Each patch image is stored as a 3-band (RGB) GeoTIFF cropped from the
original tile.  File names in the JSON follow the pattern:
  <split>/patches_64/<tile_id>_<row>_<col>.tif

Usage:
  python scripts/build_coco_dataset.py --data_root /path/to/sentinel-2-asia
  python scripts/build_coco_dataset.py --data_root /path/to/sentinel-2-asia --split train

Set TEST_LIMIT to a small integer (e.g. 2) to process only the first N tiles
per split during development; set to None to process all tiles.
"""

import argparse
import json
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, Polygon, MultiPolygon
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

PATCH_SIZE  = 64        # patch size in pixels — 0.64 km × 0.64 km at 10 m/px
STRIDE      = 64        # stride = patch size → contiguous, no overlap
CATEGORY    = {"id": 1, "name": "field"}
TEST_LIMIT  = None      # set to a small int to process only the first N tiles
MIN_AREA_PX = 16        # minimum polygon area (px²) after clipping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_reference_file(tile_name: str, ref_dir: str) -> str | None:
    """Return the path to <tile_name>_areas.gpkg, or None if not found."""
    expected = os.path.join(ref_dir, f"{tile_name}_areas.gpkg")
    return expected if os.path.isfile(expected) else None


def generate_patches(width: int, height: int, patch_size: int, stride: int):
    """Yield (col_off, row_off) for every complete patch that fits in the raster."""
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            yield col, row


def clip_polygons_to_patch(gdf, src, col_off, row_off, patch_size):
    """
    Return a list of Shapely geometries clipped to the patch bounding box.
    """
    t = src.transform
    x_min = t.c + col_off * t.a
    y_max = t.f + row_off * t.e        # t.e is negative
    x_max = t.c + (col_off + patch_size) * t.a
    y_min = t.f + (row_off + patch_size) * t.e

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


def polygon_to_pixel_coords(polygon, src, col_off, row_off):
    """
    Convert a Shapely Polygon (map CRS) to a flat pixel coordinate list
    [x0,y0, x1,y1, ...] in the local frame of the patch (origin = top-left).
    Only the exterior ring is used (fields have no holes).
    """
    t = src.transform

    def to_local(x_map, y_map):
        col_g = (x_map - t.c) / t.a
        row_g = (y_map - t.f) / t.e
        return col_g - col_off, row_g - row_off

    coords = []
    for x_map, y_map in polygon.exterior.coords:
        lx, ly = to_local(x_map, y_map)
        coords.extend([lx, ly])
    return coords


def bbox_and_area_from_flat(flat):
    """Compute COCO bbox [x, y, w, h] and area (px²) from a flat coordinate list."""
    if len(flat) < 6:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    pts = np.array(flat, dtype=np.float64).reshape(-1, 2)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    xs, ys = pts[:, 0], pts[:, 1]
    x_min, y_min = float(xs.min()), float(ys.min())
    w = float(xs.max()) - x_min
    h = float(ys.max()) - y_min

    # Shoelace formula for polygon area
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0

    return [x_min, y_min, w, h], area


def save_patch_image(src, col_off, row_off, patch_size, out_path):
    """Crop a 3-band RGB patch from the raster and save as a GeoTIFF."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    window = Window(col_off, row_off, patch_size, patch_size)
    data = src.read([4, 3, 2], window=window)   # Red, Green, Blue bands
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

def process_tile(tile_path, ref_path, split, tile_id, data_root,
                 patch_size=PATCH_SIZE, stride=STRIDE, min_area=MIN_AREA_PX):
    """
    Process one tile: generate all patches, clip polygons, build COCO records.
    Returns a list of image record dicts for build_coco_json() to consume.
    """
    image_records = []

    with rasterio.open(tile_path) as src:
        width, height = src.width, src.height
        t = src.transform
        res_x = abs(t.a)
        res_y = abs(t.e)

        gdf = gpd.read_file(ref_path)
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        gdf = gdf.copy()
        gdf.sindex  # build spatial index

        for col_off, row_off in tqdm(
            list(generate_patches(width, height, patch_size, stride)),
            desc=f"  {tile_id}", leave=False, unit="patch"
        ):
            rel_path = os.path.join(
                split, f"patches_{patch_size}",
                f"{tile_id}_{row_off:05d}_{col_off:05d}.tif"
            )
            abs_path = os.path.join(data_root, rel_path)

            tl_x = t.c + col_off * t.a
            tl_y = t.f + row_off * t.e
            top_left = [tl_x, tl_y]

            clipped = clip_polygons_to_patch(gdf, src, col_off, row_off, patch_size)

            seg_list, area_list, bbox_list = [], [], []
            for poly in clipped:
                flat = polygon_to_pixel_coords(poly, src, col_off, row_off)
                bbox, area = bbox_and_area_from_flat(flat)
                if area < min_area:
                    continue
                seg_list.append(flat)
                area_list.append(area)
                bbox_list.append(bbox)

            save_patch_image(src, col_off, row_off, patch_size, abs_path)

            image_records.append({
                "file_name":  rel_path,
                "image_path": rel_path,
                "width":      patch_size,
                "height":     patch_size,
                "res_x":      res_x,
                "res_y":      res_y,
                "top_left":   top_left,
                "_seg_list":  seg_list,
                "_area_list": area_list,
                "_bbox_list": bbox_list,
            })

    return image_records


# ---------------------------------------------------------------------------
# COCO JSON builder
# ---------------------------------------------------------------------------

def build_coco_json(all_image_records, category, image_id_start=1, ann_id_start=1):
    """
    Convert a flat list of image records into a COCO JSON dict whose structure
    matches the P3 NY dataset format expected by p3_coco.py / P3Dataset.
    lidar_path is intentionally omitted — this dataset has no LiDAR.
    """
    coco = {
        "info": {
            "year": 2026, "version": "1.0",
            "description": "AI4SmallFarms Asia — COCO patches for Pix2Poly",
            "contributor": "", "url": "", "date_created": "",
        },
        "categories": [category],
        "images": [],
        "annotations": [],
    }

    ann_id = ann_id_start
    for img_id, rec in enumerate(all_image_records, start=image_id_start):
        coco["images"].append({
            "id":         img_id,
            "file_name":  rec["file_name"],
            "image_path": rec["image_path"],
            "width":      rec["width"],
            "height":     rec["height"],
            "res_x":      round(rec["res_x"], 6),
            "res_y":      round(rec["res_y"], 6),
            "top_left":   [round(rec["top_left"][0], 6), round(rec["top_left"][1], 6)],
        })
        for feature_id, (flat, area, bbox) in enumerate(
            zip(rec["_seg_list"], rec["_area_list"], rec["_bbox_list"])
        ):
            coco["annotations"].append({
                "feature_id":   feature_id,
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
    parser = argparse.ArgumentParser(description="Build COCO dataset from Sentinel-2 tiles")
    parser.add_argument(
        "--data_root", required=True,
        help="Root directory of the Sentinel-2 Asia dataset "
             "(must contain train/images, validate/images, test/images, reference/)"
    )
    parser.add_argument(
        "--split", choices=["train", "validate", "test", "all"], default="all",
        help="Which split to process (default: all)"
    )
    args = parser.parse_args()

    data_root = args.data_root
    ref_dir   = os.path.join(data_root, "reference")
    out_dir   = os.path.join(data_root, "output_coco_64")
    os.makedirs(out_dir, exist_ok=True)

    all_splits = {
        "train":    os.path.join(data_root, "train",    "images"),
        "validate": os.path.join(data_root, "validate", "images"),
        "test":     os.path.join(data_root, "test",     "images"),
    }
    out_json = {
        "train":    os.path.join(out_dir, "train_coco.json"),
        "validate": os.path.join(out_dir, "val_coco.json"),
        "test":     os.path.join(out_dir, "test_coco.json"),
    }

    splits = all_splits if args.split == "all" else {args.split: all_splits[args.split]}

    for split, images_dir in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing split: {split.upper()}  ({images_dir})")
        print(f"{'='*60}")

        tile_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".tif"))
        if TEST_LIMIT is not None:
            tile_files = tile_files[:TEST_LIMIT]
            print(f"  [TEST_LIMIT={TEST_LIMIT}] Processing only: {tile_files}")

        all_records = []
        for tile_file in tile_files:
            tile_id   = tile_file.replace(".tif", "")
            tile_path = os.path.join(images_dir, tile_file)
            ref_path  = find_reference_file(tile_id, ref_dir)

            if ref_path is None:
                print(f"  WARNING: no reference file for {tile_id}, skipping.")
                continue

            print(f"\n  Tile: {tile_id}")
            records = process_tile(tile_path, ref_path, split, tile_id, data_root)
            print(f"    → {len(records)} patches generated")
            all_records.extend(records)

        coco = build_coco_json(all_records, CATEGORY)
        with open(out_json[split], "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        print(f"\n  Saved: {out_json[split]}")
        print(f"  Images: {n_img}  |  Annotations: {n_ann}  |  Avg ann/patch: {n_ann/n_img:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
