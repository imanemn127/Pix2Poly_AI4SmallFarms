#!/usr/bin/env python3
"""
Plot training curves (loss and IoU) from a metrics.csv file.

Usage:
    python plot_losses_ai4sf.py                          # auto‑detect latest run
    python plot_losses_ai4sf.py /path/to/run/folder      # specific run
"""

import argparse
import glob
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def find_latest_run():
    """Return the path of the most recent run containing a metrics.csv file."""
    base = "/mnt/DATA/IMANE/AI4SmallFarms_output/pix2poly/32/v1_image_vit_bs4_ai4smallfarms"
    pattern = os.path.join(base, "*/metrics.csv")
    runs = sorted(glob.glob(pattern))
    if not runs:
        return None
    return os.path.dirname(runs[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss and IoU curves from a Pix2Poly metrics.csv file."
    )
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="Run folder containing metrics.csv.  If omitted, the latest run is used automatically.",
    )
    args = parser.parse_args()

    # -------- determine run directory --------
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            sys.exit(
                "No runs found under "
                "/mnt/DATA/IMANE/AI4SmallFarms_output/pix2poly/32/"
                "v1_image_vit_bs4_ai4smallfarms"
            )
        print(f"Auto‑detected latest run: {run_dir}")

    # -------- read CSV --------
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, na_values=["", " "])
    for col in ["epoch", "train_loss", "val_loss", "val_iou"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # train_iou column is optional — present only in runs that used the new
    # patched_train with TRAIN_IOU_SUBSET support.
    has_train_iou = "train_iou" in df.columns
    if has_train_iou:
        df["train_iou"] = pd.to_numeric(df["train_iou"], errors="coerce")

    iou_df       = df.dropna(subset=["val_iou"])
    train_iou_df = df.dropna(subset=["train_iou"]) if has_train_iou else pd.DataFrame()

    # -------- plot --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(df["epoch"], df["train_loss"], color="steelblue", linewidth=1.2, label="Train loss")
    ax1.plot(df["epoch"], df["val_loss"],   color="tomato",    linewidth=1.2, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train / Val Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IoU curves (val + optional train)
    ax2.plot(iou_df["epoch"], iou_df["val_iou"],
             color="seagreen", linewidth=1.2, marker="o", markersize=4, label="Val IoU")
    if has_train_iou and len(train_iou_df):
        ax2.plot(train_iou_df["epoch"], train_iou_df["train_iou"],
                 color="steelblue", linewidth=1.2, marker="s", markersize=4,
                 linestyle="--", label="Train IoU (subset)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("IoU (every 10 epochs)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "loss_curves_ai4sf.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    n_train_iou = len(train_iou_df) if has_train_iou else 0
    print(
        f"Epochs plotted: {df['epoch'].min():.0f}–{df['epoch'].max():.0f}  "
        f"|  Val IoU points: {len(iou_df)}  "
        f"|  Train IoU points: {n_train_iou}"
    )


if __name__ == "__main__":
    main()