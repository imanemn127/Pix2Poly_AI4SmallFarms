#!/usr/bin/env python3
"""
plot_losses.py

Plot training curves (loss and IoU) from a metrics.csv file produced by
the Pix2Poly trainer.

Usage:
  # Auto-detect the latest run under the output root:
  python scripts/plot_losses.py --output_root /path/to/AI4SmallFarms_output

  # Point directly at a specific run folder:
  python scripts/plot_losses.py --run_dir /path/to/AI4SmallFarms_output/pix2poly/64/v1_.../2026-05-07_12-41-40
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def find_latest_run(output_root: str) -> str | None:
    """Return the most recent run directory containing a metrics.csv file."""
    pattern = os.path.join(output_root, "**", "metrics.csv")
    runs = sorted(glob.glob(pattern, recursive=True))
    if not runs:
        return None
    return os.path.dirname(runs[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss and IoU curves from a Pix2Poly metrics.csv file"
    )
    parser.add_argument(
        "--run_dir", default=None,
        help="Run folder containing metrics.csv. "
             "If omitted, the latest run under --output_root is used."
    )
    parser.add_argument(
        "--output_root", default=None,
        help="Root output directory to search for runs "
             "(used when --run_dir is not provided)."
    )
    args = parser.parse_args()

    # Determine run directory
    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.output_root is not None:
        run_dir = find_latest_run(args.output_root)
        if run_dir is None:
            sys.exit(f"No runs found under {args.output_root}")
        print(f"Auto-detected latest run: {run_dir}")
    else:
        sys.exit("Provide either --run_dir or --output_root.")

    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, na_values=["", " "])
    for col in ["epoch", "train_loss", "val_loss", "val_iou"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    iou_df = df.dropna(subset=["val_iou"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["epoch"], df["train_loss"], color="steelblue", linewidth=1.2, label="Train loss")
    ax1.plot(df["epoch"], df["val_loss"],   color="tomato",    linewidth=1.2, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train / Val Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(iou_df["epoch"], iou_df["val_iou"],
             color="seagreen", linewidth=1.2, marker="o", markersize=4, label="Val IoU")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("Validation IoU")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "loss_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    print(f"Epochs: {df['epoch'].min():.0f}–{df['epoch'].max():.0f}  |  IoU points: {len(iou_df)}")


if __name__ == "__main__":
    main()
