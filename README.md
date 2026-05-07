# Pix2Poly on AI4SmallFarms

This repository adapts [Pix2Poly](https://github.com/raphaelsulzer/PixelsPointsPolygons)
— a polygon prediction model originally designed for building footprints — to delineate
smallholder crop fields from Sentinel-2 imagery using the
[AI4SmallFarms](https://doi.org/10.17026/dans-xy6-ngg6) dataset (Vietnam and Cambodia).

It is **not a reimplementation**. The model, trainer, and backbone all come from
[pix2poly_p3_image_only](https://github.com/imanemn127/pix2poly_p3_image_only), a fork
of the original P3 codebase with image-only fixes and training optimisations already
applied. This repo contains only the adaptation layer: a local dataset class, a modified
ViT loader, Hydra configs, and data preparation scripts.

---

## Why this is non-trivial

The original model was designed for 0.25 m/pixel RGB aerial images (uint8, 224×224 px).
Sentinel-2 is 10 m/pixel, 3-band uint16. That gap creates several concrete problems:

| Problem | What breaks | Fix |
|---------|------------|-----|
| Much larger fields at 10 m/px | Patches with hundreds of vertices exceed `max_num_vertices` | Reduced patch size to 64×64 |
| uint16 reflectance values | Default normalisation assumes uint8 | Custom `denormalize_s2` patched at runtime |
| ViT expects 224×224 input | Shape mismatch when loading pretrained weights | Pretrained positional embeddings discarded; relearned from scratch on 64×64 |
| NumPy ≥ 2.0 in base env | `torch.from_numpy()` raises `RuntimeError` | Dedicated `ai4sf` conda env pinned to NumPy 1.26.4 |

The approach is to patch the installed package **at runtime** rather than modifying it on
disk, keeping the fork clean and making the patches explicit and auditable.

---

## Repository structure

```
Pix2Poly_AI4SmallFarms/
├── train.py              # entry point — applies 3 runtime patches, launches trainer
├── p3_coco.py            # local dataset class (safe tensor conversion, no LiDAR)
├── modified_vit.py       # ViT-S/8 adapted for 64×64 input
├── run_train.sh          # convenience launch script
├── config/               # Hydra YAML configs
│   ├── config.yaml
│   ├── dataset/ai4smallfarms.yaml
│   ├── encoder/vit_s2.yaml
│   ├── experiment/p2p_ai4smallfarms.yaml
│   ├── host/default.yaml
│   ├── model/pix2poly_fields.yaml
│   └── run_type/ai4smallfarms.yaml
└── scripts/
    ├── build_coco_dataset.py   # tile → COCO patches
    ├── inspect_coco.py         # sanity-check visualisation
    ├── coco_to_gpkg.py         # export annotations to GeoPackage
    ├── stats.py                # dataset statistics
    └── plot_losses.py          # plot metrics.csv curves
```

---

## Setup

### 1. Get the P3 fork

The fork at
[pix2poly_p3_image_only](https://github.com/imanemn127/pix2poly_p3_image_only)
is required — **not** the original `raphaelsulzer/PixelsPointsPolygons`. The original
has broken LiDAR imports, a crashing `get_tile_names_from_dataloader`, and no training
optimisations. The fork fixes all of that.

```bash
git clone https://github.com/imanemn127/pix2poly_p3_image_only.git
```

### 2. Create the conda environment

NumPy must be pinned before anything else resolves it.

```bash
conda create -n ai4sf python=3.11 -y
conda activate ai4sf

pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu118

pip install "numpy==1.26.4"

pip install \
    rasterio==1.3.10 \
    opencv-python==4.9.0.80 \
    transformers==4.38.2 \
    hydra-core==1.3.2 \
    omegaconf pycocotools shapely geopandas \
    matplotlib pandas tqdm timm

# Install the fork in editable mode
pip install -e /path/to/pix2poly_p3_image_only
```

### 3. Clone this repo

```bash
git clone https://github.com/imanemn127/Pix2Poly_AI4SmallFarms.git
cd Pix2Poly_AI4SmallFarms
```

### 4. Download the DINO backbone

```bash
wget -P /path/to/backbones \
  https://huggingface.co/rsi/PixelsPointsPolygons/resolve/main/backbones/dino_deitsmall8_pretrain.pth
```

---

## Dataset preparation

The AI4SmallFarms dataset is available at:
> https://doi.org/10.17026/dans-xy6-ngg6

It provides Sentinel-2 tiles and reference field polygons as GeoPackages for
Vietnam and Cambodia. Download and organise as:

```
sentinel-2-asia/
├── train/images/      *.tif
├── validate/images/   *.tif
├── test/images/       *.tif
└── reference/         *_areas.gpkg   (one per tile)
```

### Build the COCO patches

```bash
python scripts/build_coco_dataset.py \
    --data_root /path/to/sentinel-2-asia \
    --split all
```

This splits each tile into non-overlapping 64×64 px patches, clips field polygons
to each patch, converts UTM to local pixel coordinates, and writes:

```
sentinel-2-asia/output_coco_64/
├── train_coco.json
├── val_coco.json
└── test_coco.json
```

plus the actual patch images under `<split>/patches_64/`.

**Why 64×64?**  
At larger patch sizes many patches contained fields with more vertices than
`max_num_vertices` after clipping, causing sequences to be truncated before the
polygon closed. 64×64 (640 m × 640 m at 10 m/px) keeps field complexity manageable.

### Check the dataset

```bash
# Quick statistics
python scripts/stats.py \
    --json /path/to/sentinel-2-asia/output_coco_64/train_coco.json

# Visual check — saves a PNG with polygon overlays
python scripts/inspect_coco.py \
    --json /path/to/sentinel-2-asia/output_coco_64/val_coco.json \
    --root /path/to/sentinel-2-asia \
    --out  patch_check.png

# Export to GeoPackage for QGIS
python scripts/coco_to_gpkg.py \
    --json /path/to/sentinel-2-asia/output_coco_64/val_coco.json \
    --out  val_geo.gpkg \
    --epsg 32648
```

---

## Configuration

Fill in the placeholder paths (marked `# SET`) before training:

- `config/dataset/ai4smallfarms.yaml` — `in_path`, `out_path`
- `config/encoder/vit_s2.yaml` — `checkpoint_file`
- `config/host/default.yaml` — `data_root`, `model_root`

Key parameters:

| Parameter | Value | Note |
|-----------|-------|------|
| `patch_size` | 64 px | 640 m × 640 m at 10 m/px |
| `num_bins` | 64 | one bin per pixel column/row |
| `max_num_vertices` | 1024 | total vertices per patch |
| `backbone` | ViT-S/8 DINO | positional embeddings reinitialised |
| `augmentations` | D4 + Normalize | 8-fold dihedral group |
| `evaluation mode` | `iou` | same as original P3 |
| `batch_size` | 4 | on A100 40 GB |
| `learning_rate` | 3e-4 | AdamW |

---

## Training

```bash
tmux new -s ai4sf
conda activate ai4sf
cd /path/to/Pix2Poly_AI4SmallFarms

CUDA_VISIBLE_DEVICES=0 python train.py experiment=p2p_ai4smallfarms
```

Or:

```bash
bash run_train.sh
```

Checkpoints and `metrics.csv` go to a timestamped directory under `out_path`.

### What `train.py` actually does

Three runtime patches are applied before the trainer is instantiated:

1. **Dataset** — `P3Dataset` and its Train/Val/Test subclasses in the installed
   package are replaced with the local versions from `p3_coco.py`, which use
   `_safe_to_tensor()` instead of `torch.from_numpy()` and drop all LiDAR code.

2. **ViT** — `ViT` in `pixelspointspolygons.models.vision_transformer` and
   `pixelspointspolygons.models.pix2poly.model_pix2poly` is replaced with the
   local `modified_vit.ViT`, which builds the model at `img_size=64` and discards
   the pretrained positional embeddings so they are learned from scratch.

3. **Visualisation** — `denormalize_image_for_visualization` in both
   `shared_utils` and `trainer_pix2poly` is replaced with `denormalize_s2`, which
   correctly inverts the Sentinel-2 normalisation for display.

Nothing in the installed package is touched on disk.

### Training optimisations

Because this project uses the
[pix2poly_p3_image_only](https://github.com/imanemn127/pix2poly_p3_image_only)
fork as its backend, all the optimisations from there are inherited automatically:

| Optimisation | Effect |
|---|---|
| Mixed precision (AMP) | ~2× memory reduction, faster forward/backward |
| `torch.compile` | Faster iterations after the first step |
| Gradient clipping (norm 1.0) | Prevents divergence with long vertex sequences |
| Separate visualisation loader | Clean, augmentation-free validation previews |
| BCELoss NaN fix | Clamps logits to avoid `log(0)` |
| Memory cleanup per epoch | `empty_cache()` + `gc.collect()` |

---

## Monitoring

```bash
watch -n 1 nvidia-smi
tail -f train_ai4smallfarms.log
```

Each epoch appends a row to `metrics.csv`: `epoch`, `train_loss`, `val_loss`,
`val_iou` (computed every 5 epochs). `val_iou` is used for best-checkpoint selection.

```bash
python scripts/plot_losses.py --output_root /path/to/AI4SmallFarms_output
# or for a specific run:
python scripts/plot_losses.py --run_dir /path/to/run/2026-05-07_12-41-40
```

Saves `loss_curves.png` alongside `metrics.csv`.

---

## Troubleshooting

**`RuntimeError: can't convert np.ndarray to tensor`**  
NumPy ≥ 2.0 is installed. Fix: `pip install "numpy==1.26.4"`.

**`FileNotFoundError` for checkpoint or dataset**  
The placeholder paths in the config files haven't been set yet. Check
`config/encoder/vit_s2.yaml` and `config/dataset/ai4smallfarms.yaml`.

**`ImportError: open3d` or any LiDAR-related error**  
You installed the original P3 repo instead of the fork. Reinstall from
`https://github.com/imanemn127/pix2poly_p3_image_only`.

**ViT shape mismatch at startup**  
The ViT patch in `train.py` wasn't applied. Always launch with
`python train.py`, not by importing the trainer directly.

**`CUDA out of memory`**  
Lower `batch_size` in `config/run_type/ai4smallfarms.yaml`.

**tmux session dies silently**  
Use the full Python path: `/path/to/envs/ai4sf/bin/python train.py`.

---

## Dependencies

Python 3.11 · PyTorch 2.2.2 · NumPy 1.26.4 · transformers 4.38.2 ·  
rasterio 1.3.10 · hydra-core 1.3.2 · geopandas ≥ 0.14 · CUDA 11.8

---

## Citation

```bibtex
@misc{sulzer2025p3,
  title  = {The P$^3$ Dataset: Pixels, Points and Polygons},
  author = {Raphael Sulzer et al.},
  year   = {2025}
}

@dataset{ai4smallfarms2024,
  title  = {AI4SmallFarms},
  year   = {2024},
  doi    = {10.17026/dans-xy6-ngg6},
  url    = {https://doi.org/10.17026/dans-xy6-ngg6}
}
```

Original model: [raphaelsulzer/PixelsPointsPolygons](https://github.com/raphaelsulzer/PixelsPointsPolygons)  
P3 fork used here: [imanemn127/pix2poly_p3_image_only](https://github.com/imanemn127/pix2poly_p3_image_only)
