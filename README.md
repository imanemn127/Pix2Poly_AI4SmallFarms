# Pix2Poly on AI4SmallFarms

This repository adapts [Pix2Poly](https://github.com/raphaelsulzer/PixelsPointsPolygons)
— a polygon prediction model originally designed for building footprints — to delineate
smallholder crop fields from Sentinel-2 imagery using the
[AI4SmallFarms](https://doi.org/10.17026/dans-xy6-ngg6) dataset (Vietnam and Cambodia).

It is **not a reimplementation**. The model, trainer, and backbone all come from
[Pix2poly_P3_image_only](https://github.com/imanemn127/Pix2poly_P3_image_only), a fork
of the original P3 codebase with image-only fixes and training optimisations already
applied. This repo contains only the adaptation layer: a local dataset class, a modified
ViT loader, Hydra configs, and data preparation scripts.

---

## Why this is non-trivial

The original model was designed for 0.25 m/pixel RGB aerial images (uint8, 224×224 px).
Sentinel-2 is 10 m/pixel, 3-band uint16. That gap creates several concrete problems:

| Problem | What breaks | Fix |
|---------|------------|-----|
| Much larger fields at 10 m/px | Patches with hundreds of vertices exceed `max_num_vertices` | Reduced patch size to 32×32; set `max_num_vertices` to 384 (only 1.1 % of patches exceed this) |
| uint16 reflectance values | Default normalisation assumes uint8 | Custom `denormalize_s2` patched at runtime |
| ViT expects 224×224 input | Shape mismatch when loading pretrained weights | Pretrained positional embeddings discarded; relearned from scratch on 32×32 |
| NumPy ≥ 2.0 in base env | `torch.from_numpy()` raises `RuntimeError` | Dedicated `ai4sf` conda env pinned to NumPy 1.26.4 |
| Evaluator hard-codes category 100 (building) | IoU evaluated against wrong category | Evaluator and predictor patched to use category id=1 (field) |

The approach is to patch the installed package **at runtime** rather than modifying it on
disk, keeping the fork clean and making the patches explicit and auditable.

---

## Repository structure

```
Pix2Poly_AI4SmallFarms/
├── train.py              # entry point — applies runtime patches, launches trainer
├── p3_coco.py            # local dataset class (safe tensor conversion, no LiDAR)
├── modified_vit.py       # ViT-S/2 adapted for 32×32 input
├── run_train.sh          # convenience launch script
├── config/               # Hydra YAML configs
│   ├── config.yaml
│   ├── dataset/ai4smallfarms.yaml
│   ├── encoder/vit_s2.yaml
│   ├── evaluation/val.yaml
│   ├── experiment/p2p_ai4smallfarms.yaml
│   ├── host/default.yaml
│   ├── model/pix2poly_fields.yaml
│   ├── run_type/ai4smallfarms.yaml
│   └── training/default.yaml
└── scripts/
    ├── build_coco_dataset.py   # tile → COCO patches
    ├── inspect_coco.py         # sanity-check visualisation
    ├── coco_to_gpkg.py         # export annotations to GeoPackage
    ├── stats.py                # dataset statistics
    └── plot_losses_ai4sf.py    # plot metrics.csv curves
```

---

## Setup

### 1. Get the P3 fork

The fork at
[Pix2poly_P3_image_only](https://github.com/imanemn127/Pix2poly_P3_image_only)
is required — **not** the original `raphaelsulzer/PixelsPointsPolygons`. The original
has broken LiDAR imports, a crashing `get_tile_names_from_dataloader`, and no training
optimisations. The fork fixes all of that.

```bash
git clone https://github.com/imanemn127/Pix2poly_P3_image_only.git
```

### 2. Create the conda environment

NumPy must be pinned before anything else resolves it. NumPy ≥ 2.0 breaks
`torch.from_numpy()` inside the P3 package; pinning to 1.26.4 is non-negotiable.

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
    scikit-learn matplotlib pandas tqdm timm

# Install the fork in editable mode
pip install -e /path/to/Pix2poly_P3_image_only
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

It provides Sentinel-2 tiles (multi-band GeoTIFF) and reference field polygons as
GeoPackages for Vietnam and Cambodia. The raw download uses `validate/` as the split
name (not `val/`). Download and organise as:

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

This splits each tile into non-overlapping 32×32 px patches, clips field polygons
to each patch, converts UTM to local pixel coordinates, and writes:

```
sentinel-2-asia/output_coco_32/
├── train_coco.json
├── val_coco.json
└── test_coco.json
```

plus the actual patch images under `<split>/patches_32/`.

The patch images are saved as 3-band GeoTIFFs, reading bands `[4, 3, 2]` (Red, Green,
Blue) from the original 10-band Sentinel-2 tiles.

### Too many polygon vertices per patch

**Problem.** A 224×224 px patch at 10 m resolution covers 2.24 km × 2.24 km and can
contain several hundred fields. The original P3 model was designed for 1–10 buildings
per image (`max_num_vertices = 192`). With hundreds of fields, a single patch can have
thousands of vertices, far exceeding that limit.

**Solution: reduce patch size and raise `max_num_vertices`.** I progressively reduced
the patch size and simultaneously increased the model's vertex capacity.

All values below were measured on the actual dataset using `scripts/stats.py`.

| Patch size | Ground area | Avg fields/patch | Avg total vertices |
|------------|-------------|------------------|--------------------|
| 224×224 | 2.24 km × 2.24 km | ~650 | ~6200 |
| 112×112 | 1.12 km × 1.12 km | ~166 | ~1559 |
| 64×64 | 640 m × 640 m | ~56 | ~512 |
| 56×56 | 560 m × 560 m | ~42 | ~387 |
| **32×32** | 320 m × 320 m | ~14 | ~128 |

The 32×32 size was chosen to keep most patches well under the vertex budget.
`max_num_vertices` was tuned using `scripts/stats.py`:

| max_num_vertices | Images > max (% of total) | Vertices lost (% of total) |
|------------------|---------------------------|----------------------------|
| 192 | 2446 (91.2 %) | 63.1 % |
| 256 | 2199 (82.0 %) | 52.1 % |
| **384** | **29 (1.1 %)** | **~0.5 %** |
| 512 | 1084 (40.4 %) | 21.8 % |
| 768 | 380 (14.2 %) | 8.4 % |
| 1024 | 115 (4.3 %) | 4.6 % |

With 384, over 98 % of patches are fully preserved. The resulting sequence length
(`max_len = 384 × 2 + 2 = 770`) is also tractable for autoregressive generation.

### Filtering truncated patches

Rather than letting the tokenizer silently truncate long sequences, patches whose total
vertex count exceeds `max_num_vertices` are **removed at load time** by `p3_coco.py`.
The `__init__` method filters `tile_ids` before any epoch starts, discarding ~1 % of
patches (~112 images in the 32 px dataset). This keeps all remaining training examples
exact — no polygon is ever partially predicted.

### Check the dataset

```bash
# Quick statistics
python scripts/stats.py \
    --json /path/to/sentinel-2-asia/output_coco_32/train_coco.json

# Visual check — saves a PNG with polygon overlays
python scripts/inspect_coco.py \
    --json /path/to/sentinel-2-asia/output_coco_32/val_coco.json \
    --root /path/to/sentinel-2-asia \
    --out  patch_check.png

# Export to GeoPackage for QGIS
python scripts/coco_to_gpkg.py \
    --json /path/to/sentinel-2-asia/output_coco_32/val_coco.json \
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
| `in_size` | 32 px | 320 m × 320 m at 10 m/px |
| `patch_size` | 2 | Finer ViT tokenisation on 32×32 input |
| `patch_feature_size` | 16 | `in_size / patch_size` |
| `num_patches` | 256 | `(32 / 2)²` |
| `num_bins` | 32 | one bin per pixel column/row |
| `max_num_vertices` | 384 | total vertices across all polygons in one patch; ~1.1 % of patches exceed this and are dropped at load time |
| `max_len` | 770 | 384 × 2 coords + BOS + EOS, computed at runtime |
| `generation_steps` | 770 | matches `max_len` |
| `backbone` | ViT-S/2 DINO | positional embeddings reinitialised from scratch |
| `augmentations` | D4 + Normalize | 8-fold dihedral group (rotations 0°/90°/180°/270° + reflections) |
| `batch_size` | 4 | in `run_type/ai4smallfarms.yaml` |
| `learning_rate` | decoder 1e-3, patch/pos embed 5e-4, other 3e-4 | per-group AdamW — see training strategy below |
| `num_epochs` | 100 | |
| `val_every` | 5 | IoU evaluated every 5 epochs |

The Sentinel-2 normalisation is currently set to identity (mean=0, std=1,
max_pixel=10000) in `config/encoder/vit_s2.yaml`. Update with per-dataset per-channel
statistics for better convergence once you have them.

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

`run_train.sh` sets `WANDB_MODE=offline` and tees output to `train_ai4smallfarms.log`.
Checkpoints and `metrics.csv` go to a timestamped directory under `out_path`:

```
<out_path>/pix2poly/32/v1_image_vit_bs4_ai4smallfarms/<YYYY-MM-DD_HH-MM-SS>/
├── checkpoints/
│   ├── best.pth
│   └── latest.pth
└── metrics.csv
```

### What `train.py` actually does

The file applies several runtime patches before the trainer is instantiated. Because
Python's import system caches modules by reference, replacing a name in a module
namespace makes all subsequent lookups in that module use the new object — without
touching anything on disk.

| Patch | Affected module(s) | What it changes |
|-------|-------------------|-----------------|
| 1. Dataset | `pixelspointspolygons.datasets.p3_coco` | Replaces `P3Dataset` / `TrainDataset` / `ValDataset` / `TestDataset` with the local `p3_coco.py` versions, which use `_safe_to_tensor()`, drop all LiDAR code, and filter out patches exceeding `max_num_vertices` at load time. |
| 2. ViT backbone | `pixelspointspolygons.models.vision_transformer` and `…model_pix2poly` | Replaces the original `ViT` class with `modified_vit.ViT`, which builds the model at `img_size=32, patch_size=2`, and discards the pretrained `pos_embed` and `patch_embed` weights so they are learned from scratch (attention weights are kept). |
| 3a. Evaluator category | `pixelspointspolygons.eval.evaluator` | Overrides `compute_coco_metrics` to use `catIds=[1]` (field) instead of the hard-coded `[100]` (building). |
| 3b. Prediction category | `pixelspointspolygons.misc.coco_conversions` and `…predictor_pix2poly` | Wraps `generate_coco_ann` to write `category_id=1` into every predicted annotation. |
| 4. Visualisation | `pixelspointspolygons.misc.shared_utils` and `…trainer_pix2poly` | Replaces `denormalize_image_for_visualization` with `denormalize_s2`, which inverts the Normalize transform then applies a per-channel p2–p98 percentile stretch. Linear rescaling by `max_pixel_value=10000` produces near-black images because S2 reflectance clusters in the bottom 20–30 % of the range; the percentile stretch makes training visualisations readable. |
| 5. Decoder regularisation | model object, applied after `setup_model()` | `patch_decoder(model)` increases dropout from 0.1 → 0.2 in every `TransformerDecoderLayer` (self-attn, cross-attn, FFN). |
| 6. Training strategy | optimizer + scheduler, applied after `setup_optimizer()` | `patch_optimizer_strategy_a` freezes all ViT attention blocks (`model.encoder.vit.blocks`) and replaces the single-LR AdamW with per-group rates (see training strategy below). Rebuilds the warmup–linear-decay scheduler on the new optimiser. |

Nothing in the installed package is touched on disk.

### Adapting the ViT backbone for 32×32 input

The DINO checkpoint was trained at 224×224 with patch size 8, giving a 28×28 grid (785
position embeddings including the CLS token, shape `(1, 785, 384)`). The training
patches are 32×32 with patch size 2, which yields a 16×16 grid (257 position
embeddings). The shapes are incompatible, and the patch projection kernel changes from
8×8 to 2×2.

**Approach: discard incompatible weights, keep attention.** `modified_vit.py` builds
the ViT with `img_size=32, patch_size=2, pretrained=False`, loads the DINO checkpoint,
removes both `pos_embed` and `patch_embed` keys, and calls `load_state_dict(...,
strict=False)`.

| Component | Initialisation |
|-----------|---------------|
| `patch_embed` (2×2 conv) | Random — kernel shape changed from 8×8, incompatible |
| `pos_embed` (257 positions) | Random — grid changed from 28×28 to 16×16 |
| Attention weights, MLP, layer norms | Loaded from DINO — kept frozen during training |

### Training strategy

Two strategies are defined in `train.py`. Switch between them by changing one line in
`patched_train`: `patch_optimizer_strategy_a(self)` or `patch_optimizer_strategy_b(self)`.

**Strategy A — freeze attention, train patch embed + decoder**

`patch_optimizer_strategy_a` freezes all ViT attention blocks
(`model.encoder.vit.blocks`) and replaces the single-LR AdamW with per-group rates:

| Parameter group | Learning rate |
|-----------------|--------------|
| `patch_embed` | 5e-4 |
| `pos_embed` | 5e-4 |
| Decoder | 1e-3 |
| Everything else (not frozen) | 3e-4 |

Rationale: DINO features are strong general-purpose representations; freezing attention
avoids catastrophic forgetting. The downside is that with S2 32×32 input (very different
from the ImageNet 224×224 distribution DINO was trained on), the encoder features may
remain poorly adapted and the decoder stagnates near random loss.

**Strategy B — train everything with separate LRs**

`patch_optimizer_strategy_b` unfreezes all blocks and adds a low LR group for attention:

| Parameter group | Learning rate |
|-----------------|--------------|
| `patch_embed` | 5e-4 |
| `pos_embed` | 5e-4 |
| Attention blocks | 1e-5 |
| Decoder | 1e-3 |
| Everything else | 3e-4 |

Recommended when Strategy A stagnates (train loss stays near `log(vocab_size) ≈ 3.56`)
because the frozen encoder cannot adapt to Sentinel-2 reflectance values.

The warmup–linear-decay scheduler (5 % warmup) is rebuilt on the new optimiser after
the swap in both strategies.

### Training optimisations

Because this project uses the
[Pix2poly_P3_image_only](https://github.com/imanemn127/Pix2poly_P3_image_only)
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
# Auto-detect the latest run:
python scripts/plot_losses_ai4sf.py

# Point at a specific run folder:
python scripts/plot_losses_ai4sf.py /path/to/run/2026-05-07_12-41-40
```

Saves `loss_curves_ai4sf.png` alongside `metrics.csv`.

---

## Troubleshooting

**`RuntimeError: can't convert np.ndarray to tensor`**  
NumPy ≥ 2.0 is installed. Fix: `pip install "numpy==1.26.4"`. This must be pinned
*before* installing any package that could pull in a newer NumPy.

**`FileNotFoundError` for checkpoint or dataset**  
The placeholder paths in the config files haven't been set yet. Check
`config/encoder/vit_s2.yaml` and `config/dataset/ai4smallfarms.yaml`.

**`ImportError: open3d` or any LiDAR-related error**  
You installed the original P3 repo instead of the fork. Reinstall from
`https://github.com/imanemn127/Pix2poly_P3_image_only`.

**ViT shape mismatch at startup**  
The ViT patch in `train.py` wasn't applied. Always launch with
`python train.py`, not by importing the trainer directly. The patching relies on
Python's module reference system — it only takes effect if `train.py` is the entry
point.

**`CUDA out of memory`**  
Lower `batch_size` in `config/run_type/ai4smallfarms.yaml`. The default is 4; reducing
to 2 or 1 may be necessary on GPUs with less than 24 GB.

**`ModuleNotFoundError: sklearn`**  
`p3_coco.py` imports `sklearn.preprocessing.MinMaxScaler` (used in the stubbed-out
LiDAR path, which never executes during training). Fix: `pip install scikit-learn`.

**IoU is always 0 or evaluation crashes**  
The evaluator category patch wasn't applied. Make sure you are launching with
`python train.py` (not importing the trainer directly) and that patch 3a/3b in
`train.py` is present.

**`val_iou` is always NaN / "No polygons predicted"**  
The model has not converged yet — train loss must fall well below `log(vocab_size) ≈
3.56` before EOS tokens appear in predictions. With Strategy A, convergence can be very
slow. Switch to Strategy B or use the full dataset.

**Training visualisations are very dark / near-black**  
`denormalize_s2` is not being applied. Confirm that the `shared_utils` and
`trainer_pix2poly` patches (patch 4 in the table above) are present in `train.py` and
that `train.py` is the entry point. The function uses per-channel p2–p98 percentile
stretch to handle the narrow S2 reflectance distribution.

**tmux session dies silently**  
Use the full Python path: `/path/to/envs/ai4sf/bin/python train.py`.

**`build_coco_dataset.py` finds no tiles**  
The script expects `train/images/`, `validate/images/`, and `test/images/` under
`--data_root`. Note the raw AI4SmallFarms download uses `validate/`, not `val/`.

---

## Dependencies

Python 3.11 · PyTorch 2.2.2 · NumPy **1.26.4** · transformers 4.38.2 ·  
rasterio 1.3.10 · hydra-core 1.3.2 · geopandas ≥ 0.14 · scikit-learn · CUDA 11.8

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
P3 fork used here: [imanemn127/Pix2poly_P3_image_only](https://github.com/imanemn127/Pix2poly_P3_image_only)
