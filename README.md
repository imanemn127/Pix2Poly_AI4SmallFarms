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
| Much larger fields at 10 m/px | Patches with hundreds of vertices exceed `max_num_vertices` | Reduced patch size to 32×32; set `max_num_vertices` to 192 |
| uint16 reflectance values | Default normalisation assumes uint8 | Custom `denormalize_s2` patched at runtime; real per-channel stats computed on training set |
| ViT expects 224×224 input | Shape mismatch when loading pretrained weights | Both `patch_embed` and `pos_embed` discarded; relearned from scratch on 32×32 |
| NumPy ≥ 2.0 in base env | `torch.from_numpy()` raises `RuntimeError` | Dedicated `ai4sf` conda env pinned to NumPy 1.26.4 |
| Evaluator hard-codes category 100 (building) | IoU evaluated against wrong category | Evaluator and predictor patched to use category id=1 (field) |
| `calc_IoU` returns 1.0 for void images | Fake IoU floor (~0.051) masks real model performance | Patched to return `nan`; void images excluded from the mean |
| Decoder mode collapse | Coordinate tokens stuck at constant bin (0 or 31) for all inputs | Decoder LR reduced from 1e-3 to 3e-4; `vertex_loss_weight` raised to force coordinate learning; `shuffle_polygons` enabled so polygon order varies each batch |
| fp16 overflow in Sinkhorn | `log_sinkhorn_iterations` uses `logsumexp` which overflows in fp16 → NaN in `perm_mat` → BCELoss CUDA kernel assertion | Entire model forward (encoder + decoder + Sinkhorn) forced to float32 via `autocast('cuda', enabled=False)`; AMP and `torch.compile` disabled at runtime |

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
    ├── build_coco_dataset.py     # tile → COCO patches (clips annotations to patch boundary)
    ├── inspect_coco.py           # sanity-check visualisation
    ├── coco_to_gpkg.py           # export annotations to GeoPackage
    ├── stats.py                  # dataset statistics
    ├── compute_normalization.py  # compute per-channel mean/std on training patches
    ├── diag_eos.py               # diagnostic: EOS position analysis + perm_matrix/coord inspection
    └── plot_losses_ai4sf.py      # plot metrics.csv curves
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

**GT polygon border clipping.** Reference field polygons sometimes extend slightly
outside the 32×32 patch boundary (fields that straddle tile edges). `build_coco_dataset.py`
clips every polygon to the patch extent before writing pixel coordinates. Without this,
visualisations show annotation vertices outside the image, and the model receives
out-of-range coordinate targets.

### Too many polygon vertices per patch

**Problem.** A 224×224 px patch at 10 m resolution covers 2.24 km × 2.24 km and can
contain several hundred fields. The original P3 model was designed for 1–10 buildings
per image (`max_num_vertices = 192`). With hundreds of fields, a single patch can have
thousands of vertices, far exceeding that limit.

**Solution: reduce patch size.** I progressively reduced the patch size and measured
vertex counts at each size using `scripts/stats.py`.

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
| **192** | **1792 (15.5 %)** | **8.8 %** |
| 256 | 524 (4.5 %) | 4.2 % |
| 384 | 126 (1.1 %) | 2.0 % |
| 512 | 75 (0.6 %) | 1.1 % |
| 768 | 28 (0.2 %) | 0.3 % |
| 1024 | 6 (0.1 %) | 0.1 % |

The current value is **192** (`max_len = 192 × 2 + 2 = 386`, `generation_steps = 385`),
dropping ~15.5 % of patches.

**Training history with `max_num_vertices`:**
- First run used **384** (1.1 % patches dropped). After 48 epochs IoU never exceeded 0.051.
  Hypothesis: too many vertices per sequence was harming decoder learning.
- Reduced to **192**. After 30 more epochs IoU was still exactly 0.051 every eval epoch.
  This looked like stagnation but was in fact the `calc_IoU` void-image bug (patch 3c):
  127 of 2477 val images have no annotation; on those, union=0 → `calc_IoU` returned 1.0,
  creating a persistent floor of 127/2477 ≈ 0.051 regardless of what the model predicted.

### Filtering truncated patches

Rather than letting the tokenizer silently truncate long sequences, patches whose total
vertex count exceeds `max_num_vertices` are **removed at load time** by `p3_coco.py`.
The `__init__` method filters `tile_ids` before any epoch starts. This keeps all
remaining training examples exact — no polygon is ever partially predicted.

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
| `patch_size` | 4 | 4×4 px ViT tokens (40 m × 40 m); was 2, reduced token count from 256 to 64 |
| `patch_feature_size` | 8 | `in_size / patch_size` |
| `num_patches` | 64 | `(32 / 4)²` |
| `num_bins` | 32 | one bin per pixel column/row |
| `shuffle_tokens` | false | vertex order fixed; shuffling disabled for sequence stability |
| `shuffle_polygons` | true | polygon order randomised each batch; prevents decoder from memorising a fixed sequence order |
| `max_num_vertices` | 192 | total vertices across all polygons in one patch; ~15.5 % of patches dropped at load time |
| `max_len` | 386 | 192 × 2 coords + BOS + EOS, computed at runtime |
| `generation_steps` | 385 | matches `max_len` |
| `backbone` | ViT-S/2 DINO | `patch_embed` and `pos_embed` reinitialised from scratch; attention weights loaded from DINO |
| `image_mean` | [0.2416, 0.1120, 0.1008] | per-channel mean after dividing by 10000, computed on 11579 train patches |
| `image_std` | [0.0694, 0.0510, 0.0324] | per-channel std, same computation |
| `image_max_pixel_value` | 10000.0 | S2 reflectance scale factor |
| `augmentations` | D4 + Normalize | 8-fold dihedral group + channel normalisation |
| `batch_size` | 4 | in `run_type/ai4smallfarms.yaml` |
| `vertex_loss_weight` | 10.0 | coordinate cross-entropy weight; raised from 1.0 to fix decoder mode collapse |
| `perm_loss_weight` | 3.0 | permutation matrix BCE weight; reduced from P3 default of 10.0 |
| `learning_rate` | decoder 3e-4, patch/pos embed 5e-4, attn 3e-5, other 3e-4 | per-group AdamW |
| `num_epochs` | 100 | |
| `val_every` | 10 | IoU evaluated every 10 epochs |

### Sentinel-2 normalisation

S2 reflectance values are stored as uint16 scaled by 10000. The ViT backbone expects
inputs centred near zero. Earlier runs used identity normalisation (mean=0, std=1),
which fed values in [0, 0.4] to a backbone expecting [-2, 2] inputs.

The normalisation stats were computed on all 11579 training patches using the Welford
online algorithm (`scripts/compute_normalization.py`):

```
image_mean: [0.2416, 0.1120, 0.1008]   # bands R, G, B after /10000
image_std:  [0.0694, 0.0510, 0.0324]
```

To recompute for a different dataset:

```bash
python scripts/compute_normalization.py
# or on a random subset for speed:
python scripts/compute_normalization.py --max_images 2000
```

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
│   ├── best_val_loss.pth
│   ├── best_val_iou.pth
│   ├── latest.pth
│   └── epoch_N.pth
└── metrics.csv
```

### What `train.py` actually does

The file applies several runtime patches before the trainer is instantiated. Because
Python's import system caches modules by reference, replacing a name in a module
namespace makes all subsequent lookups in that module use the new object — without
touching anything on disk.

| Patch | Affected module(s) | What it changes |
|-------|-------------------|-----------------|
| 1a. Dataset | `pixelspointspolygons.datasets.p3_coco` | Replaces dataset classes with local `p3_coco.py` versions: safe tensor conversion, no LiDAR, filters patches exceeding `max_num_vertices` at load time. |
| 1b. cv2 import fix | `pixelspointspolygons.datasets.build_datasets` | Injects `cv2` into the module namespace; the package imports it at the top level but does not list it as a dependency, causing `NameError` at dataloader build time. |
| 2. ViT backbone | `pixelspointspolygons.models.vision_transformer` and `…model_pix2poly` | Replaces `ViT` with `modified_vit.ViT`: builds at `img_size=32, patch_size=4`, discards pretrained `pos_embed` and `patch_embed`, keeps attention weights. |
| 3a. Evaluator category | `pixelspointspolygons.eval.evaluator` | Overrides `compute_coco_metrics` to use `catIds=[1]` (field) instead of hard-coded `[100]` (building). |
| 3b. Prediction category | `pixelspointspolygons.misc.coco_conversions` and `…predictor_pix2poly` | Wraps `generate_coco_ann` to write `category_id=1` into every predicted annotation. |
| 3c. IoU metric fix | `pixelspointspolygons.eval.cIoU` and `…eval.evaluator` | `calc_IoU` returned 1.0 when both GT and prediction are empty (union=0), creating a fake floor of ~0.051 (5.1 % of val images have no annotation). Now returns `nan`. `compute_IoU_cIoU` rewritten to skip void images using `nanmean`; the local binding inside `evaluator.py` is also repatched (it imports `compute_IoU_cIoU` by name, so the module-level replacement alone is not enough). |
| 4. Visualisation | `pixelspointspolygons.misc.shared_utils` and `…trainer_pix2poly` | Replaces `denormalize_image_for_visualization` with `denormalize_s2`: inverts Normalize, then applies per-channel p2–p98 percentile stretch. Needed because S2 reflectance clusters in the bottom 20–30 % of the range; linear rescaling produces near-black images. |
| 5. Decoder regularisation | model object | `patch_decoder(model)` increases dropout from 0.1 → 0.2 in every `TransformerDecoderLayer`. |
| 6. Training strategy | optimizer + scheduler | Two-phase: phase 1 freezes ViT attention for `FREEZE_EPOCHS` epochs; phase 2 unfreezes with low LR and restarts warmup. |
| 7. Visualization gating | `trainer.visualization` | Skips visualization on non-val epochs; draws GT from raw COCO pixel coordinates instead of decoded tokens. |
| 8. Train IoU logging | CSV writer (swapped at module level) + `save_best_and_latest_checkpoint` hook | Every `val_every` epochs the predictor runs on a fixed 512-image train subset selected randomly (seed=0, shuffle=False) spread across all tiles; the result is appended as a `train_iou` column to `metrics.csv` without modifying `train_val_loop`. |
| 9. Loss breakdown logging | `train_one_epoch` hook + CSV writer | Captures `coords_loss` and `perm_loss` each epoch and writes them as extra CSV columns, enabling diagnosis of coordinate vs permutation learning. |
| 10. Float32 forward | `type(model).forward` monkey-patch + `torch.compile` override | Replaces `EncoderDecoder.forward` with `_safe_perm_forward`: runs the entire forward (encoder, decoder, Sinkhorn) inside `autocast('cuda', enabled=False)`. Prevents fp16 overflow in `log_sinkhorn_iterations` which produced all-NaN `perm_mat` and crashed BCELoss. `torch.compile` is overridden to identity so the patch is not bypassed by `OptimizedModule`. |

Nothing in the installed package is touched on disk.

### Adapting the ViT backbone for 32×32 input

The DINO checkpoint was trained at 224×224 with patch size 8, giving a 28×28 grid.
The training patches are 32×32 with patch size 4, giving an 8×8 grid (65 position
embeddings). Shapes are incompatible.

**Approach: discard incompatible weights, keep attention.** `modified_vit.py` builds
the ViT with `img_size=32, patch_size=4, pretrained=False`, loads the DINO checkpoint,
removes `pos_embed` and `patch_embed` keys, and calls `load_state_dict(..., strict=False)`.

| Component | Initialisation |
|-----------|---------------|
| `patch_embed` (4×4 conv) | Random — kernel changed from 8×8 |
| `pos_embed` (65 positions) | Random — grid changed from 28×28 to 8×8 |
| Attention weights | Loaded from DINO — frozen phase 1, fine-tuned at 3e-5 phase 2 |
| MLP layers, layer norms | Loaded from DINO — trained from epoch 0 |
| Decoder | Random — not part of DINO checkpoint |

### Training strategy

Two-phase schedule controlled by `FREEZE_EPOCHS` in `train.py`.

**Phase 1 — freeze attention** (`patch_optimizer_frozen`)

| Parameter group | Learning rate |
|-----------------|--------------|
| `patch_embed` | 5e-4 |
| `pos_embed` | 5e-4 |
| Decoder | 3e-4 |
| Everything else | 3e-4 |

**Phase 2 — train everything** (`patch_optimizer_strategy_b`)

| Parameter group | Learning rate |
|-----------------|--------------|
| `patch_embed` | 5e-4 |
| `pos_embed` | 5e-4 |
| Attention blocks | 3e-5 |
| Decoder | 3e-4 |
| Everything else | 3e-4 |

Set `FREEZE_EPOCHS = 0` to skip phase 1.

#### Why these learning rates and loss weights

Diagnostic analysis (using `coords_loss` and `perm_loss` from the CSV) showed the
decoder producing the same coordinate token for every input position after 30 epochs
— a classic mode collapse. `coords_loss` was flat at ~3.40, close to the random
baseline `log(35) ≈ 3.56`, while `perm_loss` continued falling. Two root causes:

- **Decoder LR too high (1e-3)**: the decoder converged to the lowest-loss constant
  before the encoder features stabilised. Reduced to 3e-4.
- **`vertex_loss_weight` too low (1.0)**: with perm_loss at ~0.05 and coords_loss at
  ~3.4, the coordinate gradient was overwhelmed. Raised to 10.0.

### Training optimisations

All inherited from the
[Pix2poly_P3_image_only](https://github.com/imanemn127/Pix2poly_P3_image_only) fork:

| Optimisation | Effect |
|---|---|
| Mixed precision (AMP) | Disabled — fp16 overflows in `log_sinkhorn_iterations`; entire forward runs in float32 |
| `torch.compile` | Disabled — wraps model in `OptimizedModule`, breaking the `type(model).forward` monkey-patch |
| Gradient clipping (norm 1.0) | Prevents divergence with long vertex sequences |
| BCELoss NaN guard | `nan_to_num` before clamp; safety net in case any NaN survives to the loss |
| Memory cleanup per epoch | `empty_cache()` + `gc.collect()` |

---

## Monitoring

```bash
watch -n 1 nvidia-smi
tail -f train_ai4smallfarms.log
```

Each epoch appends a row to `metrics.csv`:

| Column | Description |
|--------|-------------|
| `epoch` | Epoch index |
| `train_loss` | Total training loss |
| `val_loss` | Total validation loss |
| `val_iou` | Mask IoU on full val set (every `val_every` epochs, else `nan`); void images excluded |
| `train_iou` | Mask IoU on fixed 512-image train subset (every `val_every` epochs, else `nan`) |
| `coords_loss` | Coordinate cross-entropy × `vertex_loss_weight` |
| `perm_loss` | Permutation matrix BCE × `perm_loss_weight` |

A healthy run shows `coords_loss` decreasing from its epoch-0 value within the first
10 epochs. If it stays flat while `perm_loss` falls, the decoder is not learning
coordinates — check decoder LR and `vertex_loss_weight`.

```bash
python scripts/plot_losses_ai4sf.py                        # auto-detect latest run
python scripts/plot_losses_ai4sf.py /path/to/run/folder   # specific run
```

---

## Troubleshooting

**`RuntimeError: can't convert np.ndarray to tensor`**  
NumPy ≥ 2.0 is installed. Fix: `pip install "numpy==1.26.4"`.

**`FileNotFoundError` for checkpoint or dataset**  
Placeholder paths not set. Check `config/encoder/vit_s2.yaml` and `config/dataset/ai4smallfarms.yaml`.

**`ImportError: open3d` or any LiDAR-related error**  
Wrong P3 repo installed. Reinstall from `https://github.com/imanemn127/Pix2poly_P3_image_only`.

**ViT shape mismatch at startup**  
Always launch with `python train.py`, not by importing the trainer directly.

**`CUDA out of memory`**  
Lower `batch_size` in `config/run_type/ai4smallfarms.yaml`.

**Checkpoint resume fails with `unexpected key "_orig_mod.*"`**  
`torch.compile` wraps the model and prefixes all parameter keys with `_orig_mod.`.
If you save with a compiled model and load into an uncompiled one (or vice versa),
`load_state_dict` will complain. Fix: strip or add the prefix when loading, or always
compile before loading.

**`Assertion 'input_val >= zero && input_val <= one' failed` in `Loss.cu`**  
BCELoss CUDA kernel received NaN values. Root cause: `log_sinkhorn_iterations` uses
`logsumexp` which overflows in fp16, producing NaN in `perm_mat`. Fix: patch 10
forces the entire forward to float32. If the crash still occurs, check that
`torch.compile` is disabled (patch 10 also overrides it).

**`val_iou` is always NaN**  
`coords_loss` must drop below its epoch-0 value (~3.5 × `vertex_loss_weight`) before
meaningful predictions appear.

**`val_iou` is suspiciously stable (e.g. always ~0.051)**  
This is the void-image artifact from the original `calc_IoU` bug. Ensure patch 3c is
present in `train.py`.

**Training visualisations are very dark / near-black**  
`denormalize_s2` is not being applied. Confirm patch 4 is present and `train.py` is
the entry point.

**tmux session dies silently**  
Use the full Python path: `/path/to/envs/ai4sf/bin/python train.py`.

**`build_coco_dataset.py` finds no tiles**  
The raw AI4SmallFarms download uses `validate/`, not `val/`.

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
