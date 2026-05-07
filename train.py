import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Make local modules (p3_coco.py, modified_vit.py) importable before anything
# in pixelspointspolygons tries to import them.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Patch 1: Dataset classes
# Replace the installed P3Dataset and its Train/Val/Test subclasses with the
# local versions from p3_coco.py.  The local classes add _safe_to_tensor()
# (NumPy 2.x compatibility) and remove LiDAR loading.
# ---------------------------------------------------------------------------
import p3_coco as local_p3_coco
import pixelspointspolygons.datasets.p3_coco as pkg_p3_coco

pkg_p3_coco.P3Dataset    = local_p3_coco.P3Dataset
pkg_p3_coco.TrainDataset = local_p3_coco.TrainDataset
pkg_p3_coco.ValDataset   = local_p3_coco.ValDataset
pkg_p3_coco.TestDataset  = local_p3_coco.TestDataset

# ---------------------------------------------------------------------------
# Patch 2: ViT backbone
# Replace ViT in both the vision_transformer module and the model_pix2poly
# module with the local ModifiedViT from modified_vit.py, which builds the
# model at img_size=64 and discards pretrained positional embeddings so they
# are learned from scratch.
# ---------------------------------------------------------------------------
import modified_vit
import pixelspointspolygons.models.vision_transformer as vit_module
import pixelspointspolygons.models.pix2poly.model_pix2poly as model_pix2poly_module

vit_module.ViT           = modified_vit.ViT
model_pix2poly_module.ViT = modified_vit.ViT

# ---------------------------------------------------------------------------
# Patch 3: Visualisation denormalisation
# The trainer calls denormalize_image_for_visualization from its own module
# namespace, so we must replace it in both shared_utils and trainer_pix2poly.
# The default function assumes uint8 RGB images; we need a Sentinel-2-aware
# version that inverts the [0, 1] Normalize transform and rescales to uint8.
# ---------------------------------------------------------------------------
def denormalize_s2(image, cfg):
    """Denormalise a Sentinel-2 image tensor to a displayable uint8 numpy array."""
    image     = image.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    mean      = cfg.experiment.encoder.image_mean
    std       = cfg.experiment.encoder.image_std
    max_pixel = cfg.experiment.encoder.image_max_pixel_value

    # Inverse of Normalize: pixel = (value * std + mean) * max_pixel
    image = image * std + mean
    image = image * max_pixel
    image = np.clip(image, 0, max_pixel)
    image = (image / max_pixel * 255).astype(np.uint8)
    return image

import pixelspointspolygons.misc.shared_utils as shared_utils
import pixelspointspolygons.train.trainer_pix2poly as trainer_module

shared_utils.denormalize_image_for_visualization  = denormalize_s2
trainer_module.denormalize_image_for_visualization = denormalize_s2

# ---------------------------------------------------------------------------
# Standard training entry point
# ---------------------------------------------------------------------------
import hydra
from pixelspointspolygons.train import Pix2PolyTrainer
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf


@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)

    if cfg.experiment.model.name == "pix2poly":
        trainer = Pix2PolyTrainer(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")

    trainer.train()


if __name__ == "__main__":
    main()
