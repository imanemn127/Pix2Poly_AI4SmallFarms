import sys
import os
import numpy as np

# === AI4SmallFarms local patches ===
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. p3_coco safety patch
import p3_coco as local_p3_coco
import pixelspointspolygons.datasets.p3_coco as pkg_p3_coco
pkg_p3_coco.P3Dataset   = local_p3_coco.P3Dataset
pkg_p3_coco.TrainDataset = local_p3_coco.TrainDataset
pkg_p3_coco.ValDataset  = local_p3_coco.ValDataset
pkg_p3_coco.TestDataset = local_p3_coco.TestDataset

# 2. ViT interpolation patch (64x64)
import modified_vit
import pixelspointspolygons.models.vision_transformer as vit_module
vit_module.ViT = modified_vit.ViT
import pixelspointspolygons.models.pix2poly.model_pix2poly as model_pix2poly_module
model_pix2poly_module.ViT = modified_vit.ViT

# 3. Patch evaluator category for AI4SmallFarms (field = 1, not building 100)
from copy import deepcopy
from pycocotools.cocoeval import COCOeval
import pixelspointspolygons.eval.evaluator as eval_module

original_compute_coco = eval_module.Evaluator.compute_coco_metrics
def patched_compute_coco(self, annType='segm'):
    cocoEval = COCOeval(deepcopy(self.cocoGt), deepcopy(self.cocoDt), iouType=annType)
    cocoEval.params.catIds = [1]          # field category
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return {
        'AP': cocoEval.stats[0], 'AP50': cocoEval.stats[1],
        'AP75': cocoEval.stats[2], 'AP_small': cocoEval.stats[3],
        'AP_medium': cocoEval.stats[4], 'AP_large': cocoEval.stats[5],
        'AR1': cocoEval.stats[6], 'AR10': cocoEval.stats[7],
        'AR100': cocoEval.stats[8], 'AR_small': cocoEval.stats[9],
        'AR_medium': cocoEval.stats[10], 'AR_large': cocoEval.stats[11]
    }
eval_module.Evaluator.compute_coco_metrics = patched_compute_coco

# Fix category_id in predictions (field=1, not building=100)
import pixelspointspolygons.misc.coco_conversions as coco_conv_module
original_generate_coco_ann = coco_conv_module.generate_coco_ann

def patched_generate_coco_ann(polygon_list, img_id, scores=None):
    anns = original_generate_coco_ann(polygon_list, img_id, scores)
    for ann in anns:
        ann['category_id'] = 1
    return anns

coco_conv_module.generate_coco_ann = patched_generate_coco_ann
# Also patch the reference imported in predictor_pix2poly
import pixelspointspolygons.predict.predictor_pix2poly as pred_module
pred_module.generate_coco_ann = patched_generate_coco_ann

# 4. Fix visualization for Sentinel-2 uint16 images
import pixelspointspolygons.misc.shared_utils as shared_utils

original_denorm = shared_utils.denormalize_image_for_visualization

def denormalize_s2(image, cfg):
    """Denormalize a Sentinel-2 image tensor to a displayable uint8 numpy array."""
    image = image.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    mean = cfg.experiment.encoder.image_mean
    std  = cfg.experiment.encoder.image_std
    max_pixel = cfg.experiment.encoder.image_max_pixel_value

    # Inverse of Normalize: pixel = (value * std + mean) * max_pixel
    image = image * std + mean
    image = image * max_pixel

    # Clip to valid range and rescale to [0, 255]
    image = np.clip(image, 0, max_pixel)
    image = (image / max_pixel * 255).astype(np.uint8)

    return image

shared_utils.denormalize_image_for_visualization = denormalize_s2
# ============================================
import hydra
from pixelspointspolygons.train import Pix2PolyTrainer
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf   # ← ajouter cette ligne

# Patch trainer for visualisation
import pixelspointspolygons.train.trainer_pix2poly as trainer_module
trainer_module.denormalize_image_for_visualization = denormalize_s2

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
