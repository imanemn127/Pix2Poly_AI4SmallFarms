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

# 2. Fix missing cv2 import in build_datasets
import cv2
import pixelspointspolygons.datasets.build_datasets as build_datasets_module
build_datasets_module.cv2 = cv2

# 3. ViT interpolation patch (64x64)
import modified_vit
import pixelspointspolygons.models.vision_transformer as vit_module
vit_module.ViT = modified_vit.ViT
import pixelspointspolygons.models.pix2poly.model_pix2poly as model_pix2poly_module
model_pix2poly_module.ViT = modified_vit.ViT

# 4. Patch evaluator category for AI4SmallFarms (field = 1, not building 100)
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

# 5. Fix visualization for Sentinel-2 uint16 images
import pixelspointspolygons.misc.shared_utils as shared_utils

original_denorm = shared_utils.denormalize_image_for_visualization

def denormalize_s2(image, cfg):
    """Denormalize a Sentinel-2 image tensor to a displayable uint8 numpy array.
    Uses per-channel p2-p98 percentile stretch because S2 reflectance clusters
    in a narrow low range — linear /max_pixel produces near-black images.
    """
    image = image.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    mean = np.array(cfg.experiment.encoder.image_mean)
    std  = np.array(cfg.experiment.encoder.image_std)
    max_pixel = cfg.experiment.encoder.image_max_pixel_value

    # Inverse of Normalize: pixel = (value * std + mean) * max_pixel
    image = image * std + mean
    image = image * max_pixel
    image = np.clip(image, 0, max_pixel)

    # Per-channel percentile stretch so dark S2 imagery becomes visible
    out = np.zeros_like(image, dtype=np.uint8)
    for c in range(image.shape[2]):
        lo, hi = np.percentile(image[:, :, c], [2, 98])
        if hi > lo:
            out[:, :, c] = np.clip(
                (image[:, :, c] - lo) / (hi - lo) * 255, 0, 255
            ).astype(np.uint8)
    return out

shared_utils.denormalize_image_for_visualization = denormalize_s2
# ============================================
import types
import torch.nn as nn
import hydra
from pixelspointspolygons.train import Pix2PolyTrainer
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

# Patch trainer for visualisation
import pixelspointspolygons.train.trainer_pix2poly as trainer_module
trainer_module.denormalize_image_for_visualization = denormalize_s2


def patch_decoder(model):
    """Only bump dropout — no forward/predict wrapping to avoid train/inference mismatch."""
    dec = model.decoder

    for layer in dec.decoder.layers:
        layer.dropout.p = 0.2
        layer.self_attn.dropout = 0.2
        layer.multihead_attn.dropout = 0.2

def _rebuild_scheduler(trainer):
    # the warmup/total steps are stored in the lambda's keyword closure
    sched_keywords = trainer.lr_scheduler.lr_lambdas[0].keywords
    import warnings
    from transformers import get_linear_schedule_with_warmup
    # LambdaLR calls step() once during __init__, which triggers a spurious
    # "scheduler before optimizer" warning because the new optimizer has no
    # prior steps yet. Suppress it locally — the order is correct at runtime.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
        trainer.lr_scheduler = get_linear_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=sched_keywords["num_warmup_steps"],
            num_training_steps=sched_keywords["num_training_steps"],
        )


def patch_optimizer_strategy_a(trainer):
    """Freeze ViT attention blocks. Separate LRs for patch embed and decoder."""
    import torch.optim as optim
    model = trainer.model
    cfg = trainer.cfg.experiment.model

    # freeze attention blocks — DINO features are strong, protect them
    for param in model.encoder.vit.blocks.parameters():
        param.requires_grad = False

    patch_embed_params = list(model.encoder.vit.patch_embed.parameters())
    pos_embed_params   = [model.encoder.vit.pos_embed]  # nn.Parameter, not a module
    decoder_params     = list(model.decoder.parameters())
    other_params       = [
        p for p in model.parameters()
        if p.requires_grad
        and not any(p is q for q in patch_embed_params + pos_embed_params + decoder_params)
    ]

    trainer.optimizer = optim.AdamW([
        {"params": patch_embed_params, "lr": 5e-4},
        {"params": pos_embed_params,   "lr": 5e-4},
        {"params": decoder_params,     "lr": 1e-3},
        {"params": other_params,       "lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    _rebuild_scheduler(trainer)


def patch_optimizer_strategy_b(trainer):
    """Train all parts with separate LRs — no freezing.

    Parameter group  |  LR
    patch_embed      |  5e-4  (re-learn spatial tokens for new 32x32 resolution)
    pos_embed        |  5e-4  (discarded from pretraining, needs to be learned)
    attn_blocks      |  1e-5  (preserve DINO representations, adapt slowly)
    decoder          |  1e-3  (freshly initialised, train aggressively)
    other            |  3e-4  (MLP, norm, bottleneck, scorenets)
    """
    import torch.optim as optim
    model = trainer.model
    cfg = trainer.cfg.experiment.model

    patch_embed_params, pos_embed_params, attn_params, decoder_params, other_params = [], [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder.vit.patch_embed" in name:
            patch_embed_params.append(param)
        elif name == "encoder.vit.pos_embed":
            pos_embed_params.append(param)
        elif "encoder.vit.blocks" in name and ".attn." in name:
            attn_params.append(param)
        elif "decoder." in name:
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": patch_embed_params, "lr": 5e-4,           "name": "patch_embed"},
        {"params": pos_embed_params,   "lr": 5e-4,           "name": "pos_embed"},
        {"params": attn_params,        "lr": 1e-5,           "name": "attn_blocks"},
        {"params": decoder_params,     "lr": 1e-3,           "name": "decoder"},
        {"params": other_params,       "lr": cfg.learning_rate, "name": "other"},
    ]

    counts = {g["name"]: len(g["params"]) for g in param_groups}
    trainer.logger.info(f"Optimizer strategy B — param groups: {counts}")

    trainer.optimizer = optim.AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    _rebuild_scheduler(trainer)


@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)

    if cfg.experiment.model.name == "pix2poly":
        trainer = Pix2PolyTrainer(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")

    # trainer.model is None here — it is only built inside trainer.train().
    # We override train() to intercept the moment after setup_model() and
    # setup_optimizer() have run, so our patches see a fully built model.
    def patched_train(self):
        from pixelspointspolygons.misc import seed_everything
        seed_everything(42)
        self.setup_model()       # model is built here
        self.setup_dataloader()
        self.setup_optimizer()   # optimizer + scheduler built here
        self.setup_loss_fn_dict()

        # now model and optimizer exist — safe to patch
        patch_decoder(self.model)
        patch_optimizer_strategy_b(self)   # swap to _a when ready

        self.train_val_loop()
        self.cleanup()

    trainer.train = types.MethodType(patched_train, trainer)
    trainer.train()

if __name__ == "__main__":
    main()
