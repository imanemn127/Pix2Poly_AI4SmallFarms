import sys
import os
import numpy as np

# All AI4SmallFarms-specific patches go here. Nothing in the installed
# PixelsPointsPolygons package is ever touched on disk — everything is
# replaced at runtime via Python's module reference system.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Two-phase fine-tuning schedule.
# Phase 1 (epochs 0 … FREEZE_EPOCHS-1): only the ViT attention weights are
# frozen (Q/K/V + projection). patch_embed, pos_embed, MLP layers, norms and
# the decoder are trained normally from the start.
# Phase 2 (epoch FREEZE_EPOCHS onward): attention unfrozen, optimizer rebuilt
# with differentiated LRs (Strategy B), model recompiled.
# Set FREEZE_EPOCHS = 0 to skip phase 1 and go straight to Strategy B.
FREEZE_EPOCHS = 10   # roughly 10 % of a 100-epoch run — tune as needed

# How many train images to use for the train_iou estimate every val_every
# epochs. A fixed subset (shuffle=False) so we compare the same images
# across epochs rather than a random sample.
TRAIN_IOU_SUBSET = 128

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
    """Convert a normalized S2 tensor back to a displayable uint8 image.

    The standard linear /max_pixel rescaling produces near-black images because
    S2 reflectance values cluster in the bottom 20-30% of the uint16 range.
    Using a per-channel p2-p98 percentile stretch makes the visualizations
    actually readable without touching the training normalization.
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
import hydra
from pixelspointspolygons.train import Pix2PolyTrainer
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

# Patch trainer for visualisation
import pixelspointspolygons.train.trainer_pix2poly as trainer_module
trainer_module.denormalize_image_for_visualization = denormalize_s2

# Fix GT polygon misalignment in saved visualization PNGs.
# Two things were wrong:
#   1. ax.imshow() without an explicit origin= falls back to rcParams, which
#      can be 'lower' depending on the matplotlib install.  That flips the
#      Y-axis and makes the polygon overlays appear vertically mirrored.
#   2. Vertices decoded from tokens can reach exactly `width` or `height`
#      (the dequantize of the last bin), which falls outside the default
#      imshow extent and gets clipped at the border.
# The fix is to always pass origin='upper' and set the extent explicitly so
# the image pixel grid and the polygon coordinate system agree exactly.
import pixelspointspolygons.misc.debug_visualisations as _dbgviz

_original_plot_image = _dbgviz.plot_image

def _patched_plot_image(image, ax=None, show_axis=False, show=False):
    import torch, numpy as np, matplotlib.pyplot as plt
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    ax.axis(show_axis)
    h, w = image.shape[:2]
    # origin='upper' locks row-0 to the top regardless of rcParams.
    # The half-pixel margin in extent keeps vertices at exactly w or h
    # from being clipped at the image border.
    ax.imshow(image, origin='upper', extent=[-0.5, w - 0.5, h - 0.5, -0.5])
    if show:
        plt.show(block=False)

_dbgviz.plot_image = _patched_plot_image
# Also patch the reference already imported in trainer_pix2poly
trainer_module.plot_image = _patched_plot_image


def patch_decoder(model):
    """Increase decoder dropout to 0.2 (was 0.1) to reduce overfitting on the small dataset."""
    dec = model.decoder

    for layer in dec.decoder.layers:
        layer.dropout.p = 0.2
        layer.self_attn.dropout = 0.2
        layer.multihead_attn.dropout = 0.2

def _rebuild_scheduler(trainer, last_epoch=-1):
    """Rebuild the warmup-then-linear-decay scheduler on the current optimizer.

    last_epoch: number of optimizer steps already done. Passing -1 (default)
    restarts the warmup from scratch. Passing the actual step count at the
    phase-2 transition avoids the LR spiking back to the warmup peak.
    """
    sched_keywords = trainer.lr_scheduler.lr_lambdas[0].keywords
    import warnings
    from transformers import get_linear_schedule_with_warmup
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
        trainer.lr_scheduler = get_linear_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=sched_keywords["num_warmup_steps"],
            num_training_steps=sched_keywords["num_training_steps"],
            last_epoch=last_epoch,
        )


def patch_optimizer_frozen(trainer):
    """Set up the phase-1 optimizer: freeze ViT attention, train everything else.

    I only freeze the '.attn.' parameters inside encoder.vit.blocks (Q, K, V
    and output projection). MLP layers, layer norms, patch_embed, pos_embed
    and the decoder all stay trainable from epoch 0. The idea is to preserve
    DINO attention weights while the new patch embedding and decoder warm up.
    """
    import torch.optim as optim
    model = trainer.model
    cfg   = trainer.cfg.experiment.model

    # Freeze only attention parameters inside ViT blocks.
    frozen_count = 0
    for name, param in model.encoder.vit.blocks.named_parameters():
        if ".attn." in name:
            param.requires_grad = False
            frozen_count += 1

    # Build param groups identical to Strategy B but without the frozen attn group.
    patch_embed_params, pos_embed_params, decoder_params, other_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder.vit.patch_embed" in name:
            patch_embed_params.append(param)
        elif name == "encoder.vit.pos_embed":
            pos_embed_params.append(param)
        elif "decoder." in name:
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": patch_embed_params, "lr": 5e-4, "name": "patch_embed"},
        {"params": pos_embed_params,   "lr": 5e-4, "name": "pos_embed"},
        {"params": decoder_params,     "lr": 1e-3, "name": "decoder"},
        {"params": other_params,       "lr": cfg.learning_rate, "name": "other"},
    ]
    counts = {g["name"]: len(g["params"]) for g in param_groups}
    trainer.logger.info(
        f"Phase-1 optimizer (attn blocks frozen: {frozen_count} tensors) "
        f"— param groups: {counts}"
    )
    trainer.optimizer = optim.AdamW(
        param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    # last_epoch=-1: scheduler starts from step 0 (warmup), correct for phase 1.
    _rebuild_scheduler(trainer, last_epoch=-1)


def patch_optimizer_strategy_b(trainer, last_epoch=-1):
    """Train everything with differentiated learning rates (no freezing).

    Parameter group  |  LR
    patch_embed      |  5e-4   (has to learn new 2x2 conv for 32px input)
    pos_embed        |  5e-4   (was discarded at checkpoint load, learned from scratch)
    attn_blocks      |  1e-5   (keep DINO features, just nudge them toward S2)
    decoder          |  1e-3   (freshly initialized, can afford a higher LR)
    other            |  cfg.lr (MLP, norms, etc.)

    last_epoch: pass -1 to start a fresh warmup, or the number of optimizer
    steps already done to continue the decay curve from the phase-1 transition.
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

    _rebuild_scheduler(trainer, last_epoch=last_epoch)


@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)

    if cfg.experiment.model.name == "pix2poly":
        trainer = Pix2PolyTrainer(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")

    # trainer.model doesn't exist yet at this point — it gets built inside
    # trainer.train(). Overriding train() lets the patches hook in right
    # after setup_model() and setup_optimizer() have both run.
    def patched_train(self):
        import json, csv
        import torch
        from functools import partial
        from torch.utils.data import DataLoader, Subset
        from pixelspointspolygons.misc import seed_everything
        from pixelspointspolygons.datasets.build_datasets import get_collate_fn
        from pixelspointspolygons.predict.predictor_pix2poly import Pix2PolyPredictor as Predictor
        from pixelspointspolygons.eval import Evaluator

        seed_everything(42)
        self.setup_model()
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_loss_fn_dict()

        patch_decoder(self.model)

        # Phase-1 setup: freeze attention or go straight to Strategy B.
        if FREEZE_EPOCHS > 0:
            patch_optimizer_frozen(self)
        else:
            patch_optimizer_strategy_b(self)

        # Build a dedicated loader for the periodic train_iou estimate.
        # train_viz_loader is not reused here because it may have fewer images.
        # Using val-style transforms (no augmentation) so the predictor input
        # matches inference conditions. shuffle=False keeps the same images
        # every val_every epochs for a fair comparison across runs.
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from pixelspointspolygons.datasets.p3_coco import TrainDataset as _TrainDS

        _viz_transforms_list = []
        if self.cfg.experiment.encoder.augmentations is not None:
            if "Resize" in self.cfg.experiment.encoder.augmentations:
                _viz_transforms_list.append(
                    A.Resize(height=self.cfg.experiment.encoder.in_height,
                             width=self.cfg.experiment.encoder.in_width)
                )
            if "Normalize" in self.cfg.experiment.encoder.augmentations:
                _viz_transforms_list.append(
                    A.Normalize(
                        mean=self.cfg.experiment.encoder.image_mean,
                        std=self.cfg.experiment.encoder.image_std,
                        max_pixel_value=self.cfg.experiment.encoder.image_max_pixel_value,
                    )
                )
        _viz_transforms_list.append(ToTensorV2())
        _iou_transform = A.ReplayCompose(
            transforms=_viz_transforms_list,
            keypoint_params=A.KeypointParams(format='yx', remove_invisible=False),
        )

        _full_train_ds = _TrainDS(self.cfg, transform=_iou_transform,
                                  tokenizer=self.tokenizer)
        _n_iou = min(TRAIN_IOU_SUBSET, len(_full_train_ds))
        _iou_subset = Subset(_full_train_ds, list(range(_n_iou)))
        # The Subset wrapper drops dataset-level attributes; copy them over
        # so predict_from_loader and the Evaluator can still find them.
        _iou_subset.ann_file = _full_train_ds.ann_file
        _iou_subset.coco     = _full_train_ds.coco
        _iou_subset.split    = _full_train_ds.split

        _collate = partial(get_collate_fn(self.cfg.experiment.model.name), cfg=self.cfg)
        train_iou_loader = DataLoader(
            _iou_subset,
            batch_size=self.cfg.experiment.model.batch_size,
            collate_fn=_collate,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.run_type.name != 'debug',
            drop_last=False,
            shuffle=False,   # fixed subset — must not shuffle
        )
        self.logger.info(
            f"train_iou_loader: {_n_iou} images "
            f"(first {_n_iou} of train set, shuffle=False)"
        )

        # Gate visualization() to val_every epochs.
        # The original trainer calls visualization() every single epoch
        # (trainer_pix2poly.py line 504), which runs the full autoregressive
        # decoder — same cost as COCO eval. Skipping it on non-val epochs
        # noticeably speeds up training without losing useful information.
        _original_visualization = self.visualization.__func__  # unbound

        def _gated_visualization(self_inner, loader, epoch, **kwargs):
            if (epoch + 1) % self_inner.cfg.training.val_every != 0:
                return  # skip — not a validation epoch
            _original_visualization(self_inner, loader, epoch, **kwargs)

        self.visualization = types.MethodType(_gated_visualization, self)

        # Wrap train_one_epoch to inject the freeze→unfreeze transition.
        # At epoch FREEZE_EPOCHS: unfreeze attention, rebuild optimizer with
        # Strategy B, recompile the model.
        # GradScaler is intentionally not rebuilt: it lives for the full run
        # and resetting it at this point would reset the scale factor that has
        # already been adjusted to the gradient magnitudes.
        _original_train_one_epoch = self.train_one_epoch.__func__  # unbound

        def _patched_train_one_epoch(self_inner, epoch, iter_idx):
            if FREEZE_EPOCHS > 0 and epoch == FREEZE_EPOCHS:
                self_inner.logger.info(
                    f"Epoch {epoch}: unfreezing ViT attention blocks → "
                    "rebuilding optimizer with Strategy B."
                )
                # Re-enable requires_grad on all parameters.
                for param in self_inner.model.parameters():
                    param.requires_grad = True
                # iter_idx is the total number of batches processed so far,
                # which equals the number of optimizer steps (no grad accum).
                # Passing it as last_epoch lets the scheduler pick up its
                # decay curve from the right point instead of restarting.
                steps_done = iter_idx
                patch_optimizer_strategy_b(self_inner, last_epoch=steps_done)
                # Recompile so the graph includes the newly-trainable parameters.
                self_inner.model = torch.compile(self_inner.model, mode="default")

            return _original_train_one_epoch(self_inner, epoch, iter_idx)

        self.train_one_epoch = types.MethodType(_patched_train_one_epoch, self)

        # Hook save_best_and_latest_checkpoint to compute train_iou.
        # This is the only method called once per epoch that runs after COCO
        # eval, making it a natural place to append the train IoU measurement.
        # The result is stored in self._last_train_iou and picked up by the
        # CSV writer below. Original checkpoint logic runs first, unchanged.
        _original_save = self.save_best_and_latest_checkpoint.__func__

        def _patched_save(self_inner, epoch, val_loss_dict, val_metrics_dict):
            _original_save(self_inner, epoch, val_loss_dict, val_metrics_dict)

            if (epoch + 1) % self_inner.cfg.training.val_every != 0:
                self_inner._last_train_iou = float('nan')
                return

            self_inner.logger.info(
                f"Epoch {epoch}: computing train_iou on "
                f"{_n_iou}-image subset..."
            )
            predictor = Predictor(
                self_inner.cfg,
                local_rank=self_inner.local_rank,
                world_size=self_inner.world_size,
            )
            with torch.no_grad():
                train_preds = predictor.predict_from_loader(
                    self_inner.model, self_inner.tokenizer, train_iou_loader
                )

            if not train_preds:
                self_inner.logger.info("train_iou: no polygons predicted → NaN")
                self_inner._last_train_iou = float('nan')
                return

            _train_eval = Evaluator(self_inner.cfg)
            _train_eval.load_gt(
                self_inner.cfg.experiment.dataset.annotations["train"]
            )
            _tmp = os.path.join(self_inner.cfg.output_dir, "_tmp_train_iou.json")
            with open(_tmp, "w") as fp:
                json.dump(train_preds, fp)
            _train_eval.load_predictions(_tmp)
            _train_metrics = _train_eval.evaluate()
            os.remove(_tmp)

            self_inner._last_train_iou = _train_metrics.get('IoU', float('nan'))
            self_inner.logger.info(
                f"train_iou (epoch {epoch}): {self_inner._last_train_iou:.4f}"
            )

        self.save_best_and_latest_checkpoint = types.MethodType(
            _patched_save, self
        )

        # Inject the train_iou column into metrics.csv.
        # train_val_loop writes the CSV in a local scope so there is no clean
        # hook point for adding a column. The approach here is to swap out
        # csv.writer at the module level with a thin wrapper that appends the
        # extra value to every row. The header row gets a column name; data
        # rows get the last computed self._last_train_iou value.
        import csv as _csv_module

        _OriginalWriter = _csv_module.writer

        _trainer_ref = self  # capture for the closure

        class _PatchedWriter:
            """csv.writer wrapper that appends the train_iou column to every row."""
            def __init__(self, f, *args, **kwargs):
                self._w = _OriginalWriter(f, *args, **kwargs)

            def writerow(self, row):
                row = list(row)
                if row and row[0] == 'epoch':
                    # header row — add column name
                    row.append('train_iou')
                else:
                    # data row — append current train_iou value
                    row.append(getattr(_trainer_ref, '_last_train_iou', float('nan')))
                self._w.writerow(row)

            def __getattr__(self, name):
                return getattr(self._w, name)

        _csv_module.writer = _PatchedWriter

        self._last_train_iou = float('nan')  # initialise before loop starts

        self.train_val_loop()

        # Restore original csv.writer after training ends.
        _csv_module.writer = _OriginalWriter

        self.cleanup()

    trainer.train = types.MethodType(patched_train, trainer)
    trainer.train()

if __name__ == "__main__":
    main()
