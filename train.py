"""
AI4SmallFarms training script.
Patches applied at runtime – the PixelsPointsPolygons package is not modified.
"""
import sys
import os
import numpy as np

# --------------------------------------------------------------------
# Configurable constants
# --------------------------------------------------------------------
FREEZE_EPOCHS = 10       # number of epochs with ViT attention frozen
TRAIN_IOU_SUBSET = 512   # fixed subset size for train_iou estimate

# --------------------------------------------------------------------
# 1. Module / object replacements
# --------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1a. Custom dataset (p3_coco)
import p3_coco as local_p3_coco
import pixelspointspolygons.datasets.p3_coco as pkg_p3_coco
pkg_p3_coco.P3Dataset   = local_p3_coco.P3Dataset
pkg_p3_coco.TrainDataset = local_p3_coco.TrainDataset
pkg_p3_coco.ValDataset  = local_p3_coco.ValDataset
pkg_p3_coco.TestDataset = local_p3_coco.TestDataset

# 1b. Missing cv2 in build_datasets
import cv2
import pixelspointspolygons.datasets.build_datasets as build_datasets_module
build_datasets_module.cv2 = cv2

# 1c. Custom ViT for 32×32 input
import modified_vit
import pixelspointspolygons.models.vision_transformer as vit_module
vit_module.ViT = modified_vit.ViT
import pixelspointspolygons.models.pix2poly.model_pix2poly as model_pix2poly_module
model_pix2poly_module.ViT = modified_vit.ViT

# --------------------------------------------------------------------
# 2. COCO evaluator and prediction patches (field category = 1)
# --------------------------------------------------------------------
from copy import deepcopy
from pycocotools.cocoeval import COCOeval
import pixelspointspolygons.eval.evaluator as eval_module

original_compute_coco = eval_module.Evaluator.compute_coco_metrics
def patched_compute_coco(self, annType='segm'):
    cocoEval = COCOeval(deepcopy(self.cocoGt), deepcopy(self.cocoDt), iouType=annType)
    cocoEval.params.catIds = [1]          # field
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

# Force category_id = 1 in all COCO predictions
import pixelspointspolygons.misc.coco_conversions as coco_conv_module
original_generate_coco_ann = coco_conv_module.generate_coco_ann

def patched_generate_coco_ann(polygon_list, img_id, scores=None):
    anns = original_generate_coco_ann(polygon_list, img_id, scores)
    for ann in anns:
        ann['category_id'] = 1
    return anns

coco_conv_module.generate_coco_ann = patched_generate_coco_ann
import pixelspointspolygons.predict.predictor_pix2poly as pred_module
pred_module.generate_coco_ann = patched_generate_coco_ann

# --------------------------------------------------------------------
# 3. Sentinel-2 visualisation (percentile stretch denormalisation)
# --------------------------------------------------------------------
import pixelspointspolygons.misc.shared_utils as shared_utils

def denormalize_s2(image, cfg):
    """De-normalise S2 tensor to uint8 via p2-p98 percentile stretch."""
    image = image.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    mean = np.array(cfg.experiment.encoder.image_mean)
    std  = np.array(cfg.experiment.encoder.image_std)
    max_pixel = cfg.experiment.encoder.image_max_pixel_value

    image = (image * std + mean) * max_pixel
    image = np.clip(image, 0, max_pixel)

    out = np.zeros_like(image, dtype=np.uint8)
    for c in range(image.shape[2]):
        lo, hi = np.percentile(image[:, :, c], [2, 98])
        if hi > lo:
            out[:, :, c] = np.clip(
                (image[:, :, c] - lo) / (hi - lo) * 255, 0, 255
            ).astype(np.uint8)
    return out

shared_utils.denormalize_image_for_visualization = denormalize_s2

# --------------------------------------------------------------------
# 4. Import remaining helpers (Hydra, trainer, etc.)
# --------------------------------------------------------------------
import types
import hydra
from pixelspointspolygons.train import Pix2PolyTrainer
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

import pixelspointspolygons.train.trainer_pix2poly as trainer_module
trainer_module.denormalize_image_for_visualization = denormalize_s2

# --------------------------------------------------------------------
# 5. Decoder patch (higher dropout)
# --------------------------------------------------------------------
def patch_decoder(model):
    """Increase decoder dropout from 0.1 to 0.2 for regularisation."""
    dec = model.decoder
    for layer in dec.decoder.layers:
        layer.dropout.p = 0.2
        layer.self_attn.dropout = 0.2
        layer.multihead_attn.dropout = 0.2

# --------------------------------------------------------------------
# 6. Optimiser & scheduler utilities
# --------------------------------------------------------------------
def _rebuild_scheduler(trainer):
    """Replace the current scheduler with a fresh one tied to trainer.optimizer."""
    sched_keywords = trainer.lr_scheduler.lr_lambdas[0].keywords
    import warnings
    from transformers import get_linear_schedule_with_warmup
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
        trainer.lr_scheduler = get_linear_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=sched_keywords["num_warmup_steps"],
            num_training_steps=sched_keywords["num_training_steps"],
        )

def patch_optimizer_frozen(trainer):
    """Phase 1: freeze ViT attention (Q,K,V,proj), train everything else."""
    import torch.optim as optim
    model = trainer.model
    cfg   = trainer.cfg.experiment.model

    frozen_count = 0
    for name, param in model.encoder.vit.blocks.named_parameters():
        if ".attn." in name:
            param.requires_grad = False
            frozen_count += 1

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
    trainer.logger.info(
        f"Phase-1 optimiser (attn frozen: {frozen_count} tensors) "
        f"— groups: { {g['name']: len(g['params']) for g in param_groups} }"
    )
    trainer.optimizer = optim.AdamW(
        param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    _rebuild_scheduler(trainer)

def patch_optimizer_strategy_b(trainer):
    """Phase 2: train all parameters with differentiated learning rates."""
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
        {"params": patch_embed_params, "lr": 5e-4, "name": "patch_embed"},
        {"params": pos_embed_params,   "lr": 5e-4, "name": "pos_embed"},
        {"params": attn_params,        "lr": 3e-5, "name": "attn_blocks"},
        {"params": decoder_params,     "lr": 1e-3, "name": "decoder"},
        {"params": other_params,       "lr": cfg.learning_rate, "name": "other"},
    ]

    trainer.logger.info(
        f"Strategy B optimiser — groups: { {g['name']: len(g['params']) for g in param_groups} }"
    )
    trainer.optimizer = optim.AdamW(
        param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    _rebuild_scheduler(trainer)

# --------------------------------------------------------------------
# 7. Main entry point
# --------------------------------------------------------------------
@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)

    if cfg.experiment.model.name != "pix2poly":
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
    trainer = Pix2PolyTrainer(cfg, local_rank, world_size)

    def patched_train(self):
        import json, os, csv, torch
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

        # ----- Phase-1 (frozen attention) or directly Strategy B -----
        if FREEZE_EPOCHS > 0 and self.cfg.checkpoint is None:
            patch_optimizer_frozen(self)
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            patch_optimizer_strategy_b(self)

        # ----- Build fixed train_iou loader (val transforms, no augmentation) -----
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
                    A.Normalize(mean=self.cfg.experiment.encoder.image_mean,
                                std=self.cfg.experiment.encoder.image_std,
                                max_pixel_value=self.cfg.experiment.encoder.image_max_pixel_value)
                )
        _viz_transforms_list.append(ToTensorV2())
        _iou_transform = A.ReplayCompose(
            transforms=_viz_transforms_list,
            keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
        )

        _full_train_ds = _TrainDS(self.cfg, transform=_iou_transform, tokenizer=self.tokenizer)
        _n_iou = min(TRAIN_IOU_SUBSET, len(_full_train_ds))
        # ----- Select a fixed random subset (seed=0) so IoU is evaluated on the same images each epoch, spread across tiles -----

        _rng = np.random.default_rng(seed=0)
        _iou_indices = _rng.choice(len(_full_train_ds), size=_n_iou, replace=False).tolist()
        _iou_subset = Subset(_full_train_ds, _iou_indices)
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
            shuffle=False,
        )
        self.logger.info(
            f"train_iou_loader: {_n_iou} images (random subset seed=0, shuffle=False)"
        )

        # ----- Replace visualization: use raw COCO GT, skip non‑val epochs -----
        from collections import defaultdict
        from shapely.geometry import Polygon as _ShapelyPolygon
        import matplotlib.pyplot as _plt
        import wandb as _wandb
        from pixelspointspolygons.misc.shared_utils import (
            denormalize_image_for_visualization as _denorm,
            get_tile_names_from_dataloader as _get_tile_names,
        )
        from pixelspointspolygons.misc.debug_visualisations import (
            plot_image as _plot_image,
            plot_point_cloud as _plot_point_cloud,
            plot_shapely_polygons as _plot_shapely_polygons,
        )
        from pixelspointspolygons.misc.coco_conversions import (
            coco_anns_to_shapely_polys as _coco_anns_to_shapely_polys,
            tensor_to_shapely_polys as _tensor_to_shapely_polys,
        )

        def _patched_visualization(self_inner, loader, epoch, predictor, coco_anns=None, num_images=2):
            if (epoch + 1) % self_inner.cfg.training.val_every != 0:
                return

            self_inner.model.eval()

            ds = loader.dataset
            coco_obj = getattr(ds, 'coco', getattr(getattr(ds, 'dataset', None), 'coco', None))

            # Find the batch with the most polygons
            best_batch, best_count = None, -1
            for _viz_batch in loader:
                _eos = (_viz_batch[2] == self_inner.tokenizer.EOS_code).float().argmax(dim=-1)
                _n   = int((_eos > 1).sum().item())
                if _n > best_count:
                    best_count = _n
                    best_batch = _viz_batch
                if _n >= num_images:
                    break
            x_image, x_lidar, y_sequence, y_perm, tile_ids = best_batch

            _eos   = (y_sequence == self_inner.tokenizer.EOS_code).float().argmax(dim=-1)
            _has   = (_eos > 1).nonzero(as_tuple=True)[0].tolist()
            _empty = (_eos <= 1).nonzero(as_tuple=True)[0].tolist()
            _sel   = (_has + _empty)[:num_images]
            x_image    = x_image[_sel]
            y_sequence = y_sequence[_sel]
            y_perm     = y_perm[_sel]
            tile_ids   = tile_ids[_sel]

            if self_inner.cfg.experiment.encoder.use_images:
                x_image = x_image.to(self_inner.cfg.host.device, non_blocking=True)
            if self_inner.cfg.experiment.encoder.use_lidar:
                x_lidar = x_lidar.to(self_inner.cfg.host.device, non_blocking=True)
                x_lidar = list(x_lidar.unbind())[:num_images]
                x_lidar = torch.nested.nested_tensor(x_lidar, layout=torch.jagged)

            split   = loader.dataset.split
            outpath = os.path.join(self_inner.cfg.output_dir, "visualizations", split)
            os.makedirs(outpath, exist_ok=True)
            self_inner.logger.info(f"Save visualizations to {outpath}")

            if coco_anns is not None:
                coco_anns_dict = defaultdict(list)
                for ann in coco_anns:
                    if ann["image_id"] >= num_images:
                        break
                    coco_anns_dict[ann["image_id"]].append(ann)

            if predictor is not None:
                predicted_polygons = predictor.batch_to_polygons(
                    x_image, x_lidar, self_inner.model, self_inner.tokenizer
                )
                if coco_obj is not None:
                    gt_polygons = []
                    for img_id in tile_ids.view(-1).tolist():
                        img_info = coco_obj.imgs[int(img_id)]
                        W, H = img_info['width'], img_info['height']
                        polys = []
                        for ann in coco_obj.imgToAnns.get(int(img_id), []):
                            for seg in ann.get('segmentation', []):
                                pts = np.array(seg).reshape(-1, 2)
                                pts[:, 0] = np.clip(pts[:, 0], 0, W - 1e-6)
                                pts[:, 1] = np.clip(pts[:, 1], 0, H - 1e-6)
                                if len(pts) > 2:
                                    polys.append(_ShapelyPolygon(pts))
                        gt_polygons.append(polys)
                else:
                    gt_polygons = [
                        _tensor_to_shapely_polys(p)
                        for p in predictor.coord_and_perm_to_polygons(y_sequence, y_perm)
                    ]

            if self_inner.cfg.experiment.encoder.use_lidar:
                lidar_batches = torch.unbind(x_lidar, dim=0)

            names = _get_tile_names(loader, tile_ids.cpu().numpy().flatten().tolist())

            for i in range(num_images):
                fig, ax = _plt.subplots(1, 2, figsize=(8, 4), dpi=150)
                ax = ax.flatten()

                if self_inner.cfg.experiment.encoder.use_images:
                    image = _denorm(x_image[i], self_inner.cfg)
                    _plot_image(image, ax=ax[0])
                    _plot_image(image, ax=ax[1])
                if self_inner.cfg.experiment.encoder.use_lidar:
                    _plot_point_cloud(lidar_batches[i], ax=ax[0])
                    _plot_point_cloud(lidar_batches[i], ax=ax[1])

                coco_polys = _coco_anns_to_shapely_polys(coco_anns_dict[tile_ids[i].item()]) if coco_anns is not None else []
                if predictor is not None:
                    pred_polys = _tensor_to_shapely_polys(predicted_polygons[i])
                    gt_polys   = gt_polygons[i]
                else:
                    pred_polys, gt_polys = [], []

                if gt_polys:
                    _plot_shapely_polygons(gt_polys, ax=ax[0])
                if pred_polys and not coco_polys:
                    _plot_shapely_polygons(pred_polys, ax=ax[1])
                if coco_polys:
                    _plot_shapely_polygons(coco_polys, ax=ax[1])

                ax[0].set_title(f"GT_{split}_" + names[i])
                ax[1].set_title(f"PRED_{split}_" + names[i])
                _plt.tight_layout()
                w = len(str(self_inner.cfg.experiment.model.num_epochs))
                outfile = os.path.join(outpath, f"{epoch:0{w}d}_{names[i]}.png")
                _plt.savefig(outfile)
                if self_inner.cfg.run_type.log_to_wandb and self_inner.local_rank == 0:
                    _wandb.log({f"{epoch:0{w}d}: {split}_{names[i]}": _wandb.Image(fig)})
                _plt.close(fig)

        self.visualization = types.MethodType(_patched_visualization, self)

        # ----- Patch train_one_epoch for freeze→unfreeze transition -----
        _original_train_one_epoch = self.train_one_epoch.__func__

        def _patched_train_one_epoch(self_inner, epoch, iter_idx):
            if FREEZE_EPOCHS > 0 and epoch == FREEZE_EPOCHS and self_inner.cfg.checkpoint is None:
                self_inner.logger.info(
                    f"Epoch {epoch}: unfreezing ViT attention → rebuilding optimiser (Strategy B)."
                )
                for param in self_inner.model.parameters():
                    param.requires_grad = True
                patch_optimizer_strategy_b(self_inner)
            return _original_train_one_epoch(self_inner, epoch, iter_idx)

        self.train_one_epoch = types.MethodType(_patched_train_one_epoch, self)

        # ----- Hook save_best_and_latest_checkpoint to compute train_iou -----
        _original_save = self.save_best_and_latest_checkpoint.__func__

        def _patched_save(self_inner, epoch, val_loss_dict, val_metrics_dict):
            _original_save(self_inner, epoch, val_loss_dict, val_metrics_dict)

            if (epoch + 1) % self_inner.cfg.training.val_every != 0:
                self_inner._last_train_iou = float('nan')
                return

            self_inner.logger.info(f"Epoch {epoch}: computing train_iou on {_n_iou} images...")
            predictor = Predictor(self_inner.cfg, local_rank=self_inner.local_rank,
                                  world_size=self_inner.world_size)
            with torch.no_grad():
                train_preds = predictor.predict_from_loader(
                    self_inner.model, self_inner.tokenizer, train_iou_loader
                )

            if not train_preds:
                self_inner.logger.info("train_iou: no polygons → NaN")
                self_inner._last_train_iou = float('nan')
                return

            _train_eval = Evaluator(self_inner.cfg)
            _train_eval.load_gt(self_inner.cfg.experiment.dataset.annotations["train"])
            _tmp = os.path.join(self_inner.cfg.output_dir, "_tmp_train_iou.json")
            with open(_tmp, "w") as fp:
                json.dump(train_preds, fp)
            _train_eval.load_predictions(_tmp)
            _train_metrics = _train_eval.evaluate()
            os.remove(_tmp)

            self_inner._last_train_iou = _train_metrics.get('IoU', float('nan'))
            self_inner.logger.info(f"train_iou (epoch {epoch}): {self_inner._last_train_iou:.4f}")

        self.save_best_and_latest_checkpoint = types.MethodType(_patched_save, self)

        # ----- Inject train_iou column into metrics.csv via csv.writer wrapper -----
        import csv as _csv_module
        _OriginalWriter = _csv_module.writer
        _trainer_ref = self

        class _PatchedWriter:
            def __init__(self, f, *args, **kwargs):
                self._w = _OriginalWriter(f, *args, **kwargs)
            def writerow(self, row):
                row = list(row)
                if row and row[0] == 'epoch':
                    row.append('train_iou')
                else:
                    row.append(getattr(_trainer_ref, '_last_train_iou', float('nan')))
                self._w.writerow(row)
            def __getattr__(self, name):
                return getattr(self._w, name)

        _csv_module.writer = _PatchedWriter
        self._last_train_iou = float('nan')

        self.train_val_loop()

        _csv_module.writer = _OriginalWriter
        self.cleanup()

    trainer.train = types.MethodType(patched_train, trainer)
    trainer.train()

if __name__ == "__main__":
    main()