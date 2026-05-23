"""
Diagnostic script to figure out why the model was predicting zero polygons.
I used this to test:
  1) if EOS was predicted too early (blocking EOS for N steps)
  2) where EOS actually lands (valid / invalid positions)
  3) whether the coordinates and perm matrix are collapsing

It loads a checkpoint, runs inference on a few val batches,
and saves images to check everything visually.

Usage:
    conda run -n ai4sf python diag_eos.py
    EOS_MIN_STEPS=30 NUM_BATCHES=5 conda run -n ai4sf python diag_eos.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Same patches as train.py so the data and model load correctly ──────────
import p3_coco as local_p3_coco
import pixelspointspolygons.datasets.p3_coco as pkg_p3_coco
pkg_p3_coco.P3Dataset    = local_p3_coco.P3Dataset
pkg_p3_coco.TrainDataset = local_p3_coco.TrainDataset
pkg_p3_coco.ValDataset   = local_p3_coco.ValDataset
pkg_p3_coco.TestDataset  = local_p3_coco.TestDataset

import cv2
import pixelspointspolygons.datasets.build_datasets as build_datasets_module
build_datasets_module.cv2 = cv2

import modified_vit
import pixelspointspolygons.models.vision_transformer as vit_module
vit_module.ViT = modified_vit.ViT
import pixelspointspolygons.models.pix2poly.model_pix2poly as model_pix2poly_module
model_pix2poly_module.ViT = modified_vit.ViT

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPoly

import hydra
import pixelspointspolygons.misc.shared_utils as shared_utils
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf
from pixelspointspolygons.train import Pix2PolyTrainer
from pixelspointspolygons.predict.predictor_pix2poly import Pix2PolyPredictor
from pixelspointspolygons.misc.coco_conversions import tensor_to_shapely_polys


def denormalize_s2(image, cfg):
    """De‑normalize a Sentinel‑2 image back to uint8, using percentile stretch.
    Same as in train.py."""
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(cfg.experiment.encoder.image_mean)
    std  = np.array(cfg.experiment.encoder.image_std)
    image = (image * std + mean) * cfg.experiment.encoder.image_max_pixel_value
    image = np.clip(image, 0, image.max() + 1e-6)
    out = np.zeros_like(image, dtype=np.uint8)
    for c in range(image.shape[2]):
        lo, hi = np.percentile(image[:, :, c], [2, 98])
        if hi > lo:
            out[:, :, c] = np.clip(
                (image[:, :, c] - lo) / (hi - lo) * 255, 0, 255
            ).astype(np.uint8)
    return out

shared_utils.denormalize_image_for_visualization = denormalize_s2


# ── Infer with EOS blocked for the first N steps ──────────────────────────
def generate_with_eos_block(predictor, x_images, eos_min_steps):
    """Like test_generate, but we set EOS logit to -inf for the first
    eos_min_steps steps so the model can't stop too early."""
    tok    = predictor.tokenizer
    device = predictor.device
    model  = predictor.model
    cfg    = predictor.cfg

    batch_size  = x_images.size(0)
    batch_preds = torch.ones((batch_size, 1), device=device).fill_(tok.BOS_code).long()

    greedy = lambda p: torch.softmax(p, dim=-1).argmax(dim=-1).view(-1, 1)

    with torch.no_grad():
        features = model.encoder(x_images)

        for i in range(cfg.experiment.model.tokenizer.generation_steps):
            preds, feats = model.predict(features, batch_preds)

            if i < eos_min_steps:
                preds[:, tok.EOS_code] = float('-inf')

            batch_preds = torch.cat([batch_preds, greedy(preds)], dim=1)

        perm_preds = (
            model.scorenet1(feats) +
            torch.transpose(model.scorenet2(feats), 1, 2)
        )
        perm_preds = predictor.scores_to_permutations(perm_preds)

    return batch_preds.cpu(), perm_preds


def seq_stats(batch_preds, tok):
    """Return a list of dicts with eos position and number of coords for each
    item in the batch."""
    eos_idxs = (batch_preds == tok.EOS_code).float().argmax(dim=-1)
    return [
        {"eos_pos": int(eos), "n_coords": max(0, int(eos) - 1) // 2}
        for eos in eos_idxs.tolist()
    ]


def draw_polys(ax, polys_item, color, title):
    """Draw polygons from a list of Shapely polygons onto an axis."""
    ax.set_title(title, fontsize=8)
    if polys_item is None:
        return
    for poly in tensor_to_shapely_polys(polys_item):
        if poly.is_valid and poly.area > 0:
            x_c, y_c = poly.exterior.xy
            ax.plot(x_c, y_c, color=color, lw=1.5)
            ax.fill(x_c, y_c, alpha=0.2, color=color)


def analyze_eos_positions(cfg, checkpoint_path, num_batches=10):
    """Print stats about where the model places EOS — is it aligned? Is it too early?"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank, world_size = setup_ddp(cfg)
    trainer = Pix2PolyTrainer(cfg, local_rank, world_size)
    trainer.setup_model()
    trainer.setup_dataloader()

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", None)))
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.to(device)
    trainer.model.eval()

    predictor           = Pix2PolyPredictor(cfg, local_rank=0, world_size=1)
    predictor.model     = trainer.model
    predictor.device    = device
    predictor.tokenizer = trainer.tokenizer

    tok        = trainer.tokenizer
    val_loader = trainer.val_loader

    token_mode = tok.token_mode  # 2
    all_eos    = []

    print(f"\n{'='*60}")
    print(f"  EOS POSITION ANALYSIS — {num_batches} batches")
    print(f"  EOS_code={tok.EOS_code}  token_mode={token_mode}")
    print(f"  Valid EOS: position > 0 AND (pos-1) % {token_mode} == 0")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            x_image, _, y_seq, y_perm, tile_ids = batch
            x_image = x_image.to(device)

            preds, _, perm = predictor.test_generate(x_image, None)
            eos_idxs = (preds == tok.EOS_code).float().argmax(dim=-1).tolist()
            all_eos.extend(eos_idxs)

    all_eos = np.array(all_eos, dtype=int)
    valid   = (all_eos > 0) & ((all_eos - 1) % token_mode == 0)
    at_zero = all_eos == 0
    odd_pos = (all_eos > 0) & ((all_eos - 1) % token_mode != 0)

    n = len(all_eos)
    print(f"Total sequences : {n}")
    print(f"  EOS at pos 0 (no EOS or immediate): {at_zero.sum():>5}  ({100*at_zero.mean():.1f}%)")
    print(f"  EOS at odd position (misaligned):   {odd_pos.sum():>5}  ({100*odd_pos.mean():.1f}%)")
    print(f"  EOS valid (even pos > 0):           {valid.sum():>5}  ({100*valid.mean():.1f}%)")
    print()
    if valid.any():
        print(f"  Valid EOS — mean pos: {all_eos[valid].mean():.1f}  "
              f"min: {all_eos[valid].min()}  max: {all_eos[valid].max()}")
        print(f"  Valid EOS — median coords: {np.median((all_eos[valid]-1)//2):.0f}")
    print()
    nonzero = all_eos[all_eos > 0]
    if len(nonzero):
        buckets = [1, 5, 11, 21, 51, 101, 201, 386]
        print("  EOS position distribution (non-zero):")
        for lo, hi in zip(buckets[:-1], buckets[1:]):
            cnt = ((nonzero >= lo) & (nonzero < hi)).sum()
            bar = '#' * (cnt * 40 // max(len(nonzero), 1))
            print(f"    [{lo:>3}-{hi-1:>3}]: {cnt:>5}  {bar}")
    print(f"\n{'='*60}\n")
    return valid.mean()


def analyze_perm_matrix(cfg, checkpoint_path, num_images=4, out_dir="diag_perm_output"):
    """Look at the raw predicted vertex coordinates and the predicted perm matrix
    side‑by‑side with the GT. This is how I found the constant (0,0)/(31,31) collapse."""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank, world_size = setup_ddp(cfg)
    trainer = Pix2PolyTrainer(cfg, local_rank, world_size)
    trainer.setup_model()
    trainer.setup_dataloader()

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", None)))
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.to(device)
    trainer.model.eval()

    predictor           = Pix2PolyPredictor(cfg, local_rank=0, world_size=1)
    predictor.model     = trainer.model
    predictor.device    = device
    predictor.tokenizer = trainer.tokenizer

    tok        = trainer.tokenizer
    val_loader = trainer.val_loader
    coco       = getattr(val_loader.dataset, 'coco', None)

    print(f"\n{'='*72}")
    print(f"  PERM MATRIX & COORDINATE ANALYSIS — {num_images} images")
    print(f"  num_bins={tok.num_bins}  image_size={cfg.experiment.encoder.in_size}")
    print(f"{'='*72}\n")

    images_done = 0
    for batch in val_loader:
        if images_done >= num_images:
            break
        x_image, _, y_seq, y_perm_gt, tile_ids = batch
        x_image = x_image.to(device)

        with torch.no_grad():
            features = trainer.model.encoder(x_image)
            batch_preds = torch.ones((x_image.size(0), 1), device=device).fill_(tok.BOS_code).long()
            for _ in range(cfg.experiment.model.tokenizer.generation_steps):
                preds_logits, feats = trainer.model.predict(features, batch_preds)
                next_tok = torch.softmax(preds_logits, dim=-1).argmax(dim=-1).view(-1, 1)
                batch_preds = torch.cat([batch_preds, next_tok], dim=1)

            perm_scores_raw = (
                trainer.model.scorenet1(feats) +
                torch.transpose(trainer.model.scorenet2(feats), 1, 2)
            )
            perm_pred = predictor.scores_to_permutations(perm_scores_raw)

        for i in range(min(x_image.size(0), num_images - images_done)):
            tile_id = int(tile_ids[i].item())
            eos_pos = int((batch_preds[i] == tok.EOS_code).float().argmax().item())

            coord_seq = batch_preds[i].cpu()
            if eos_pos > 0 and (eos_pos - 1) % tok.token_mode == 0:
                pred_coords = tok.decode(coord_seq[:eos_pos + 1])
            else:
                pred_coords = np.empty((0, 2))

            gt_coords = []
            if coco is not None:
                for ann in coco.imgToAnns.get(tile_id, []):
                    for seg in ann.get('segmentation', []):
                        pts = np.array(seg).reshape(-1, 2)
                        gt_coords.append(pts)
            gt_coords_flat = np.vstack(gt_coords) if gt_coords else np.empty((0, 2))

            pm_pred = perm_pred[i].cpu().numpy()
            pm_gt   = y_perm_gt[i].cpu().numpy()

            pred_diag    = np.diag(pm_pred)
            gt_diag      = np.diag(pm_gt)
            n_active_pred = int((1 - pred_diag).sum())
            n_active_gt   = int((1 - gt_diag).sum())

            print(f"Image {images_done+1} / tile_id={tile_id}")
            print(f"  EOS at pos {eos_pos}  →  {len(pred_coords)} predicted vertices")
            print(f"  GT vertices total: {len(gt_coords_flat)}")
            if len(pred_coords):
                print(f"  Pred coords range: x=[{pred_coords[:,0].min():.1f}, {pred_coords[:,0].max():.1f}]  "
                      f"y=[{pred_coords[:,1].min():.1f}, {pred_coords[:,1].max():.1f}]")
            if len(gt_coords_flat):
                print(f"  GT   coords range: x=[{gt_coords_flat[:,0].min():.1f}, {gt_coords_flat[:,0].max():.1f}]  "
                      f"y=[{gt_coords_flat[:,1].min():.1f}, {gt_coords_flat[:,1].max():.1f}]")
            print(f"  Perm active vertices — pred: {n_active_pred}  GT: {n_active_gt}")
            print(f"  Perm pred density (non-diag 1s / N^2): "
                  f"{(pm_pred.sum() - np.trace(pm_pred)) / (pm_pred.shape[0]**2 - pm_pred.shape[0]):.4f}")
            print(f"  Perm GT   density (non-diag 1s / N^2): "
                  f"{(pm_gt.sum() - np.trace(pm_gt)) / (pm_gt.shape[0]**2 - pm_gt.shape[0]):.4f}")
            print()

            # ── Visualisation ──
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
            img_np = denormalize_s2(x_image[i].cpu(), cfg)

            axes[0].imshow(img_np)
            axes[0].set_title(f"GT (tile {tile_id}, {len(gt_coords_flat)} verts)", fontsize=8)
            for pts in gt_coords:
                if len(pts) > 2:
                    poly = ShapelyPoly(pts)
                    if poly.is_valid:
                        xc, yc = poly.exterior.xy
                        axes[0].plot(xc, yc, 'c-', lw=1.5)
                        axes[0].fill(xc, yc, alpha=0.15, color='cyan')
            axes[0].axis('off')

            axes[1].imshow(img_np)
            axes[1].set_title(f"Pred vertices (raw, {len(pred_coords)} pts)", fontsize=8)
            if len(pred_coords):
                axes[1].scatter(pred_coords[:, 0], pred_coords[:, 1],
                                c='red', s=20, zorder=5, alpha=0.7)
            axes[1].axis('off')

            N_show = min(max(n_active_gt, n_active_pred, 10), 50)
            pm_combined = np.zeros((N_show, N_show * 2 + 2))
            pm_combined[:, :N_show]           = pm_gt[:N_show, :N_show]
            pm_combined[:, N_show+2:]         = pm_pred[:N_show, :N_show]
            axes[2].imshow(pm_combined, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            axes[2].set_title(f"Perm matrix: GT (left) | Pred (right)  [{N_show}×{N_show}]", fontsize=8)
            axes[2].axis('off')

            plt.tight_layout()
            out_file = os.path.join(out_dir, f"perm_img{images_done+1:02d}_tile{tile_id}.png")
            plt.savefig(out_file)
            plt.close(fig)
            print(f"  → {out_file}\n")

            images_done += 1

    print(f"{'='*72}\n")


def run_diagnostic(cfg, checkpoint_path, eos_min_steps=20, num_batches=3, out_dir="diag_eos_output"):
    """Original EOS‑blocking test with side‑by‑side visual comparison."""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank, world_size = setup_ddp(cfg)
    trainer = Pix2PolyTrainer(cfg, local_rank, world_size)
    trainer.setup_model()
    trainer.setup_dataloader()

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", None)))
    if state is None:
        raise ValueError(
            f"No model weights found in checkpoint. Available keys: {list(ckpt.keys())}"
        )
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.to(device)
    trainer.model.eval()
    print(f"Checkpoint loaded: {checkpoint_path}\n")

    predictor         = Pix2PolyPredictor(cfg, local_rank=0, world_size=1)
    predictor.model   = trainer.model
    predictor.device  = device
    predictor.tokenizer = trainer.tokenizer

    tok        = trainer.tokenizer
    val_loader = trainer.val_loader
    print(f"Val loader : {len(val_loader)} batches\n")

    print(f"{'='*72}")
    print(f"  NORMAL vs EOS blocked ({eos_min_steps} steps)")
    print(f"{'='*72}\n")

    total_normal  = 0
    total_blocked = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_batches:
            break

        x_image, _, y_seq, y_perm, tile_ids = batch
        x_image = x_image.to(device)
        bs = x_image.size(0)

        preds_n, _, perm_n = predictor.test_generate(x_image, None)
        polys_n = predictor.coord_and_perm_to_polygons(preds_n, perm_n)
        stats_n = seq_stats(preds_n, tok)

        preds_b, perm_b = generate_with_eos_block(predictor, x_image, eos_min_steps)
        polys_b = predictor.coord_and_perm_to_polygons(preds_b, perm_b)
        stats_b = seq_stats(preds_b, tok)

        print(f"Batch {batch_idx+1}/{num_batches}  (BS={bs})")
        header = f"  {'Item':<5}  {'EOS_n':>6}  {'coords_n':>9}  {'polys_n':>7}  |  {'EOS_b':>6}  {'coords_b':>9}  {'polys_b':>7}"
        print(header)
        print(f"  {'-'*len(header.rstrip())}")
        for i in range(bs):
            n_pn = len(tensor_to_shapely_polys(polys_n[i])) if polys_n[i] else 0
            n_pb = len(tensor_to_shapely_polys(polys_b[i])) if polys_b[i] else 0
            total_normal  += n_pn
            total_blocked += n_pb
            print(f"  {i:<5}  {stats_n[i]['eos_pos']:>6}  {stats_n[i]['n_coords']:>9}  {n_pn:>7}  |  "
                  f"{stats_b[i]['eos_pos']:>6}  {stats_b[i]['n_coords']:>9}  {n_pb:>7}")
        print()

        coco = getattr(val_loader.dataset, 'coco', None)
        for i in range(min(2, bs)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
            img_np = denormalize_s2(x_image[i].cpu(), cfg)
            for ax in axes:
                ax.imshow(img_np)
                ax.axis("off")

            if coco is not None:
                img_id   = int(tile_ids[i].item())
                img_info = coco.imgs[img_id]
                W, H     = img_info['width'], img_info['height']
                for ann in coco.imgToAnns.get(img_id, []):
                    for seg in ann.get('segmentation', []):
                        pts = np.array(seg).reshape(-1, 2)
                        if len(pts) > 2:
                            poly = ShapelyPoly(pts)
                            if poly.is_valid and poly.area > 0:
                                xc, yc = poly.exterior.xy
                                axes[0].plot(xc, yc, 'c-', lw=1.5)
                                axes[0].fill(xc, yc, alpha=0.2, color='cyan')
            axes[0].set_title(f"GT (tile {tile_ids[i].item()})", fontsize=8)

            draw_polys(axes[1], polys_n[i], 'red',
                       f"Normal — eos@{stats_n[i]['eos_pos']}  coords={stats_n[i]['n_coords']}")
            draw_polys(axes[2], polys_b[i], 'lime',
                       f"EOS blocked {eos_min_steps}steps — coords={stats_b[i]['n_coords']}")

            plt.tight_layout()
            out_file = os.path.join(out_dir, f"batch{batch_idx:02d}_item{i:02d}.png")
            plt.savefig(out_file)
            plt.close(fig)
            print(f"  → {out_file}")

        print()

    print(f"{'='*72}")
    print(f"  SUMMARY — {num_batches} batches")
    print(f"  Normal polygons  : {total_normal}")
    print(f"  Blocked polygons : {total_blocked}")
    if total_blocked > total_normal:
        print("  ✓ Blocking EOS gives MORE polygons → premature EOS CONFIRMED")
    elif total_blocked == 0:
        print("  ✗ Still 0 polygons even with EOS blocked → deeper problem")
    else:
        print("  ~ Mixed results, check the visualizations")
    print(f"{'='*72}\n")


@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)

    eos_min_steps   = int(os.environ.get("EOS_MIN_STEPS", "20"))
    num_batches     = int(os.environ.get("NUM_BATCHES",   "3"))
    out_dir         = os.environ.get("DIAG_OUT", "diag_eos_output")
    checkpoint_path = os.environ.get(
        "CHECKPOINT",
        "/home/imane/DATA/AI4SmallFarms_output/pix2poly/32/"
        "v1_image_vit_bs4_ai4smallfarms/2026-05-23_07-23-06/"
        "checkpoints/latest.pth"
    )

    print(f"\n[diag_eos] checkpoint    = {checkpoint_path}")
    print(f"[diag_eos] eos_min_steps = {eos_min_steps}")
    print(f"[diag_eos] num_batches   = {num_batches}")
    print(f"[diag_eos] out_dir       = {out_dir}\n")

    # 1) Check where EOS lands
    analyze_eos_positions(cfg, checkpoint_path, num_batches=10)

    # 2) Look at the perm matrix and raw coords
    analyze_perm_matrix(cfg, checkpoint_path,
                        num_images=int(os.environ.get("PERM_IMAGES", "4")),
                        out_dir=os.environ.get("PERM_OUT", "diag_perm_output"))

    # 3) Compare normal vs EOS‑blocked inference
    run_diagnostic(cfg, checkpoint_path, eos_min_steps, num_batches, out_dir)


if __name__ == "__main__":
    main()