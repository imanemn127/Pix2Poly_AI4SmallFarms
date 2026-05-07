"""
Local copy of ViT modified for AI4SmallFarms.
Uses 64×64 input with positional embeddings learned from scratch.
The interpolation code is commented for documentation.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from pixelspointspolygons.misc import make_logger


class ViT(nn.Module):
    def __init__(self, cfg, bottleneck=False, local_rank=0) -> None:
        super().__init__()
        self.cfg = cfg

        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

        if cfg.experiment.encoder.checkpoint_file is None:
            self.logger.warning("No checkpoint file specified, using default timm model initialization.")
        if not os.path.isfile(cfg.experiment.encoder.checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file {cfg.experiment.encoder.checkpoint_file} not found.")

        logging.getLogger('timm').setLevel(logging.WARNING)

        # Build model with the new image size (64×64)
        self.vit = timm.create_model(
            model_name=cfg.experiment.encoder.type,     # e.g., "vit_small_patch8_224.dino"
            img_size=cfg.experiment.encoder.in_size,    # 64
            patch_size=cfg.experiment.encoder.patch_size,
            num_classes=0,
            global_pool='',
            pretrained=False,
        )

        if cfg.experiment.encoder.pretrained:
            state_dict = torch.load(cfg.experiment.encoder.checkpoint_file,
                                    map_location=self.cfg.host.device)

            # ------------------------------------------------------------------
            # OPTION: Interpolation – kept for reference.
            # We now learn positional embeddings from scratch instead.
            # ------------------------------------------------------------------
            # if 'pos_embed' in state_dict:
            #     pos_embed_ckpt = state_dict['pos_embed']           # (1, num_patches+1, dim)
            #     num_patches_ckpt = pos_embed_ckpt.shape[1] - 1
            #     grid_size_ckpt = int(num_patches_ckpt ** 0.5)       # 28 for 224×224
            #
            #     new_grid_size = (cfg.experiment.encoder.in_size //
            #                      cfg.experiment.encoder.patch_size)  # 8 for 64×64
            #
            #     if grid_size_ckpt != new_grid_size:
            #         self.logger.info(
            #             f"Interpolating positional embeddings "
            #             f"from {grid_size_ckpt}×{grid_size_ckpt} to {new_grid_size}×{new_grid_size}"
            #         )
            #
            #         class_token = pos_embed_ckpt[:, 0:1, :]
            #         patch_embed = pos_embed_ckpt[:, 1:, :]
            #         patch_embed = patch_embed.reshape(1, grid_size_ckpt, grid_size_ckpt, -1)
            #         patch_embed = patch_embed.permute(0, 3, 1, 2)
            #
            #         patch_embed = F.interpolate(
            #             patch_embed,
            #             size=(new_grid_size, new_grid_size),
            #             mode='bicubic',
            #             align_corners=False,
            #         )
            #         patch_embed = patch_embed.permute(0, 2, 3, 1)
            #         patch_embed = patch_embed.reshape(1, new_grid_size * new_grid_size, -1)
            #         state_dict['pos_embed'] = torch.cat([class_token, patch_embed], dim=1)

            # --- Learn positional embeddings from scratch ---
            # Remove the pretrained pos_embed so the model keeps its freshly
            # initialised random positional embeddings (for the new grid).
            if 'pos_embed' in state_dict:
                self.logger.info(
                    "Discarding pretrained positional embeddings – "
                    "learning them from scratch on the new image size."
                )
                state_dict.pop('pos_embed')

            self.vit.load_state_dict(state_dict, strict=False)

        if bottleneck:
            self.bottleneck = nn.AdaptiveAvgPool1d(
                cfg.experiment.encoder.out_feature_dim
            )
        else:
            self.bottleneck = nn.Identity()

    def forward(self, x):
        x = self.vit(x)
        x = self.bottleneck(x[:, 1:])    # drop CLS token
        return x