"""
Local copy of ViT modified for AI4SmallFarms.
Uses 32×32 input with both patch embedding and positional embeddings learned from scratch.
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

        # Build model with the new image size (32×32)
        self.vit = timm.create_model(
            model_name=cfg.experiment.encoder.type,     # e.g., "vit_small_patch8_224.dino"
            img_size=cfg.experiment.encoder.in_size,    # 32
            patch_size=cfg.experiment.encoder.patch_size,
            num_classes=0,
            global_pool='',
            pretrained=False,
        )

        if cfg.experiment.encoder.pretrained:
            state_dict = torch.load(cfg.experiment.encoder.checkpoint_file,
                                    map_location=self.cfg.host.device)

            # --- Remove pretrained patch_embed (incompatible size) ---
            for key in list(state_dict.keys()):
                if 'patch_embed' in key:
                    del state_dict[key]

            # --- Discard pretrained pos_embed (we learn it from scratch) ---
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