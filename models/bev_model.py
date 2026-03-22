# models/bev_model.py
# ══════════════════════════════════════════════════════
# Full BEV 2D Occupancy Model
# Assembles: Backbone → LSS → Decoder → Head
#
# HACKATHON: Bird's-Eye-View 2D Occupancy
# Input:  6 camera images + camera params
# Output: 2D BEV grid (200×200)
#         each pixel = P(occupied)
# ══════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import (
    IMG_CHANNELS, BEV_CHANNELS,
    BEV_H, BEV_W
)
from models.backbone        import ImageBackbone
from models.lss_transformer import LSSViewTransformer
from models.bev_decoder     import (
    BEVDecoder,
    OccupancyHead,
    total_occupancy_loss
)
from logger.custom_logger   import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class BEVOccupancyModel(nn.Module):
    """
    End-to-end BEV 2D Occupancy Model.

    Architecture based on:
      - LSS view transformation (Paper 1, Eq 6)
      - FastOcc 2D FCN decoder  (Paper 2, Eq 3-5)
      - Combined loss function  (Paper 2, Eq 7)

    Input:
        imgs:       (B, 6, 3, H, W)    6 camera images
        intrinsics: (B, 6, 3, 3)       camera K matrices
        extrinsics: (B, 6, 4, 4)       camera E matrices

    Output:
        occupancy:  (B, 1, BEV_H, BEV_W)
                    probability map — P(occupied)
    """

    def __init__(self,
                 img_channels: int  = IMG_CHANNELS,
                 bev_channels: int  = BEV_CHANNELS,
                 bev_h:        int  = BEV_H,
                 bev_w:        int  = BEV_W,
                 pretrained:   bool = True):
        super().__init__()

        try:
            # ── Module 1: Image feature extractor ────────
            # Shared ResNet50 + FPN across all 6 cameras
            self.backbone = ImageBackbone(
                out_channels = img_channels,
                pretrained   = pretrained
            )

            # ── Module 2: View transformation ────────────
            # LSS: 2D features → 3D → BEV
            # Implements Eq 6 with real camera geometry
            self.view_transformer = LSSViewTransformer(
                in_channels = img_channels,
                bev_h       = bev_h,
                bev_w       = bev_w
            )

            # ── Module 3: BEV decoder (FastOcc) ──────────
            # 2D FCN instead of 3D conv
            # k×Z times faster (Eq 5)
            self.bev_decoder = BEVDecoder(
                in_channels  = img_channels,
                out_channels = bev_channels
            )

            # ── Module 4: Occupancy head ──────────────────
            # Fuses BEV features + image interpolation
            # Outputs final 2D probability map
            self.occ_head = OccupancyHead(
                bev_channels = bev_channels,
                img_channels = img_channels
            )

            # ── Image feature pooling ─────────────────────
            # For z-axis compensation (FastOcc trick)
            # Pool image features to BEV size
            self.img_pool = nn.AdaptiveAvgPool2d(
                (bev_h, bev_w)
            )

            self.bev_h = bev_h
            self.bev_w = bev_w

            # Count total parameters
            total = sum(
                p.numel() for p in self.parameters()
            )
            trainable = sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad
            )

            logger.info(
                f"BEVOccupancyModel ready | "
                f"total params: {total:,} | "
                f"trainable: {trainable:,}"
            )

        except Exception as e:
            raise BEVException(
                "Failed to init BEVOccupancyModel", e
            ) from e

    def forward(self,
                imgs:       torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor
                ) -> dict:
        """
        Full forward pass.

        Args:
            imgs:       (B, 6, 3, H, W)
            intrinsics: (B, 6, 3, 3)
            extrinsics: (B, 6, 4, 4)

        Returns:
            dict with:
                occ:     (B, 1, BEV_H, BEV_W) final output
                bev_aux: (B, 1, H, W)          auxiliary output
        """
        try:
            B, N, C, H, W = imgs.shape

            # ── Step 1: Extract image features ───────────
            # Process all 6 cameras with shared backbone
            imgs_flat = imgs.view(B * N, C, H, W)
            # (B*6, 3, H, W)

            img_feats = self.backbone(imgs_flat)
            # (B*6, 128, H/8, W/8)

            _, Cf, fH, fW = img_feats.shape
            img_feats_3d  = img_feats.view(B, N, Cf, fH, fW)
            # (B, 6, 128, fH, fW)

            # ── Step 2: View transformation → BEV ────────
            # LSS: lifts 2D features to 3D using depth
            # then splatted onto BEV plane using K and E
            bev_feat = self.view_transformer(
                img_feats_3d,
                intrinsics,
                extrinsics
            )
            # (B, 128, BEV_H, BEV_W)

            # ── Step 3: 2D FCN decode (FastOcc) ──────────
            # Much faster than 3D convolution
            bev_decoded, bev_aux = self.bev_decoder(bev_feat)
            # decoded: (B, 64, BEV_H*2, BEV_W*2)
            # bev_aux: (B, 1,  BEV_H*2, BEV_W*2)

            # ── Step 4: Image feature interpolation ───────
            # Average 6 cameras → pool to BEV size
            # Compensates for missing Z-axis in 2D BEV
            img_avg  = img_feats.view(
                B, N, Cf, fH, fW
            ).mean(dim=1)
            # (B, 128, fH, fW)

            img_interp = self.img_pool(img_avg)
            # (B, 128, BEV_H, BEV_W)

            # ── Step 5: Final occupancy prediction ────────
            # Fuses BEV decoded features + image features
            # Output: 2D probability map
            occ = self.occ_head(bev_decoded, img_interp)
            # (B, 1, BEV_H, BEV_W)

            return {
                'occ'    : occ,      # main prediction
                'bev_aux': bev_aux   # auxiliary prediction
            }

        except Exception as e:
            raise BEVException(
                "BEVOccupancyModel forward failed", e
            ) from e

    def compute_loss(self,
                     outputs: dict,
                     occ_gt:  torch.Tensor
                     ) -> dict:
        """
        Compute combined loss from Eq 7 (FastOcc paper).

        Args:
            outputs: dict from forward()
            occ_gt:  (B, BEV_H, BEV_W) ground truth

        Returns:
            dict with all loss components
        """
        try:
            # Add channel dim to gt
            gt = occ_gt.unsqueeze(1)
            # (B, 1, BEV_H, BEV_W)

            losses = total_occupancy_loss(
                pred    = outputs['occ'],
                gt      = gt,
                bev_aux = outputs['bev_aux'],
                bev_gt  = gt
            )

            return losses

        except Exception as e:
            raise BEVException(
                "Loss computation failed", e
            ) from e