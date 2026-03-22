# models/bev_decoder.py
# ══════════════════════════════════════════════════════
# BEV Decoder + Occupancy Head
# Based on FastOcc paper — Eq 3, 4, 5, 7
#
# Key idea: replace 3D conv with 2D FCN
# Speedup: k × Z times faster (Eq 5)
# For k=3, Z=8 → 24× faster than 3D conv
#
# HACKATHON OUTPUT:
#   2D grid where each pixel = P(occupied)
#   pixel size = 0.4m × 0.4m real world
# ══════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import (
    IMG_CHANNELS, BEV_CHANNELS,
    BEV_H, BEV_W
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


# ──────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────

class ConvBnReLU(nn.Module):
    """Standard Conv → BN → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int,
                 k: int = 3, stride: int = 1,
                 padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ──────────────────────────────────────────────────────
# FastOcc 2D FCN Decoder (Eq 3, 4, 5)
# ──────────────────────────────────────────────────────

class BEVDecoder(nn.Module):
    """
    FastOcc 2D FCN Decoder.

    Implements the core FastOcc idea:
    Instead of 3D convolutions over (H,W,Z),
    we collapse Z into the channel dimension
    and decode entirely in 2D.

    Speedup (Eq 5): sj = k × Zj
    For k=3, Z=8 → 24× faster than 3D conv!

    Input:  bev_feat (B, C, BEV_H, BEV_W)
    Output: decoded  (B, BEV_CHANNELS, BEV_H*2, BEV_W*2)
    """

    def __init__(self,
                 in_channels:  int = IMG_CHANNELS,
                 out_channels: int = BEV_CHANNELS):
        super().__init__()

        try:
            # ── 2D FCN decoder ────────────────────────────
            # FLOPs = C_in × k² × C_out × H × W  (Eq 4)
            # Much cheaper than 3D version (Eq 3)
            self.decoder = nn.Sequential(
                # Block 1: reduce channels
                ConvBnReLU(in_channels, 256),
                ConvBnReLU(256, 128),

                # Upsample ×2 (restore spatial resolution)
                nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False
                ),

                # Block 2: refine features
                ConvBnReLU(128, out_channels),
                ConvBnReLU(out_channels, out_channels),
            )

            # ── BEV auxiliary head ────────────────────────
            # Eq 7: Lb = BEV binary cross-entropy loss
            # Provides extra supervision signal during train
            self.bev_aux_head = nn.Sequential(
                ConvBnReLU(out_channels, 32),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )

            logger.info(
                f"BEVDecoder initialized | "
                f"in: {in_channels} → out: {out_channels} | "
                f"upsamples 2× using 2D FCN (FastOcc)"
            )

        except Exception as e:
            raise BEVException(
                "Failed to init BEVDecoder", e
            ) from e

    def forward(self, bev_feat: torch.Tensor):
        """
        Args:
            bev_feat: (B, C, BEV_H, BEV_W)

        Returns:
            decoded:  (B, BEV_CHANNELS, BEV_H*2, BEV_W*2)
            bev_aux:  (B, 1, BEV_H*2, BEV_W*2)
                      auxiliary BEV prediction for Lb loss
        """
        try:
            decoded = self.decoder(bev_feat)
            bev_aux = self.bev_aux_head(decoded)

            return decoded, bev_aux

        except Exception as e:
            raise BEVException(
                "BEVDecoder forward failed", e
            ) from e


# ──────────────────────────────────────────────────────
# Occupancy Head (Final prediction)
# ──────────────────────────────────────────────────────

class OccupancyHead(nn.Module):
    """
    Final occupancy prediction head.

    Implements FastOcc's feature integration:
        1. Take decoded BEV features
        2. Add interpolated image features
           (compensates for missing Z-axis info)
        3. Predict final occupancy probability

    Input:
        bev_decoded:    (B, BEV_CHANNELS, H, W)
        img_feat_interp:(B, IMG_CHANNELS, H, W)

    Output:
        occupancy: (B, 1, BEV_H, BEV_W)
                   sigmoid output = P(occupied)
                   this is the final 2D BEV grid!
    """

    def __init__(self,
                 bev_channels: int = BEV_CHANNELS,
                 img_channels: int = IMG_CHANNELS):
        super().__init__()

        try:
            # ── Feature fusion ────────────────────────────
            # Fuse 2D BEV + interpolated 3D image features
            # This restores Z-axis information lost in 2D
            self.fuse = nn.Sequential(
                ConvBnReLU(
                    bev_channels + img_channels, 128
                ),
                ConvBnReLU(128, 64)
            )

            # ── Final occupancy prediction ────────────────
            # Output: probability map P(occupied)
            # sigmoid → values in [0, 1]
            # This is the 2D BEV grid the hackathon asks for
            self.predict = nn.Sequential(
                ConvBnReLU(64, 32),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )

            logger.info(
                f"OccupancyHead initialized | "
                f"fuses BEV({bev_channels}ch) + "
                f"img({img_channels}ch) → 1 channel"
            )

        except Exception as e:
            raise BEVException(
                "Failed to init OccupancyHead", e
            ) from e

    def forward(self,
                bev_decoded:     torch.Tensor,
                img_feat_interp: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            bev_decoded:     (B, BEV_CHANNELS, H, W)
            img_feat_interp: (B, IMG_CHANNELS, H, W)

        Returns:
            occupancy: (B, 1, BEV_H, BEV_W)
        """
        try:
            # Resize image features to match BEV size
            img_feat_interp = F.interpolate(
                img_feat_interp,
                size          = bev_decoded.shape[-2:],
                mode          = 'bilinear',
                align_corners = False
            )

            # Concatenate BEV + image features
            fused = torch.cat(
                [bev_decoded, img_feat_interp], dim=1
            )

            # Fuse and predict
            fused = self.fuse(fused)
            occ   = self.predict(fused)

            # Resize to fixed BEV output size
            # Ensures output always matches ground truth
            occ = F.interpolate(
                occ,
                size          = (BEV_H, BEV_W),
                mode          = 'bilinear',
                align_corners = False
            )

            logger.info(
                f"OccupancyHead forward | "
                f"output: {tuple(occ.shape)} | "
                f"range: [{occ.min():.3f}, {occ.max():.3f}]"
            )

            return occ  # (B, 1, BEV_H, BEV_W)

        except Exception as e:
            raise BEVException(
                "OccupancyHead forward failed", e
            ) from e


# ──────────────────────────────────────────────────────
# Loss functions (Eq 7 from FastOcc)
# ──────────────────────────────────────────────────────
def _build_distance_weight_map(h, w, device):
    """Near pixels = higher weight during training."""
    cx = w // 2
    cy = h // 2
    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    dist   = torch.sqrt((xx-cx)**2 + (yy-cy)**2).clamp(1e-6)
    weight = 1.0 / dist
    weight = weight / weight.mean()  # normalize to mean=1
    return weight  # (H, W)


def focal_loss(pred, gt, gamma=2.0, alpha=0.25):
    """Distance-weighted focal loss."""
    pred   = pred.clamp(1e-6, 1-1e-6)
    pos    = -alpha * (1-pred)**gamma * gt * torch.log(pred)
    neg    = -(1-alpha) * pred**gamma * (1-gt) * torch.log(1-pred)
    loss   = pos + neg  # (B,1,H,W)

    B, C, H, W = loss.shape
    weight = _build_distance_weight_map(H, W, pred.device)
    weight = weight.unsqueeze(0).unsqueeze(0)
    loss   = loss * weight

    return loss.mean()


def lovasz_loss(pred, gt):
    """Directly optimizes IoU."""
    pred = pred.view(-1)
    gt   = gt.view(-1)
    tp   = (pred * gt).sum()
    fp   = (pred * (1-gt)).sum()
    fn   = ((1-pred) * gt).sum()
    return 1.0 - tp / (tp + fp + fn + 1e-6)



def bev_loss(pred, gt):
    return F.binary_cross_entropy(pred, gt)


def total_occupancy_loss(pred, gt, bev_aux=None, bev_gt=None):
    Lf   = focal_loss(pred, gt)
    Lls  = lovasz_loss(pred, gt)
    Lb   = bev_loss(pred, gt)

    Lb_aux = torch.tensor(0.0, device=pred.device)
    if bev_aux is not None and bev_gt is not None:
        bev_gt_r = F.interpolate(
            bev_gt, size=bev_aux.shape[-2:],
            mode='bilinear', align_corners=False
        )
        Lb_aux = bev_loss(bev_aux, bev_gt_r)

    total = Lf + Lls + Lb + 0.5 * Lb_aux

    return {
        'total'  : total,
        'focal'  : Lf,
        'lovasz' : Lls,
        'bce'    : Lb,
        'bev_aux': Lb_aux
    }