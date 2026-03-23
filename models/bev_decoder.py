# models/bev_decoder.py
# ══════════════════════════════════════════════════════
# BEV Decoder + Occupancy Head
# Based on FastOcc paper — Eq 3, 4, 5, 7
#
# CRITICAL DESIGN CONTRACT (read before editing):
#   ALL heads output RAW LOGITS (no sigmoid).
#   sigmoid is applied ONLY inside loss functions.
#   This is numerically stable and avoids double-sigmoid.
#
# ARCHITECTURE:
#   BEVDecoder:    (B, 128, 200, 200) → (B, 64, 200, 200)
#   OccupancyHead: (B,  64, 200, 200) → (B,  1, 200, 200)  ← raw logits
#   bev_aux_head:  (B,  64, 200, 200) → (B,  1, 200, 200)  ← raw logits
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
# Building block
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

    FLOPs (Eq 4): Cin × k² × Cout × H × W
    vs 3D (Eq 3): Cin × k³ × Cout × H × W × Z
    Speedup (Eq 5): sj = k × Zj

    Input:  (B, IMG_CHANNELS=128, 200, 200)
    Output: (B, BEV_CHANNELS=64,  200, 200)

    NOTE: No upsampling here — our BEV grid is already
    at target resolution (200×200). Upsampling then
    resizing back would waste compute with zero gain.
    """

    def __init__(self,
                 in_channels:  int = IMG_CHANNELS,   # 128
                 out_channels: int = BEV_CHANNELS):   # 64
        super().__init__()

        try:
            self.decoder = nn.Sequential(
                # Step down: 128 → 128
                ConvBnReLU(in_channels, 128),
                ConvBnReLU(128, 128),
                # Compress: 128 → 64
                ConvBnReLU(128, out_channels),
                ConvBnReLU(out_channels, out_channels),
            )
            # Output: (B, 64, 200, 200)

            logger.info(
                f"BEVDecoder initialized | "
                f"{in_channels} → {out_channels} channels | "
                f"2D FCN (no upsample, stays at 200×200)"
            )

        except Exception as e:
            raise BEVException("Failed to init BEVDecoder", e) from e

    def forward(self, bev_feat: torch.Tensor):
        """
        Args:  bev_feat: (B, 128, 200, 200)
        Returns: decoded: (B, 64, 200, 200)
        """
        try:
            return self.decoder(bev_feat)
        except Exception as e:
            raise BEVException("BEVDecoder forward failed", e) from e


# ──────────────────────────────────────────────────────
# Occupancy Head
# ──────────────────────────────────────────────────────

class OccupancyHead(nn.Module):
    """
    Final occupancy prediction head.

    INPUT CHANGE FROM OLD VERSION:
        Old: takes (bev_decoded, img_feat_interp) — two inputs
        New: takes (bev_decoded) only — one input

    WHY: Fusing camera-averaged image features into BEV
    is geometrically meaningless. A front-camera pixel at
    (u,v) has no spatial relationship to a BEV cell at (row,col).
    The BEVFormerLite view transformer already embeds geometry-correct
    image features into BEV. Re-injecting raw averaged image
    features adds noise, not information.

    CRITICAL: Output is RAW LOGITS. No sigmoid.
    sigmoid is applied inside loss functions only.

    Input:  (B, BEV_CHANNELS=64, 200, 200)
    Output: (B, 1, 200, 200)  ← raw logits, NOT probabilities
    """

    def __init__(self, bev_channels: int = BEV_CHANNELS):  # 64
        super().__init__()

        try:
            # Main prediction path
            self.predict = nn.Sequential(
                ConvBnReLU(bev_channels, 32),
                ConvBnReLU(32, 32),
                nn.Conv2d(32, 1, kernel_size=1)
                # ← NO nn.Sigmoid() here — outputs raw logits
            )

            # Auxiliary BEV head — also raw logits
            # Used for Lb auxiliary supervision (FastOcc Eq 7)
            self.aux_head = nn.Sequential(
                ConvBnReLU(bev_channels, 32),
                nn.Conv2d(32, 1, kernel_size=1)
                # ← NO nn.Sigmoid() here either
            )

            logger.info(
                f"OccupancyHead initialized | "
                f"input: {bev_channels}ch → output: 1ch logits"
            )

        except Exception as e:
            raise BEVException("Failed to init OccupancyHead", e) from e

    def forward(self, bev_decoded: torch.Tensor):
        """
        Args:
            bev_decoded: (B, 64, 200, 200)

        Returns:
            occ_logits: (B, 1, 200, 200)  ← raw logits
            aux_logits: (B, 1, 200, 200)  ← raw logits (for aux loss)
        """
        try:
            occ_logits = self.predict(bev_decoded)   # (B, 1, 200, 200)
            aux_logits = self.aux_head(bev_decoded)  # (B, 1, 200, 200)

            return occ_logits, aux_logits

        except Exception as e:
            raise BEVException("OccupancyHead forward failed", e) from e


# ──────────────────────────────────────────────────────
# Loss Functions (Eq 7 from FastOcc)
# ALL functions accept RAW LOGITS, not probabilities
# ──────────────────────────────────────────────────────

def focal_loss(
    pred_logits: torch.Tensor,  # (B, 1, H, W) — raw logits, NO sigmoid applied yet
    gt:          torch.Tensor,  # (B, 1, H, W) — binary {0.0, 1.0}
    gamma:       float = 2.0,
    alpha:       float = 0.75   # 0.75 = weight for POSITIVE class (occupied = minority ~5%)
) -> torch.Tensor:
    """
    Focal Loss — handles class imbalance.
    alpha=0.75 means occupied cells get 3× more weight than empty cells.
    gamma=2.0 down-weights easy negatives (vast empty BEV space).

    CORRECT alpha semantics:
        alpha_t = 0.75 for gt=1 (occupied — minority, hard to learn)
        alpha_t = 0.25 for gt=0 (empty   — majority, easy, down-weight)
    """
    # Apply sigmoid HERE — once. Not in the head.
    pred = torch.sigmoid(pred_logits)
    pred = pred.clamp(1e-6, 1.0 - 1e-6)  # numerical stability

    # Per-cell alpha: 0.75 for positives, 0.25 for negatives
    alpha_t = alpha * gt + (1.0 - alpha) * (1.0 - gt)

    # Standard cross-entropy then modulate with focal weight
    bce   = -(gt * torch.log(pred) + (1.0 - gt) * torch.log(1.0 - pred))
    pt    = torch.exp(-bce)                             # pt = 1 if easy, 0 if hard
    focal = alpha_t * (1.0 - pt) ** gamma * bce        # hard examples get full weight

    return focal.mean()


def dice_loss(
    pred_logits: torch.Tensor,  # (B, 1, H, W) — raw logits
    gt:          torch.Tensor   # (B, 1, H, W) — binary {0.0, 1.0}
) -> torch.Tensor:
    """
    Soft Dice Loss — directly optimizes overlap (≈ IoU).

    This is what you are actually evaluated on.
    Focal loss handles class imbalance;
    Dice loss handles spatial structure quality.

    Uses smooth=1e-6 (NOT +1) to avoid fake perfect scores
    on all-zero predictions against all-zero GT.
    """
    pred   = torch.sigmoid(pred_logits)
    smooth = 1e-6

    # Sum over all spatial dims per batch item → shape (B,)
    intersection = (pred * gt).sum(dim=[1, 2, 3])
    union        = pred.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3])

    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def aux_bce_loss(
    aux_logits: torch.Tensor,  # (B, 1, H, W) — raw logits from aux_head
    gt:         torch.Tensor   # (B, 1, H, W) — binary GT
) -> torch.Tensor:
    """
    Auxiliary BEV supervision — Lb from FastOcc Eq 7.
    Uses BCEWithLogits which is numerically stable
    (combines sigmoid + BCE in one stable operation).
    """
    return F.binary_cross_entropy_with_logits(aux_logits, gt)


def total_occupancy_loss(
    occ_logits:  torch.Tensor,           # (B, 1, H, W) main head raw logits
    gt:          torch.Tensor,           # (B, 1, H, W) binary ground truth
    aux_logits:  torch.Tensor = None,    # (B, 1, H, W) aux head raw logits
    focal_w:     float = 1.0,
    dice_w:      float = 1.0,
    aux_w:       float = 0.5
) -> dict:
    """
    Combined loss — FastOcc Eq 7:
        Loss = Lf (focal) + Ldice + 0.5 * Lb (aux BCE)

    All inputs are RAW LOGITS.
    sigmoid is applied inside each sub-loss.

    Args:
        occ_logits:  main head output   (B, 1, 200, 200)
        gt:          LiDAR occupancy GT (B, 1, 200, 200)
        aux_logits:  aux head output    (B, 1, 200, 200) or None

    Returns:
        dict: {
            'total':   scalar — backprop this
            'focal':   scalar — for logging
            'dice':    scalar — for logging
            'aux_bce': scalar — for logging
        }
    """
    Lf   = focal_loss(occ_logits, gt) * focal_w
    Ld   = dice_loss(occ_logits, gt)  * dice_w

    Lb   = torch.tensor(0.0, device=occ_logits.device)
    if aux_logits is not None:
        Lb = aux_bce_loss(aux_logits, gt) * aux_w

    total = Lf + Ld + Lb

    return {
        'total'  : total,
        'focal'  : Lf.detach(),
        'dice'   : Ld.detach(),
        'aux_bce': Lb.detach() if isinstance(Lb, torch.Tensor) else Lb
    }