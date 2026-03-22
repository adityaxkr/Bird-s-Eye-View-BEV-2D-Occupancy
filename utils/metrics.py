# utils/metrics.py
# ══════════════════════════════════════════════════════
# Evaluation metrics for BEV 2D Occupancy
# EXACTLY what hackathon judges will measure:
#   1. Occupancy IoU
#   2. Distance-Weighted Error
# ══════════════════════════════════════════════════════

import torch
import numpy as np

from config.config import BEV_H, BEV_W
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


# ──────────────────────────────────────────────────────
# Metric 1 — Occupancy IoU
# ──────────────────────────────────────────────────────

def occupancy_iou(pred:      torch.Tensor,
                  gt:        torch.Tensor,
                  threshold: float = 0.5
                  ) -> float:
    """
    Occupancy IoU — PRIMARY hackathon metric.

    Measures how well predicted BEV occupancy
    matches LiDAR ground truth.

    Formula:
        IoU = |pred ∩ gt| / |pred ∪ gt|

    Args:
        pred:      (B, 1, H, W) or (B, H, W)
                   model output — probabilities [0,1]
        gt:        (B, 1, H, W) or (B, H, W)
                   ground truth — binary {0, 1}
        threshold: probability cutoff for occupied

    Returns:
        iou: float in [0, 1]
             0.0 = no overlap at all
             1.0 = perfect prediction
    """
    try:
        # Ensure same shape
        pred = pred.squeeze(1) if pred.dim() == 4 else pred
        gt   = gt.squeeze(1)   if gt.dim() == 4   else gt
        # Both: (B, H, W)

        # Binarize predictions
        pred_bin = (pred >= threshold).float()
        gt_bin   = (gt   >= threshold).float()

        # Flatten for easier computation
        pred_flat = pred_bin.view(-1)
        gt_flat   = gt_bin.view(-1)

        # Intersection: both pred and gt = 1
        intersection = (pred_flat * gt_flat).sum()

        # Union: either pred or gt = 1
        union = ((pred_flat + gt_flat) > 0).float().sum()

        # Avoid division by zero
        if union == 0:
            return 1.0  # both empty = perfect match

        iou = (intersection / union).item()
        return iou

    except Exception as e:
        raise BEVException(
            "Failed to compute Occupancy IoU", e
        ) from e


# ──────────────────────────────────────────────────────
# Metric 2 — Distance-Weighted Error
# ──────────────────────────────────────────────────────

def _build_distance_weight_map(
        h: int = BEV_H,
        w: int = BEV_W,
        device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Build a weight map where pixels closer to the
    ego vehicle (grid centre) have higher weight.

    weight(r,c) = 1 / distance(r,c) from centre
    Then normalized so weights sum to 1.

    Returns:
        weight_map: (H, W) tensor
    """
    cx = w // 2   # centre col = ego vehicle x
    cy = h // 2   # centre row = ego vehicle y

    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)

    # Grid of distances from centre
    yy, xx    = torch.meshgrid(ys, xs, indexing='ij')
    dist      = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Avoid division by zero at centre pixel
    dist      = dist.clamp(min=1e-6)

    # Closer = higher weight
    weight    = 1.0 / dist

    # Normalize: weights sum to 1
    weight    = weight / weight.sum()

    return weight  # (H, W)


def distance_weighted_error(pred:   torch.Tensor,
                             gt:     torch.Tensor
                             ) -> float:
    """
    Distance-Weighted Error — SECONDARY hackathon metric.

    Errors near the ego vehicle are penalized more.
    This is critical for autonomous driving safety —
    nearby obstacles matter more than distant ones.

    Formula:
        DWE = Σ weight(r,c) × |pred(r,c) - gt(r,c)|

    Where weight(r,c) = 1/dist from grid centre

    Args:
        pred: (B, 1, H, W) or (B, H, W) predictions
        gt:   (B, 1, H, W) or (B, H, W) ground truth

    Returns:
        dwe: float — lower is better
    """
    try:
        # Squeeze channel dim if present
        pred = pred.squeeze(1) if pred.dim() == 4 else pred
        gt   = gt.squeeze(1)   if gt.dim() == 4   else gt
        # Both: (B, H, W)

        B, H, W = pred.shape

        # Build weight map (same for every sample)
        weight = _build_distance_weight_map(
            H, W, device=pred.device
        )
        # (H, W)

        # Per-pixel absolute error
        error = (pred - gt).abs()
        # (B, H, W)

        # Apply distance weights
        # Broadcast weight (H,W) across batch (B,H,W)
        weighted = error * weight.unsqueeze(0)
        # (B, H, W)

        # Average over batch
        dwe = weighted.sum(dim=[1, 2]).mean().item()

        return dwe

    except Exception as e:
        raise BEVException(
            "Failed to compute Distance-Weighted Error", e
        ) from e


# ──────────────────────────────────────────────────────
# Combined metrics function
# ──────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor,
                    gt:   torch.Tensor
                    ) -> dict:
    """
    Compute all hackathon metrics at once.

    Args:
        pred: (B, 1, H, W) model output
        gt:   (B, 1, H, W) or (B, H, W) ground truth

    Returns:
        dict:
            occ_iou : float  ↑ higher is better
            dwe     : float  ↓ lower is better
    """
    # Ensure gt has channel dim
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    iou = occupancy_iou(pred, gt)
    dwe = distance_weighted_error(pred, gt)

    logger.info(
        f"Metrics | "
        f"Occ IoU: {iou:.4f} | "
        f"DWE: {dwe:.6f}"
    )

    return {
        'occ_iou': iou,
        'dwe'    : dwe
    }