# utils/metrics.py
# ══════════════════════════════════════════════════════
# Evaluation metrics for BEV 2D Occupancy
# EXACTLY what hackathon judges will measure:
#   1. Occupancy IoU
#   2. Distance-Weighted Error
#
# CONTRACT: all functions accept RAW LOGITS from model.
# sigmoid is applied internally. Do NOT pre-apply sigmoid
# before calling these functions.
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

def occupancy_iou(
    pred:      torch.Tensor,   # raw logits (B,1,H,W) or (B,H,W)
    gt:        torch.Tensor,   # binary {0,1} (B,1,H,W) or (B,H,W)
    threshold: float = 0.5     # applied AFTER sigmoid
) -> float:
    """
    Occupancy IoU — PRIMARY hackathon metric.
    Accepts raw logits — sigmoid applied internally.

    Formula: IoU = |pred ∩ gt| / |pred ∪ gt|

    Returns:
        iou: float in [0, 1]  — higher is better
             Returns 0.0 if both pred and GT are empty
             (not 1.0 — empty prediction is not perfect)
    """
    try:
        pred = pred.squeeze(1) if pred.dim() == 4 else pred
        gt   = gt.squeeze(1)   if gt.dim() == 4   else gt
        # Both: (B, H, W)

        # ── FIX: apply sigmoid before threshold ───────
        pred_probs = torch.sigmoid(pred)
        pred_bin   = (pred_probs >= threshold).float()
        gt_bin     = (gt >= 0.5).float()

        pred_flat = pred_bin.view(-1)
        gt_flat   = gt_bin.view(-1)

        intersection = (pred_flat * gt_flat).sum()
        union        = ((pred_flat + gt_flat) > 0).float().sum()

        # ── FIX: return 0.0 not 1.0 for empty ─────────
        if union == 0:
            return 0.0

        return (intersection / union).item()

    except Exception as e:
        raise BEVException("Failed to compute Occupancy IoU", e) from e


# ──────────────────────────────────────────────────────
# Metric 2 — Distance-Weighted Error
# ──────────────────────────────────────────────────────

def _build_distance_weight_map(
    h: int = BEV_H,
    w: int = BEV_W,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Weight map: closer to ego = higher weight.
    Normalized so all weights sum to 1.
    Returns: (H, W)
    """
    cx = w // 2
    cy = h // 2

    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)

    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    dist   = torch.sqrt((xx - cx)**2 + (yy - cy)**2).clamp(min=1e-6)

    weight = 1.0 / dist
    weight = weight / weight.sum()
    return weight  # (H, W)


def distance_weighted_error(
    pred: torch.Tensor,   # raw logits (B,1,H,W) or (B,H,W)
    gt:   torch.Tensor    # binary GT  (B,1,H,W) or (B,H,W)
) -> float:
    """
    Distance-Weighted Error — SECONDARY hackathon metric.
    Accepts raw logits — sigmoid applied internally.

    Formula: DWE = Σ weight(r,c) × |pred_prob(r,c) - gt(r,c)|
    Returns: float — lower is better
    """
    try:
        pred = pred.squeeze(1) if pred.dim() == 4 else pred
        gt   = gt.squeeze(1)   if gt.dim() == 4   else gt
        # Both: (B, H, W)

        B, H, W = pred.shape

        # ── FIX: convert logits to probabilities ───────
        pred_probs = torch.sigmoid(pred)   # (B, H, W) in [0,1]

        weight   = _build_distance_weight_map(H, W, device=pred.device)
        error    = (pred_probs - gt).abs()           # (B, H, W)
        weighted = error * weight.unsqueeze(0)       # (B, H, W)
        dwe      = weighted.sum(dim=[1, 2]).mean().item()

        return dwe

    except Exception as e:
        raise BEVException("Failed to compute DWE", e) from e


# ──────────────────────────────────────────────────────
# Combined metrics
# ──────────────────────────────────────────────────────

def compute_metrics(
    pred: torch.Tensor,   # raw logits (B,1,H,W)
    gt:   torch.Tensor    # binary GT  (B,1,H,W) or (B,H,W)
) -> dict:
    """
    Compute all hackathon metrics.
    Accepts raw logits — sigmoid applied internally.
    """
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    iou = occupancy_iou(pred, gt)
    dwe = distance_weighted_error(pred, gt)

    logger.info(f"Metrics | Occ IoU: {iou:.4f} | DWE: {dwe:.6f}")

    return {'occ_iou': iou, 'dwe': dwe}