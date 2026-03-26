# utils/metrics.py
# ══════════════════════════════════════════════════════
# Evaluation metrics for BEV 2D Occupancy
# CONTRACT: all functions accept RAW LOGITS
# ══════════════════════════════════════════════════════

import torch
import numpy as np

from config.config import BEV_H, BEV_W, THRESHOLD
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


# ──────────────────────────────────────────────────────
# Metric 1 — Occupancy IoU
# ──────────────────────────────────────────────────────

def occupancy_iou(
    pred:      torch.Tensor,
    gt:        torch.Tensor,
    threshold: float = THRESHOLD
) -> float:
    """
    Occupancy IoU — PRIMARY hackathon metric.
    Accepts raw logits — sigmoid applied internally.

    Returns: float in [0,1] — higher is better
    """
    try:
        pred = pred.squeeze(1) if pred.dim() == 4 else pred
        gt   = gt.squeeze(1)   if gt.dim() == 4   else gt

        pred_probs = torch.sigmoid(pred)
        pred_bin   = (pred_probs >= threshold).float()
        gt_bin     = (gt >= 0.5).float()

        pred_flat = pred_bin.view(-1)
        gt_flat   = gt_bin.view(-1)

        intersection = (pred_flat * gt_flat).sum()
        union        = ((pred_flat + gt_flat) > 0).float().sum()

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
    weight = weight / weight.sum()    # sums to 1 → proper weighted average
    return weight                     # (H, W)


def distance_weighted_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.80   # use your tuned threshold
) -> float:
    try:
        pred = pred.squeeze(1) if pred.dim() == 4 else pred
        gt   = gt.squeeze(1)   if gt.dim()   == 4 else gt
        B, H, W = pred.shape

        pred_probs = torch.sigmoid(pred)
        pred_bin   = (pred_probs >= threshold).float()  # ← binarize first
        gt_bin     = (gt >= 0.5).float()

        weight  = _build_distance_weight_map(H, W, device=pred.device)
        error   = torch.abs(pred_bin - gt_bin)          # ← binary error
        weighted = error * weight.unsqueeze(0)
        dwe = weighted.sum(dim=[1, 2]).mean().item()
        return dwe
    except Exception as e:
        raise BEVException("Failed to compute DWE", e) from e
def compute_metrics(pred, gt, threshold: float = 0.80) -> dict:
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    iou = occupancy_iou(pred, gt, threshold=threshold)
    dwe = distance_weighted_error(pred, gt, threshold=threshold)
    logger.debug(f"Metrics | Occ IoU: {iou:.4f} | DWE: {dwe:.6f}")
    return {'occ_iou': iou, 'dwe': dwe}