# utils/visualize.py
# ══════════════════════════════════════════════════════
# Visualization for BEV 2D Occupancy results
# Generates plots for PPT, GitHub, and evaluation
#
# CONTRACT:
#   All plotting functions expect PROBABILITIES [0,1]
#   NOT raw logits. Apply torch.sigmoid(logits) before
#   passing to any function in this file.
#   Only metrics.py accepts raw logits.
# ══════════════════════════════════════════════════════

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from config.config import (
    BEV_H, BEV_W,
    X_RANGE, Y_RANGE,
    CAM_NAMES,
    RESULTS_DIR
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


# ──────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Safely convert any tensor → 2D numpy array."""
    if isinstance(t, torch.Tensor):
        return t.squeeze().detach().cpu().numpy()
    return np.squeeze(t)


def _denorm_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for display.
    Input:  (3, H, W) tensor
    Output: (H, W, 3) numpy array in [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_tensor.permute(1, 2, 0).cpu().numpy()
    img  = img * std + mean
    return np.clip(img, 0.0, 1.0)


def _bev_extent():
    """Returns [xmin, xmax, ymin, ymax] for imshow extent."""
    return [X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]]


def _ego_marker(ax, size=10):
    """Plot ego vehicle marker at origin."""
    ax.plot(0, 0, 'b^', markersize=size, label='Ego vehicle',
            zorder=5)


# ──────────────────────────────────────────────────────
# 1. BEV Occupancy comparison plot
# ──────────────────────────────────────────────────────

def plot_bev_comparison(
    pred:      torch.Tensor,   # PROBABILITIES (1,H,W) or (H,W)
    gt:        torch.Tensor,   # binary GT     (1,H,W) or (H,W)
    save_path: str  = None,
    title:     str  = "BEV Occupancy",
    threshold: float = 0.5
) -> plt.Figure:
    """
    3-panel figure: GT | Prediction | Error map.

    Args:
        pred:      PROBABILITIES [0,1] — NOT raw logits.
                   Call torch.sigmoid(logits) before passing.
        gt:        Binary ground truth {0, 1}.
        threshold: Binarization threshold for error map.

    Returns:
        matplotlib Figure
    """
    try:
        pred_np = _to_numpy(pred)   # (H, W)
        gt_np   = _to_numpy(gt)     # (H, W)

        pred_bin = (pred_np >= threshold).astype(np.float32)
        error    = np.abs(pred_bin - gt_np)

        # Error breakdown for title
        tp = ((pred_bin == 1) & (gt_np == 1)).sum()
        fp = ((pred_bin == 1) & (gt_np == 0)).sum()
        fn = ((pred_bin == 0) & (gt_np == 1)).sum()
        iou_val = tp / (tp + fp + fn + 1e-6)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f'{title}  |  IoU: {iou_val:.4f}',
            fontsize=14, fontweight='bold'
        )
        extent = _bev_extent()

        # ── Panel 1: Ground Truth ──────────────────────
        axes[0].imshow(gt_np, cmap='hot', origin='lower',
                       extent=extent, vmin=0, vmax=1)
        axes[0].set_title('Ground Truth\n(LiDAR projection)',
                          fontsize=11)
        axes[0].set_xlabel('X — forward (m)')
        axes[0].set_ylabel('Y — left (m)')
        _ego_marker(axes[0])
        axes[0].legend(fontsize=8)

        # ── Panel 2: Prediction ────────────────────────
        im = axes[1].imshow(pred_np, cmap='hot', origin='lower',
                            extent=extent, vmin=0, vmax=1)
        axes[1].set_title('Predicted BEV Occupancy\n(our model)',
                          fontsize=11)
        axes[1].set_xlabel('X — forward (m)')
        _ego_marker(axes[1])
        axes[1].legend(fontsize=8)
        plt.colorbar(im, ax=axes[1], label='P(occupied)',
                     fraction=0.046)

        # ── Panel 3: Error map ─────────────────────────
        # Green = correct, Red = wrong
        axes[2].imshow(error, cmap='RdYlGn_r', origin='lower',
                       extent=extent, vmin=0, vmax=1)
        axes[2].set_title(
            f'Error Map\n'
            f'TP={tp}  FP={fp}  FN={fn}',
            fontsize=11
        )
        axes[2].set_xlabel('X — forward (m)')
        _ego_marker(axes[2])
        axes[2].legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            os.makedirs(
                os.path.dirname(save_path) or '.', exist_ok=True
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"BEV comparison saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException("Failed to plot BEV comparison", e) from e


# ──────────────────────────────────────────────────────
# 2. Camera images grid
# ──────────────────────────────────────────────────────

def plot_cameras(
    imgs:      torch.Tensor,   # (6, 3, H, W) normalized
    save_path: str = None
) -> plt.Figure:
    """
    6-panel grid of all camera inputs.

    Args:
        imgs: (6, 3, H, W) — ImageNet normalized tensors
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('nuScenes Camera Inputs (6 cameras)',
                     fontsize=14, fontweight='bold')
        axes = axes.flatten()

        for i, cam_name in enumerate(CAM_NAMES):
            img = _denorm_image(imgs[i])
            axes[i].imshow(img)
            axes[i].set_title(
                cam_name.replace('CAM_', ''),
                fontsize=10
            )
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Camera grid saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException("Failed to plot cameras", e) from e


# ──────────────────────────────────────────────────────
# 3. Full results summary (for PPT / README)
# ──────────────────────────────────────────────────────

def plot_full_results(
    imgs:      torch.Tensor,   # (6, 3, H, W)
    pred:      torch.Tensor,   # PROBABILITIES (1,H,W) — NOT logits
    gt:        torch.Tensor,   # binary GT (1,H,W) or (H,W)
    metrics:   dict,           # {'occ_iou': float, 'dwe': float}
    save_path: str  = None,
    sample_id: int  = 0
) -> plt.Figure:
    """
    Full 2×3 results figure for PPT and GitHub README.

    Row 1: Front | Front-Left | Front-Right camera images
    Row 2: GT BEV | Predicted BEV | Error map + metrics

    Args:
        pred: PROBABILITIES [0,1] — call torch.sigmoid first.
        gt:   Binary ground truth {0, 1}.
    """
    try:
        iou = metrics.get('occ_iou', 0.0)
        dwe = metrics.get('dwe', 0.0)

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f'BEV 2D Occupancy — Sample {sample_id}  |  '
            f'IoU: {iou:.4f}  |  DWE: {dwe:.6f}',
            fontsize=13, fontweight='bold'
        )

        extent = _bev_extent()

        # ── Row 1: front-facing cameras ───────────────
        # CAM_NAMES[0]=FRONT, [1]=FRONT_LEFT, [2]=FRONT_RIGHT
        for plot_i, cam_i in enumerate([0, 1, 2]):
            ax  = fig.add_subplot(2, 3, plot_i + 1)
            img = _denorm_image(imgs[cam_i])
            ax.imshow(img)
            ax.set_title(
                CAM_NAMES[cam_i].replace('CAM_', ''),
                fontsize=10
            )
            ax.axis('off')

        # ── Row 2: BEV panels ─────────────────────────
        pred_np = _to_numpy(pred)   # (H, W) probabilities
        gt_np   = _to_numpy(gt)     # (H, W) binary
        error   = np.abs((pred_np >= 0.5).astype(float) - gt_np)

        # GT BEV
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(gt_np, cmap='hot', origin='lower', extent=extent)
        ax4.set_title('Ground Truth (LiDAR)', fontsize=10)
        ax4.set_xlabel('X — forward (m)')
        ax4.set_ylabel('Y — left (m)')
        _ego_marker(ax4, size=8)
        ax4.legend(fontsize=8)

        # Predicted BEV
        ax5 = fig.add_subplot(2, 3, 5)
        im5 = ax5.imshow(pred_np, cmap='hot', origin='lower',
                         extent=extent, vmin=0, vmax=1)
        ax5.set_title('Predicted BEV Occupancy', fontsize=10)
        ax5.set_xlabel('X — forward (m)')
        _ego_marker(ax5, size=8)
        ax5.legend(fontsize=8)
        plt.colorbar(im5, ax=ax5, fraction=0.046, label='P(occ)')

        # Error map
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(error, cmap='RdYlGn_r', origin='lower',
                   extent=extent, vmin=0, vmax=1)
        ax6.set_title(
            f'Error Map\n'
            f'IoU={iou:.3f}  |  DWE={dwe:.5f}',
            fontsize=10
        )
        ax6.set_xlabel('X — forward (m)')
        _ego_marker(ax6, size=8)
        ax6.legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            os.makedirs(
                os.path.dirname(save_path) or '.', exist_ok=True
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Full results saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException("Failed to plot full results", e) from e


# ──────────────────────────────────────────────────────
# 4. Training curves
# ──────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: list,
    val_ious:     list,
    val_dwes:     list = None,   # optional — computed at eval time
    save_path:    str  = None
) -> plt.Figure:
    """
    Training loss + validation IoU (+ optional DWE) over epochs.

    Args:
        train_losses: loss per epoch (from train.py history)
        val_ious:     IoU per epoch  (from train.py history)
        val_dwes:     DWE per epoch  (optional — from eval.py)
                      If None, only 2 panels are shown.
    """
    try:
        epochs  = range(1, len(train_losses) + 1)
        n_plots = 3 if val_dwes else 2

        fig, axes = plt.subplots(1, n_plots,
                                 figsize=(5 * n_plots, 4))
        if n_plots == 2:
            axes = list(axes)   # make indexable

        fig.suptitle('Training Progress',
                     fontsize=13, fontweight='bold')

        # ── Loss ──────────────────────────────────────
        axes[0].plot(epochs, train_losses,
                     'b-o', linewidth=2, markersize=5,
                     label='train loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Mark best (lowest) loss epoch
        best_epoch = int(np.argmin(train_losses)) + 1
        best_loss  = min(train_losses)
        axes[0].axvline(best_epoch, color='b',
                        linestyle='--', alpha=0.5,
                        label=f'best={best_loss:.3f}')

        # ── IoU ───────────────────────────────────────
        axes[1].plot(epochs, val_ious,
                     'g-o', linewidth=2, markersize=5,
                     label='val IoU')
        axes[1].set_title('Validation Occupancy IoU ↑')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Mark best IoU epoch
        best_iou_epoch = int(np.argmax(val_ious)) + 1
        best_iou       = max(val_ious)
        axes[1].axvline(best_iou_epoch, color='g',
                        linestyle='--', alpha=0.5,
                        label=f'best={best_iou:.3f}')
        axes[1].legend()

        # ── DWE (optional) ────────────────────────────
        if val_dwes and n_plots == 3:
            axes[2].plot(epochs, val_dwes,
                         'r-o', linewidth=2, markersize=5,
                         label='val DWE')
            axes[2].set_title('Validation DWE ↓')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('DWE (lower = better)')
            axes[2].grid(True, alpha=0.3)
            best_dwe_epoch = int(np.argmin(val_dwes)) + 1
            axes[2].axvline(best_dwe_epoch, color='r',
                            linestyle='--', alpha=0.5)
            axes[2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException("Failed to plot training curves", e) from e


# ──────────────────────────────────────────────────────
# 5. Before vs After Training (for PPT impact slide)
# ──────────────────────────────────────────────────────

def plot_before_after_training(
    imgs:           torch.Tensor,   # (6, 3, H, W)
    pred_before:    torch.Tensor,   # PROBABILITIES before training
    pred_after:     torch.Tensor,   # PROBABILITIES after training
    gt:             torch.Tensor,   # binary GT
    metrics_before: dict,           # {'occ_iou': float, 'dwe': float}
    metrics_after:  dict,
    save_path:      str = None,
    sample_id:      int = 0
) -> plt.Figure:
    """
    Side-by-side Before vs After training comparison.
    The most visually impactful slide for your presentation.

    Layout:
        Row 1: Front | Front-Left | Front-Right (camera inputs)
        Row 2: GT BEV | Before pred | After pred

    Args:
        pred_before: PROBABILITIES [0,1] from untrained model
        pred_after:  PROBABILITIES [0,1] from trained model
        Both: call torch.sigmoid(logits) before passing.
    """
    try:
        iou_before = metrics_before.get('occ_iou', 0.0)
        iou_after  = metrics_after.get('occ_iou',  0.0)
        improvement = iou_after - iou_before

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f'Before vs After Training — Sample {sample_id}\n'
            f'IoU Before: {iou_before:.4f}  →  '
            f'IoU After: {iou_after:.4f}  '
            f'(+{improvement:.4f})',
            fontsize=13, fontweight='bold',
            color='darkgreen' if improvement > 0 else 'red'
        )

        extent = _bev_extent()

        # ── Row 1: front cameras ──────────────────────
        for i, (cam_i, label) in enumerate(zip(
            [0, 1, 2],
            ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT']
        )):
            ax  = fig.add_subplot(2, 3, i + 1)
            img = _denorm_image(imgs[cam_i])
            ax.imshow(img)
            ax.set_title(label, fontsize=10)
            ax.axis('off')

        # ── Row 2: GT | Before | After ────────────────
        gt_np     = _to_numpy(gt)           # (H, W) binary
        before_np = _to_numpy(pred_before)  # (H, W) probabilities
        after_np  = _to_numpy(pred_after)   # (H, W) probabilities

        # GT
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(gt_np, cmap='hot',
                   origin='lower', extent=extent)
        ax4.set_title('Ground Truth (LiDAR)',
                      fontsize=11, fontweight='bold')
        ax4.set_xlabel('X — forward (m)')
        ax4.set_ylabel('Y — left (m)')
        _ego_marker(ax4)
        ax4.legend(fontsize=8)

        # Before training
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(before_np, cmap='hot',
                   origin='lower', extent=extent,
                   vmin=0, vmax=1)
        ax5.set_title(
            f'Before Training\nIoU = {iou_before:.4f}',
            fontsize=11, color='red'
        )
        ax5.set_xlabel('X — forward (m)')
        _ego_marker(ax5)
        ax5.legend(fontsize=8)

        # After training
        ax6 = fig.add_subplot(2, 3, 6)
        im6 = ax6.imshow(after_np, cmap='hot',
                         origin='lower', extent=extent,
                         vmin=0, vmax=1)
        ax6.set_title(
            f'After Training\nIoU = {iou_after:.4f}',
            fontsize=11, color='green'
        )
        ax6.set_xlabel('X — forward (m)')
        _ego_marker(ax6)
        ax6.legend(fontsize=8)
        plt.colorbar(im6, ax=ax6, fraction=0.046,
                     label='P(occupied)')

        plt.tight_layout()

        if save_path:
            os.makedirs(
                os.path.dirname(save_path) or '.', exist_ok=True
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Before/After saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException("Failed to plot before/after", e) from e


# ──────────────────────────────────────────────────────
# 6. Coverage map (for presentation — shows geometry)
# ──────────────────────────────────────────────────────

def plot_coverage_map(
    coverage_map: torch.Tensor,   # (BEV_H, BEV_W) — camera count per cell
    save_path:    str = None
) -> plt.Figure:
    """
    Plot how many cameras see each BEV cell.
    Great for explaining the geometry in your presentation.

    Args:
        coverage_map: (200, 200) tensor — values 0–6
    """
    try:
        cov_np = _to_numpy(coverage_map)
        extent = _bev_extent()

        fig, ax = plt.subplots(figsize=(7, 7))

        im = ax.imshow(cov_np, cmap='hot', origin='lower',
                       extent=extent, vmin=0, vmax=6)
        plt.colorbar(im, ax=ax, label='# cameras visible (max=6)')

        ax.set_title('BEV Camera Coverage Map\n'
                     '(how many cameras see each ground cell)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('X — forward (m)')
        ax.set_ylabel('Y — left (m)')
        _ego_marker(ax)
        ax.legend(fontsize=9)

        # Annotate coverage stats
        n_zero = (cov_np == 0).sum()
        n_total = cov_np.size
        ax.text(
            0.02, 0.02,
            f'Blind cells: {n_zero}/{n_total} '
            f'({100*n_zero/n_total:.1f}%)',
            transform=ax.transAxes,
            fontsize=9, color='cyan',
            bbox=dict(boxstyle='round', facecolor='black',
                      alpha=0.5)
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Coverage map saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException("Failed to plot coverage map", e) from e