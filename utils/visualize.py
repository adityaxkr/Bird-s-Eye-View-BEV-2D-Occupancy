# utils/visualize.py
# ══════════════════════════════════════════════════════
# Visualization for BEV 2D Occupancy results
# Generates plots for PPT, GitHub, and evaluation
# ══════════════════════════════════════════════════════

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# 1. BEV Occupancy comparison plot
# ──────────────────────────────────────────────────────

def plot_bev_comparison(pred:      torch.Tensor,
                        gt:        torch.Tensor,
                        save_path: str = None,
                        title:     str = "BEV Occupancy"
                        ) -> plt.Figure:
    """
    Plot predicted vs ground truth BEV occupancy.

    Creates a 3-panel figure:
        Panel 1: Ground Truth (LiDAR)
        Panel 2: Predicted BEV
        Panel 3: Error map

    Args:
        pred:      (1, H, W) or (H, W) prediction
        gt:        (1, H, W) or (H, W) ground truth
        save_path: where to save — None = just show
        title:     figure title

    Returns:
        matplotlib Figure object
    """
    try:
        # Convert to numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.squeeze().detach().cpu().numpy()

        # Binarize prediction at 0.5
        pred_bin = (pred >= 0.5).astype(np.float32)
        error    = np.abs(pred_bin - gt)

        # ── Create figure ─────────────────────────────
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5)
        )
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Common plot settings
        extent = [
            X_RANGE[0], X_RANGE[1],
            Y_RANGE[0], Y_RANGE[1]
        ]

        # ── Panel 1: Ground Truth ──────────────────────
        axes[0].imshow(
            gt,
            cmap='hot',
            origin='lower',
            extent=extent,
            vmin=0, vmax=1
        )
        axes[0].set_title(
            'Ground Truth\n(LiDAR projection)',
            fontsize=11
        )
        axes[0].set_xlabel('X (metres)')
        axes[0].set_ylabel('Y (metres)')
        axes[0].plot(0, 0, 'b^', markersize=10,
                     label='Ego vehicle')
        axes[0].legend(fontsize=8)

        # ── Panel 2: Prediction ────────────────────────
        im = axes[1].imshow(
            pred,
            cmap='hot',
            origin='lower',
            extent=extent,
            vmin=0, vmax=1
        )
        axes[1].set_title(
            'Predicted BEV Occupancy\n(our model)',
            fontsize=11
        )
        axes[1].set_xlabel('X (metres)')
        axes[1].plot(0, 0, 'b^', markersize=10,
                     label='Ego vehicle')
        axes[1].legend(fontsize=8)
        plt.colorbar(im, ax=axes[1],
                     label='P(occupied)',
                     fraction=0.046)

        # ── Panel 3: Error map ─────────────────────────
        axes[2].imshow(
            error,
            cmap='RdYlGn_r',
            origin='lower',
            extent=extent,
            vmin=0, vmax=1
        )
        axes[2].set_title(
            'Error Map\n(red=wrong, green=correct)',
            fontsize=11
        )
        axes[2].set_xlabel('X (metres)')
        axes[2].plot(0, 0, 'b^', markersize=10,
                     label='Ego vehicle')
        axes[2].legend(fontsize=8)

        plt.tight_layout()

        # ── Save ──────────────────────────────────────
        if save_path:
            os.makedirs(
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.',
                exist_ok=True
            )
            plt.savefig(save_path, dpi=150,
                        bbox_inches='tight')
            logger.info(f"BEV comparison saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException(
            "Failed to plot BEV comparison", e
        ) from e


# ──────────────────────────────────────────────────────
# 2. Camera images grid
# ──────────────────────────────────────────────────────

def plot_cameras(imgs:      torch.Tensor,
                 save_path: str = None
                 ) -> plt.Figure:
    """
    Plot all 6 camera images in a grid.
    Shows the input to the model.

    Args:
        imgs:      (6, 3, H, W) camera images
        save_path: where to save

    Returns:
        matplotlib Figure
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            'nuScenes Camera Inputs (6 cameras)',
            fontsize=14, fontweight='bold'
        )
        axes = axes.flatten()

        # ImageNet denormalization
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        for i, cam_name in enumerate(CAM_NAMES):
            img = imgs[i].permute(1, 2, 0).cpu().numpy()

            # Denormalize
            img = (img * std + mean)
            img = np.clip(img, 0, 1)

            axes[i].imshow(img)
            axes[i].set_title(
                cam_name.replace('CAM_', ''),
                fontsize=10
            )
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150,
                        bbox_inches='tight')
            logger.info(f"Camera grid saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException(
            "Failed to plot cameras", e
        ) from e


# ──────────────────────────────────────────────────────
# 3. Full results summary (for PPT)
# ──────────────────────────────────────────────────────

def plot_full_results(imgs:      torch.Tensor,
                      pred:      torch.Tensor,
                      gt:        torch.Tensor,
                      metrics:   dict,
                      save_path: str = None,
                      sample_id: int = 0
                      ) -> plt.Figure:
    """
    Full results figure for PPT and GitHub README.

    Layout:
        Row 1: 3 camera images (front, front-left, front-right)
        Row 2: GT BEV | Predicted BEV | Error + metrics

    Args:
        imgs:      (6, 3, H, W)
        pred:      (1, H, W) or (H, W)
        gt:        (1, H, W) or (H, W)
        metrics:   dict with occ_iou and dwe
        save_path: save location
        sample_id: for title

    Returns:
        matplotlib Figure
    """
    try:
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f'BEV 2D Occupancy — Sample {sample_id}  |  '
            f'IoU: {metrics.get("occ_iou", 0):.4f}  |  '
            f'DWE: {metrics.get("dwe", 0):.6f}',
            fontsize=13, fontweight='bold'
        )

        # ImageNet denorm
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        extent = [
            X_RANGE[0], X_RANGE[1],
            Y_RANGE[0], Y_RANGE[1]
        ]

        # ── Row 1: 3 front cameras ─────────────────────
        front_cams = [0, 1, 2]  # FRONT, FRONT_LEFT, FRONT_RIGHT
        for plot_i, cam_i in enumerate(front_cams):
            ax = fig.add_subplot(2, 3, plot_i + 1)
            img = imgs[cam_i].permute(
                1, 2, 0
            ).cpu().numpy()
            img = np.clip(img * std + mean, 0, 1)
            ax.imshow(img)
            ax.set_title(
                CAM_NAMES[cam_i].replace('CAM_', ''),
                fontsize=10
            )
            ax.axis('off')

        # ── Row 2: BEV results ─────────────────────────
        pred_np = pred.squeeze().detach().cpu().numpy()
        gt_np   = gt.squeeze().detach().cpu().numpy()
        error   = np.abs((pred_np >= 0.5) - gt_np)

        # GT BEV
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(gt_np, cmap='hot',
                   origin='lower', extent=extent)
        ax4.set_title('Ground Truth (LiDAR)', fontsize=10)
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.plot(0, 0, 'b^', markersize=8)

        # Predicted BEV
        ax5 = fig.add_subplot(2, 3, 5)
        im5 = ax5.imshow(pred_np, cmap='hot',
                         origin='lower', extent=extent,
                         vmin=0, vmax=1)
        ax5.set_title('Predicted BEV Occupancy',
                      fontsize=10)
        ax5.set_xlabel('X (m)')
        ax5.plot(0, 0, 'b^', markersize=8)
        plt.colorbar(im5, ax=ax5, fraction=0.046,
                     label='P(occ)')

        # Error map
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(error, cmap='RdYlGn_r',
                   origin='lower', extent=extent,
                   vmin=0, vmax=1)
        ax6.set_title(
            f'Error Map\n'
            f'IoU={metrics.get("occ_iou",0):.3f} | '
            f'DWE={metrics.get("dwe",0):.5f}',
            fontsize=10
        )
        ax6.set_xlabel('X (m)')
        ax6.plot(0, 0, 'b^', markersize=8,
                 label='Ego car')
        ax6.legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            os.makedirs(
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.',
                exist_ok=True
            )
            plt.savefig(save_path, dpi=150,
                        bbox_inches='tight')
            logger.info(f"Full results saved: {save_path}")

        return fig

    except Exception as e:
        raise BEVException(
            "Failed to plot full results", e
        ) from e


# ──────────────────────────────────────────────────────
# 4. Training curves
# ──────────────────────────────────────────────────────

def plot_training_curves(train_losses: list,
                         val_ious:     list,
                         val_dwes:     list,
                         save_path:    str = None
                         ) -> plt.Figure:
    """
    Plot training loss and validation metrics over epochs.
    Shows model is learning.

    Args:
        train_losses: list of loss per epoch
        val_ious:     list of IoU per epoch
        val_dwes:     list of DWE per epoch
        save_path:    where to save
    """
    try:
        epochs = range(1, len(train_losses) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            'Training Progress',
            fontsize=13, fontweight='bold'
        )

        # Loss
        axes[0].plot(epochs, train_losses,
                     'b-o', linewidth=2, markersize=4)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

        # IoU
        axes[1].plot(epochs, val_ious,
                     'g-o', linewidth=2, markersize=4)
        axes[1].set_title('Validation Occupancy IoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU ↑')
        axes[1].grid(True, alpha=0.3)

        # DWE
        axes[2].plot(epochs, val_dwes,
                     'r-o', linewidth=2, markersize=4)
        axes[2].set_title('Validation DWE')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('DWE ↓')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150,
                        bbox_inches='tight')
            logger.info(
                f"Training curves saved: {save_path}"
            )

        return fig

    except Exception as e:
        raise BEVException(
            "Failed to plot training curves", e
        ) from e