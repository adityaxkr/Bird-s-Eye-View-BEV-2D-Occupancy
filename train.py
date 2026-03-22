# train.py
# ══════════════════════════════════════════════════════
# Training script — runs locally on friend's RTX 3050
# All improvements included:
#   ✓ ResNet101 backbone
#   ✓ 800×800 BEV (10cm/pixel)
#   ✓ 50 epochs
#   ✓ Warmup + cosine LR
#   ✓ Distance-weighted focal loss
#   ✓ Save every epoch
#   ✓ Resume from checkpoint
#
# Usage:
#   python train.py
# ══════════════════════════════════════════════════════

import os
import sys
import math
import torch
import torch.optim as optim
from tqdm import tqdm

import config.config as cfg

# ── All improvements set here ──────────────────────────
cfg.BEV_H      = 800
cfg.BEV_W      = 800
cfg.BEV_RES    = (cfg.X_RANGE[1] - cfg.X_RANGE[0]) / cfg.BEV_W
cfg.BATCH_SIZE = 1
cfg.BACKBONE   = 'resnet101'
cfg.EPOCHS     = 50
cfg.LR         = 2e-4
cfg.WEIGHT_DECAY = 1e-4

TOTAL_EPOCHS  = cfg.EPOCHS
WARMUP_EPOCHS = 5

from data.nuscenes_loader   import get_dataloaders
from models.bev_model       import BEVOccupancyModel
from utils.metrics          import compute_metrics
from utils.visualize        import (
    plot_bev_comparison,
    plot_training_curves
)
from logger.custom_logger   import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


# ── LR schedule: warmup + cosine ──────────────────────
def get_lr(epoch: int) -> float:
    """
    Linear warmup for first 5 epochs.
    Cosine annealing after warmup.
    """
    if epoch < WARMUP_EPOCHS:
        return cfg.LR * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (
        TOTAL_EPOCHS - WARMUP_EPOCHS
    )
    return cfg.LR * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def train():
    """Main training function."""

    logger.info("═" * 55)
    logger.info("BEV 2D Occupancy Training")
    logger.info("═" * 55)
    logger.info(f"Device     : {cfg.DEVICE}")
    logger.info(f"Backbone   : {cfg.BACKBONE}")
    logger.info(f"BEV size   : {cfg.BEV_H}×{cfg.BEV_W}")
    logger.info(f"Resolution : {cfg.BEV_RES*100:.0f}cm/pixel")
    logger.info(f"Epochs     : {TOTAL_EPOCHS}")
    logger.info(f"Batch size : {cfg.BATCH_SIZE}")
    logger.info(f"LR         : {cfg.LR}")
    logger.info("═" * 55)

    device = cfg.DEVICE

    # ── Dirs ───────────────────────────────────────────
    os.makedirs(cfg.CKPT_DIR,    exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # ── Data ───────────────────────────────────────────
    logger.info("Loading dataset...")
    train_loader, val_loader, ts, vs = get_dataloaders(
        dataroot = cfg.DATAROOT,
        version  = cfg.VERSION
    )
    logger.info(f"Train: {ts} | Val: {vs}")

    # ── Model ──────────────────────────────────────────
    logger.info(f"Building model with {cfg.BACKBONE}...")
    model = BEVOccupancyModel(pretrained=True).to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    logger.info(f"Total params: {total_params:,}")

    # ── Optimizer ──────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg.LR,
        weight_decay = cfg.WEIGHT_DECAY
    )

    # ── Resume from checkpoint ─────────────────────────
    CKPT_PATH    = os.path.join(cfg.CKPT_DIR, 'best_model.pth')
    start_epoch  = 1
    best_iou     = 0.0
    train_losses = []
    val_ious     = []
    val_dwes     = []

    if os.path.exists(CKPT_PATH):
        logger.info(f"Resuming from: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch  = ckpt['epoch'] + 1
        best_iou     = ckpt['val_iou']
        train_losses = ckpt.get('train_losses', [])
        val_ious     = ckpt.get('val_ious',     [])
        val_dwes     = ckpt.get('val_dwes',     [])
        logger.info(
            f"Resumed from epoch {ckpt['epoch']} | "
            f"Best IoU: {best_iou:.4f}"
        )
    else:
        logger.info("Starting fresh training")

    # ── Training loop ──────────────────────────────────
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

        # Set LR with warmup
        current_lr = get_lr(epoch - 1)
        set_lr(optimizer, current_lr)

        # ── Train phase ────────────────────────────────
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc  = f"Epoch {epoch:02d}/{TOTAL_EPOCHS} [Train]",
            ncols = 80
        )

        for batch in pbar:
            imgs       = batch['imgs'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            occ_gt     = batch['occ_gt'].to(device)

            # Forward
            outputs = model(imgs, intrinsics, extrinsics)

            # Distance-weighted focal + lovász + BCE loss
            losses  = model.compute_loss(outputs, occ_gt)
            loss    = losses['total']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr'  : f"{current_lr:.6f}"
            })

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # ── Validation phase ───────────────────────────
        model.eval()
        val_iou_sum = 0.0
        val_dwe_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc  = f"Epoch {epoch:02d}/{TOTAL_EPOCHS} [Val]",
                ncols = 80
            ):
                imgs       = batch['imgs'].to(device)
                intrinsics = batch['intrinsics'].to(device)
                extrinsics = batch['extrinsics'].to(device)
                occ_gt     = batch['occ_gt'].to(device)

                pred    = model(
                    imgs, intrinsics, extrinsics
                )['occ']
                metrics = compute_metrics(
                    pred,
                    occ_gt.unsqueeze(1)
                )
                val_iou_sum += metrics['occ_iou']
                val_dwe_sum += metrics['dwe']
                val_batches += 1

        avg_iou = val_iou_sum / val_batches
        avg_dwe = val_dwe_sum / val_batches
        val_ious.append(avg_iou)
        val_dwes.append(avg_dwe)

        logger.info(
            f"Epoch {epoch:02d}/{TOTAL_EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"IoU: {avg_iou:.4f} | "
            f"DWE: {avg_dwe:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        # ── Checkpoint data ────────────────────────────
        ckpt_data = {
            'epoch'       : epoch,
            'model_state' : model.state_dict(),
            'optimizer'   : optimizer.state_dict(),
            'val_iou'     : avg_iou,
            'val_dwe'     : avg_dwe,
            'best_iou'    : best_iou,
            'train_losses': train_losses,
            'val_ious'    : val_ious,
            'val_dwes'    : val_dwes,
            'backbone'    : cfg.BACKBONE,
            'bev_size'    : f"{cfg.BEV_H}x{cfg.BEV_W}",
            'bev_res_cm'  : cfg.BEV_RES * 100,
        }

        # Save every epoch
        torch.save(
            ckpt_data,
            os.path.join(
                cfg.CKPT_DIR, f'epoch_{epoch:02d}.pth'
            )
        )

        # Save best model
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(ckpt_data, CKPT_PATH)
            logger.info(
                f"  ✓ Best model saved | IoU: {best_iou:.4f}"
            )
        else:
            logger.info(
                f"  ✓ Epoch {epoch} saved"
            )

        # ── Visualization every 10 epochs ─────────────
        if epoch % 10 == 0:
            with torch.no_grad():
                sample = next(iter(val_loader))
                pred_v = model(
                    sample['imgs'].to(device),
                    sample['intrinsics'].to(device),
                    sample['extrinsics'].to(device)
                )['occ']
                plot_bev_comparison(
                    pred      = pred_v[0].cpu(),
                    gt        = sample['occ_gt'][0],
                    save_path = os.path.join(
                        cfg.RESULTS_DIR,
                        f'bev_epoch_{epoch:02d}.png'
                    ),
                    title = (
                        f"Epoch {epoch} | "
                        f"IoU: {avg_iou:.4f} | "
                        f"DWE: {avg_dwe:.6f}"
                    )
                )

        # ── Training curves every 5 epochs ────────────
        if epoch % 5 == 0 and len(train_losses) > 1:
            plot_training_curves(
                train_losses = train_losses,
                val_ious     = val_ious,
                val_dwes     = val_dwes,
                save_path    = os.path.join(
                    cfg.RESULTS_DIR,
                    'training_curves.png'
                )
            )

    # ── Training complete ─────────────────────────────
    logger.info("═" * 55)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best IoU      : {best_iou:.4f}")
    logger.info(
        f"BEV resolution: "
        f"{cfg.BEV_H}×{cfg.BEV_W} "
        f"({cfg.BEV_RES*100:.0f}cm/px)"
    )
    logger.info(f"Backbone      : {cfg.BACKBONE}")
    logger.info(f"Total epochs  : {TOTAL_EPOCHS}")
    logger.info(f"Checkpoints   : {cfg.CKPT_DIR}/")
    logger.info(f"Results       : {cfg.RESULTS_DIR}/")
    logger.info("═" * 55)

    # Final training curves
    plot_training_curves(
        train_losses = train_losses,
        val_ious     = val_ious,
        val_dwes     = val_dwes,
        save_path    = os.path.join(
            cfg.RESULTS_DIR,
            'final_training_curves.png'
        )
    )
    logger.info("Final training curves saved ✓")

    return best_iou


if __name__ == '__main__':
    train()