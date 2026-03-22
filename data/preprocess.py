# data/preprocess.py
# ══════════════════════════════════════════════════════
# Preprocessing functions for BEV Occupancy Project
# Handles: images, camera params, LiDAR → BEV grid
# ══════════════════════════════════════════════════════

import numpy as np
import cv2
import torch
from pyquaternion import Quaternion

from config.config import (
    IMG_H, IMG_W,
    IMG_MEAN, IMG_STD,
    BEV_H, BEV_W,
    X_RANGE, Y_RANGE
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


# ──────────────────────────────────────────────────────
# 1. Image preprocessing
# ──────────────────────────────────────────────────────

def preprocess_image(img_path: str) -> torch.Tensor:
    """
    Load one camera image, resize and normalize it.

    Args:
        img_path: full path to the .jpg camera image

    Returns:
        tensor of shape (3, IMG_H, IMG_W)
        values normalized with ImageNet mean/std
    """
    try:
        # ── Load ───────────────────────────────────────
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(
                f"Image not found at: {img_path}"
            )

        # ── BGR → RGB ──────────────────────────────────
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Resize ─────────────────────────────────────
        # Original: 1600×900 → Target: 704×256
        img = cv2.resize(
            img,
            (IMG_W, IMG_H),           # cv2 takes (W, H)
            interpolation=cv2.INTER_LINEAR
        )

        # ── Normalize to [0, 1] ────────────────────────
        img = img.astype(np.float32) / 255.0

        # ── ImageNet normalization ─────────────────────
        # Makes pretrained ResNet work better
        mean = np.array(IMG_MEAN, dtype=np.float32)
        std  = np.array(IMG_STD,  dtype=np.float32)
        img  = (img - mean) / std

        # ── HWC → CHW (PyTorch format) ─────────────────
        img = img.transpose(2, 0, 1)   # (H,W,3) → (3,H,W)

        tensor = torch.tensor(img, dtype=torch.float32)

        logger.info(
            f"Image preprocessed | "
            f"shape: {tuple(tensor.shape)}"
        )
        return tensor

    except Exception as e:
        raise BEVException(
            f"Failed to preprocess image: {img_path}", e
        ) from e


# ──────────────────────────────────────────────────────
# 2. Camera parameter preprocessing
# ──────────────────────────────────────────────────────

def preprocess_intrinsic(
    camera_intrinsic: list,
    orig_w: int,
    orig_h: int
) -> torch.Tensor:
    """
    Build and scale the camera intrinsic matrix K.

    The intrinsic matrix K maps 3D camera coords → 2D pixels:
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    Since we resize the image, we must scale fx, cx, fy, cy.

    Args:
        camera_intrinsic: 3×3 list from nuScenes calibration
        orig_w: original image width  (1600)
        orig_h: original image height (900)

    Returns:
        K tensor of shape (3, 3)
    """
    try:
        K = np.array(camera_intrinsic, dtype=np.float32)

        # Scale factors because we resized the image
        scale_w = IMG_W / orig_w   # 704 / 1600
        scale_h = IMG_H / orig_h   # 256 / 900

        # Scale focal lengths and principal point
        K[0, 0] *= scale_w   # fx
        K[0, 2] *= scale_w   # cx
        K[1, 1] *= scale_h   # fy
        K[1, 2] *= scale_h   # cy

        logger.info("Intrinsic K scaled for resized image")
        return torch.tensor(K, dtype=torch.float32)

    except Exception as e:
        raise BEVException(
            "Failed to preprocess camera intrinsic", e
        ) from e


def preprocess_extrinsic(
    rotation: list,
    translation: list
) -> torch.Tensor:
    """
    Build the camera extrinsic matrix E.

    The extrinsic matrix E describes where the camera is
    relative to the ego car:
        E = [[R | t],
             [0 | 1]]   (4×4 matrix)

    Args:
        rotation:    quaternion [w, x, y, z] from nuScenes
        translation: [x, y, z] in metres from nuScenes

    Returns:
        E tensor of shape (4, 4)
    """
    try:
        # Convert quaternion → 3×3 rotation matrix
        rot   = Quaternion(rotation).rotation_matrix
        trans = np.array(translation, dtype=np.float32)

        # Build 4×4 extrinsic matrix
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = rot
        E[:3,  3] = trans

        logger.info("Extrinsic E matrix built")
        return torch.tensor(E, dtype=torch.float32)

    except Exception as e:
        raise BEVException(
            "Failed to preprocess camera extrinsic", e
        ) from e


# ──────────────────────────────────────────────────────
# 3. LiDAR → BEV occupancy grid
# ──────────────────────────────────────────────────────

def build_bev_occupancy(
    lidar_points: np.ndarray,
    lidar_rotation: list,
    lidar_translation: list
) -> torch.Tensor:
    """
    Convert LiDAR 3D points into a 2D BEV occupancy grid.

    Steps:
        1. Transform points from LiDAR frame → ego car frame
        2. Filter points that fall inside our BEV range
        3. Convert real-world (x,y) coords → grid (row,col)
        4. Mark those grid cells as occupied (1.0)

    Args:
        lidar_points:       (N, 4) array — x,y,z,intensity
        lidar_rotation:     quaternion from nuScenes calib
        lidar_translation:  [x,y,z] from nuScenes calib

    Returns:
        occupancy grid tensor of shape (BEV_H, BEV_W)
        dtype float32, values 0.0 or 1.0
    """
    try:
        # ── Take only x, y, z columns ──────────────────
        points = lidar_points[:3, :].T    # (N, 3)

        # ── Transform LiDAR → ego car frame ────────────
        rot   = Quaternion(lidar_rotation).rotation_matrix
        trans = np.array(lidar_translation, dtype=np.float32)

        # Apply rotation then translation
        points = (rot @ points.T).T + trans   # (N, 3)

        # ── Build empty occupancy grid ──────────────────
        occ = np.zeros(
            (BEV_H, BEV_W), dtype=np.float32
        )

        # ── Resolution: metres per pixel ───────────────
        x_res = (X_RANGE[1] - X_RANGE[0]) / BEV_W  # 0.4 m
        y_res = (Y_RANGE[1] - Y_RANGE[0]) / BEV_H  # 0.4 m

        # ── Convert real-world → grid coordinates ──────
        col = ((points[:, 0] - X_RANGE[0]) / x_res).astype(int)
        row = ((points[:, 1] - Y_RANGE[0]) / y_res).astype(int)

        # ── Filter: keep only points inside grid ───────
        mask = (
            (col >= 0) & (col < BEV_W) &
            (row >= 0) & (row < BEV_H)
        )

        # ── Mark occupied cells ─────────────────────────
        occ[row[mask], col[mask]] = 1.0

        occupied = int(occ.sum())
        total    = BEV_H * BEV_W
        logger.info(
            f"BEV grid built | "
            f"occupied: {occupied}/{total} cells "
            f"({100*occupied/total:.1f}%)"
        )

        return torch.tensor(occ, dtype=torch.float32)

    except Exception as e:
        raise BEVException(
            "Failed to build BEV occupancy grid", e
        ) from e
    # ADD at bottom of data/preprocess.py

import random

def augment_image(img:      torch.Tensor,
                  is_train: bool = True
                  ) -> torch.Tensor:
    """
    Data augmentation — from Paper 1 Section 4.1.
    Only applied during training.

    Augmentations:
      - Random brightness/contrast
      - Random horizontal flip
      - Random scale 0.9-1.1
      - Color jitter

    Proven to improve BEV perception performance.
    """
    if not is_train:
        return img

    img_np = img.permute(1, 2, 0).numpy().copy()

    # ── Brightness jitter ───────────────────────────
    if random.random() > 0.5:
        factor  = random.uniform(0.7, 1.3)
        img_np  = np.clip(img_np * factor, 0, 1)

    # ── Contrast jitter ─────────────────────────────
    if random.random() > 0.5:
        mean    = img_np.mean()
        factor  = random.uniform(0.8, 1.2)
        img_np  = np.clip(
            (img_np - mean) * factor + mean, 0, 1
        )

    # ── Color jitter ────────────────────────────────
    if random.random() > 0.5:
        for c in range(3):
            factor     = random.uniform(0.9, 1.1)
            img_np[:,:,c] = np.clip(
                img_np[:,:,c] * factor, 0, 1
            )

    return torch.tensor(
        img_np.transpose(2, 0, 1),
        dtype=torch.float32
    )


def augment_bev(occ:      np.ndarray,
                flip:     bool = False
                ) -> np.ndarray:
    """
    BEV occupancy grid augmentation.
    Must match image augmentation (same flip).
    """
    if flip:
        occ = np.fliplr(occ).copy()
    return occ