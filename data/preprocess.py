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

# data/preprocess.py — FIXED build_bev_occupancy
# Change 1: Add Z-height filter (CRITICAL — removes ground contamination)
# Change 2: No other changes needed

def build_bev_occupancy(
    lidar_points: np.ndarray,       # (4, N) — x,y,z,intensity
    lidar_rotation: list,
    lidar_translation: list
) -> torch.Tensor:
    try:
        points = lidar_points[:3, :].T    # (N, 3)

        # Transform LiDAR → ego frame
        rot   = Quaternion(lidar_rotation).rotation_matrix
        trans = np.array(lidar_translation, dtype=np.float32)
        points = (rot @ points.T).T + trans   # (N, 3)

        # ── Z-HEIGHT FILTER (THE FIX) ──────────────────────────────
        # In ego frame, ground is at approximately Z ≈ -ego_height ≈ -1.7m
        # After lidar→ego transform, ground returns cluster at Z ≈ -1.5 to 0.1m
        # Objects (cars, pedestrians) are at Z ≈ 0.2m to 3.5m in ego frame
        z_ego = points[:, 2]
        height_mask = (z_ego > 0.2) & (z_ego < 3.5)
        points = points[height_mask]
        # ──────────────────────────────────────────────────────────

        occ   = np.zeros((BEV_H, BEV_W), dtype=np.float32)
        x_res = (X_RANGE[1] - X_RANGE[0]) / BEV_W  # 0.4 m/px
        y_res = (Y_RANGE[1] - Y_RANGE[0]) / BEV_H  # 0.4 m/px

        col = ((points[:, 0] - X_RANGE[0]) / x_res).astype(int)
        row = ((points[:, 1] - Y_RANGE[0]) / y_res).astype(int)

        mask = (col >= 0) & (col < BEV_W) & (row >= 0) & (row < BEV_H)
        occ[row[mask], col[mask]] = 1.0

        occupied = int(occ.sum())
        total    = BEV_H * BEV_W
        logger.info(
            f"BEV GT built | occupied: {occupied}/{total} "
            f"({100*occupied/total:.1f}%) | after Z-filter"
        )
        return torch.tensor(occ, dtype=torch.float32)

    except Exception as e:
        raise BEVException("Failed to build BEV occupancy grid", e) from e