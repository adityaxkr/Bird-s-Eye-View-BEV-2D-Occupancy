# data/nuscenes_loader.py
# ══════════════════════════════════════════════════════
# PyTorch Dataset + DataLoader for nuScenes BEV project
# Uses preprocess.py functions for all transformations
# ══════════════════════════════════════════════════════

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from config.config import (
    DATAROOT, VERSION,
    CAM_NAMES,
    BATCH_SIZE, NUM_WORKERS,
    TRAIN_SPLIT
)
from data.preprocess import (
    preprocess_image,
    preprocess_intrinsic,
    preprocess_extrinsic,
    build_bev_occupancy
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class BEVOccupancyDataset(Dataset):
    """
    PyTorch Dataset for nuScenes BEV 2D Occupancy.

    For each sample returns:
        imgs        → (6, 3, IMG_H, IMG_W)  6 camera images
        intrinsics  → (6, 3, 3)             camera K matrices
        extrinsics  → (6, 4, 4)             camera E matrices
        occ_gt      → (BEV_H, BEV_W)        LiDAR occupancy GT

    Usage:
        dataset = BEVOccupancyDataset()
        sample  = dataset[0]
        print(sample['imgs'].shape)  # (6, 3, 256, 704)
    """

    def __init__(self,
             dataroot: str  = DATAROOT,
             version:  str  = VERSION,
             is_train: bool = True):   # ← ADD THIS
        try:
            self.nusc     = NuScenes(
                version=version, dataroot=dataroot, verbose=False
            )
            self.samples  = self.nusc.sample
            self.is_train = is_train   # ← ADD THIS

            logger.info(
                f"Dataset loaded | "
                f"samples: {len(self.samples)} | "
                f"mode: {'train' if is_train else 'val'}"
            )
        except Exception as e:
            raise BEVException(
                "Failed to init dataset", e
            ) from e

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns one fully preprocessed sample.
        Called automatically by DataLoader.
        """
        try:
            sample = self.samples[idx]

            imgs        = []
            intrinsics  = []
            extrinsics  = []

            # ── Process all 6 cameras ───────────────────
            for cam_name in CAM_NAMES:
                img, K, E = self._load_camera(sample, cam_name)
                imgs.append(img)
                intrinsics.append(K)
                extrinsics.append(E)

            # ── Build BEV ground truth from LiDAR ───────
            occ_gt = self._load_lidar_bev(sample)

            return {
                # Stack 6 cameras along dim 0
                'imgs'       : torch.stack(imgs),        # (6,3,H,W)
                'intrinsics' : torch.stack(intrinsics),  # (6,3,3)
                'extrinsics' : torch.stack(extrinsics),  # (6,4,4)
                'occ_gt'     : occ_gt                    # (200,200)
            }

        except Exception as e:
            raise BEVException(
                f"Failed to load sample at index {idx}", e
            ) from e

    # ──────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────

    def _load_camera(self, sample, cam_name):
        cam_token = sample['data'][cam_name]
        cam_data  = self.nusc.get('sample_data', cam_token)
        calib     = self.nusc.get(
            'calibrated_sensor',
            cam_data['calibrated_sensor_token']
        )

        img_path = os.path.join(
            self.nusc.dataroot, cam_data['filename']
        )

        img = preprocess_image(img_path)

        # Apply augmentation during training
        if self.is_train:
            from data.preprocess import augment_image
            img = augment_image(img, is_train=True)

        K = preprocess_intrinsic(
            calib['camera_intrinsic'],
            orig_w=cam_data['width'],
            orig_h=cam_data['height']
        )
        E = preprocess_extrinsic(
            calib['rotation'],
            calib['translation']
        )
        return img, K, E
    
    def _load_lidar_bev(self, sample: dict) -> torch.Tensor:
        """
        Load LiDAR scan and convert to 2D BEV occupancy grid.

        Returns:
            occ: tensor (BEV_H, BEV_W) — float32, 0 or 1
        """
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data  = self.nusc.get('sample_data', lidar_token)
        lidar_calib = self.nusc.get(
            'calibrated_sensor',
            lidar_data['calibrated_sensor_token']
        )

        # Load point cloud
        pc_path = os.path.join(
            self.nusc.dataroot,
            lidar_data['filename']
        )
        pc = LidarPointCloud.from_file(pc_path)

        # Build BEV occupancy grid
        occ = build_bev_occupancy(
            lidar_points      = pc.points,
            lidar_rotation    = lidar_calib['rotation'],
            lidar_translation = lidar_calib['translation']
        )

        return occ


# ──────────────────────────────────────────────────────
# DataLoader factory function
# ──────────────────────────────────────────────────────

def get_dataloaders(dataroot=DATAROOT, version=VERSION):
    full_dataset = BEVOccupancyDataset(
        dataroot=dataroot, version=version, is_train=True
    )

    total      = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total)
    val_size   = total - train_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Val set should NOT augment
    val_ds.dataset.is_train = False

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    logger.info(
        f"DataLoaders | train: {train_size} | val: {val_size}"
    )
    return train_loader, val_loader, train_size, val_size