# models/bev_former_lite.py
# ══════════════════════════════════════════════════════════════════
# BEVFormer-Lite: Geometry-based View Transformer
# V4: Removed CPU roundtrip on torch.linalg.inv
#     V3 forced E_cam2ego to CPU for inversion every forward pass
#     — unnecessary on modern PyTorch, CUDA linalg.inv is stable
#       for 4×4 matrices. Removed the .cpu() / .to(device) roundtrip.
#
# Multi-height Z sampling [0.0, 0.5, 1.5] — unchanged from V3.1
# ══════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import (
    IMG_CHANNELS,
    BEV_H, BEV_W,
    X_RANGE, Y_RANGE,
    IMG_H, IMG_W
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)

# Multi-height levels in ego frame (metres)
# 0.0 → ground plane (road, flat objects)
# 0.5 → bumper height (lower car body)
# 1.5 → roof height (car roof, pedestrian torso)
Z_HEIGHTS = [0.0, 0.5, 1.5]


class BEVFormerLite(nn.Module):
    """
    Geometry-based BEV View Transformer.
    Stored extrinsics (from preprocess_extrinsic) are T_cam→ego.
    We invert them inside forward() to get T_ego→cam.

    Input:
        feats:      (B, N_cam, C, fH, fW) backbone features
        intrinsics: (B, N_cam, 3, 3)      K matrices (scaled)
        extrinsics: (B, N_cam, 4, 4)      T_cam→ego
    Output:
        bev_feat:   (B, C, BEV_H, BEV_W)
    """

    def __init__(
        self,
        in_channels: int  = IMG_CHANNELS,
        bev_h: int        = BEV_H,
        bev_w: int        = BEV_W,
        num_cams: int     = 6,
        z_heights: list   = Z_HEIGHTS
    ):
        super().__init__()
        try:
            self.C         = in_channels
            self.bev_h     = bev_h
            self.bev_w     = bev_w
            self.num_cams  = num_cams
            self.z_heights = z_heights

            # BEV grid cell centers
            x_half = (X_RANGE[1] - X_RANGE[0]) / (2 * bev_w)
            y_half = (Y_RANGE[1] - Y_RANGE[0]) / (2 * bev_h)
            xs = torch.linspace(X_RANGE[0] + x_half, X_RANGE[1] - x_half, bev_w)
            ys = torch.linspace(Y_RANGE[0] + y_half, Y_RANGE[1] - y_half, bev_h)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

            # One buffer per Z height
            for i, z_val in enumerate(z_heights):
                grid_z  = torch.full_like(grid_x, z_val)
                bev_pts = torch.stack(
                    [grid_x, grid_y, grid_z], dim=-1
                ).reshape(-1, 3)                        # (N_pts, 3)
                self.register_buffer(f'bev_pts_z{i}', bev_pts)

            # Channel refinement
            self.channel_reduce = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

            logger.info(
                f"BEVFormerLite V4 | "
                f"BEV={bev_h}×{bev_w} | "
                f"Z heights: {z_heights} | "
                f"N_pts/height={bev_h * bev_w} | "
                f"C={in_channels}"
            )
        except Exception as e:
            raise BEVException("Failed to init BEVFormerLite", e) from e

    def _sample_one_height(
        self,
        pts_ego:   torch.Tensor,    # (N_pts, 3)
        feats:     torch.Tensor,    # (B, N_cam, C, fH, fW)
        intrinsics:torch.Tensor,    # (B, N_cam, 3, 3)
        R_ego2cam: torch.Tensor,    # (B, N_cam, 3, 3)
        t_ego2cam: torch.Tensor,    # (B, N_cam, 3)
        fH: int, fW: int
    ) -> torch.Tensor:
        """
        Sample BEV features at one Z height across all cameras.
        Returns: (B, C, N_pts) — mean over valid cameras
        """
        B     = feats.shape[0]
        N     = feats.shape[1]
        C     = feats.shape[2]
        N_pts = pts_ego.shape[0]
        device = feats.device

        bev_feat_sum = torch.zeros(B, C, N_pts, device=device)
        valid_sum    = torch.zeros(B, 1, N_pts, device=device)

        for cam_idx in range(N):
            # ego → camera frame
            R = R_ego2cam[:, cam_idx]               # (B, 3, 3)
            t = t_ego2cam[:, cam_idx]               # (B, 3)
            pts_ego_b = pts_ego.unsqueeze(0).expand(B, -1, -1)
            pts_cam   = torch.bmm(pts_ego_b, R.transpose(1, 2)) \
                        + t.unsqueeze(1)            # (B, N_pts, 3)

            depth = pts_cam[:, :, 2]                # (B, N_pts)

            # camera frame → image pixels
            K     = intrinsics[:, cam_idx]          # (B, 3, 3)
            p_img = torch.bmm(pts_cam, K.transpose(1, 2))
            u = p_img[:, :, 0] / (p_img[:, :, 2] + 1e-6)
            v = p_img[:, :, 1] / (p_img[:, :, 2] + 1e-6)

            # scale to feature map resolution
            scale_u = fW / IMG_W
            scale_v = fH / IMG_H
            u_feat  = u * scale_u
            v_feat  = v * scale_v

            # normalize to [-1, 1] for grid_sample
            u_norm = (u_feat / (fW - 1.0)) * 2.0 - 1.0
            v_norm = (v_feat / (fH - 1.0)) * 2.0 - 1.0

            # validity mask
            valid = (
                (depth   >  0.1)  &
                (u_norm  >= -1.0) &
                (u_norm  <=  1.0) &
                (v_norm  >= -1.0) &
                (v_norm  <=  1.0)
            ).float()                               # (B, N_pts)

            # grid_sample
            feat_map = feats[:, cam_idx]            # (B, C, fH, fW)
            grid     = torch.stack(
                [u_norm, v_norm], dim=-1
            ).unsqueeze(1)                          # (B, 1, N_pts, 2)

            sampled = F.grid_sample(
                feat_map, grid,
                mode          = 'bilinear',
                padding_mode  = 'zeros',
                align_corners = True
            ).squeeze(2)                            # (B, C, N_pts)

            valid_bc  = valid.unsqueeze(1)          # (B, 1, N_pts)
            sampled   = sampled * valid_bc
            bev_feat_sum += sampled
            valid_sum    += valid_bc

        return bev_feat_sum / (valid_sum + 1e-6)    # (B, C, N_pts)

    def forward(
        self,
        feats:      torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns bev_feat: (B, C, BEV_H, BEV_W)
        """
        try:
            B, N, C, fH, fW = feats.shape
            device = feats.device

            # ── Step 1: invert extrinsics ONCE (shared across all heights)
            # ✅ V4 FIX: removed .cpu() / .to(device) roundtrip
            #    V3 did: torch.linalg.inv(...cpu()...).to(device)
            #    This forced a GPU→CPU→GPU copy every forward pass (~12 matrices)
            #    torch.linalg.inv is numerically stable on CUDA for 4×4 matrices
            E_cam2ego = extrinsics                          # (B, N, 4, 4)
            E_ego2cam = torch.linalg.inv(
                E_cam2ego.reshape(B * N, 4, 4)             # stay on device
            ).reshape(B, N, 4, 4)

            R_ego2cam = E_ego2cam[:, :, :3, :3]            # (B, N, 3, 3)
            t_ego2cam = E_ego2cam[:, :, :3,  3]            # (B, N, 3)

            # ── Step 2: sample at each Z height
            height_feats = []
            for i in range(len(self.z_heights)):
                pts_ego = getattr(self, f'bev_pts_z{i}')   # (N_pts, 3)
                feat_z  = self._sample_one_height(
                    pts_ego   = pts_ego,
                    feats     = feats,
                    intrinsics= intrinsics,
                    R_ego2cam = R_ego2cam,
                    t_ego2cam = t_ego2cam,
                    fH=fH, fW=fW
                )                                          # (B, C, N_pts)
                height_feats.append(feat_z)

            # ── Step 3: mean pool across heights → (B, C, N_pts)
            bev_feat_avg = torch.stack(
                height_feats, dim=0
            ).mean(dim=0)

            # ── Step 4: reshape flat → spatial
            bev_feat = bev_feat_avg.reshape(
                B, C, self.bev_h, self.bev_w
            )                                              # (B, 128, 200, 200)

            # ── Step 5: refine
            bev_feat = self.channel_reduce(bev_feat)

            logger.debug(
                f"BEVFormerLite V4 | "
                f"Z={self.z_heights} | "
                f"bev: {tuple(bev_feat.shape)}"
            )
            return bev_feat

        except Exception as e:
            raise BEVException("BEVFormerLite forward failed", e) from e