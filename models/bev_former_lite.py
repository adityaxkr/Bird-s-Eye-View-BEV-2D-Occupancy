# models/bev_former_lite.py
# ══════════════════════════════════════════════════════════════════
# BEVFormer-Lite: Geometry-based View Transformer
#
# Replaces: lss_transformer.py entirely
#
# Core math (Delving into Devils, Eq. 5):
#   F3D(x,y,z) = sample(F_2D, project(P_ego → camera → image))
#
# Pipeline per BEV cell:
#   1. P_cam  = R_ego2cam @ P_ego + t_ego2cam
#   2. P_img  = K @ P_cam
#   3. (u,v)  = P_img[:2] / P_img[2]   (perspective divide)
#   4. feat   = grid_sample(feat_map, normalize(u,v))
#   5. fuse   = mean over valid cameras
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


class BEVFormerLite(nn.Module):
    """
    Geometry-based BEV View Transformer.

    Stored extrinsics (from preprocess_extrinsic) are T_cam→ego.
    We invert them inside forward() to get T_ego→cam.
    This is intentional — preprocess.py stays unchanged.

    Input:
        feats:      (B, N_cam, C, fH, fW)  backbone features
        intrinsics: (B, N_cam, 3, 3)       K matrices (scaled)
        extrinsics: (B, N_cam, 4, 4)       T_cam→ego

    Output:
        bev_feat: (B, C, BEV_H, BEV_W)
    """

    def __init__(
        self,
        in_channels: int = IMG_CHANNELS,   # 128
        bev_h:       int = BEV_H,          # 200
        bev_w:       int = BEV_W,          # 200
        num_cams:    int = 6
    ):
        super().__init__()

        try:
            self.C        = in_channels
            self.bev_h    = bev_h
            self.bev_w    = bev_w
            self.num_cams = num_cams

            # ── BEV grid: cell centers in ego frame ───────────────────
            # X_RANGE=(-40,40), Y_RANGE=(-40,40), res=0.4m → 200×200
            # Cell centers avoid exact boundary values:
            #   linspace(-39.8, 39.8, 200) — 0.4m spacing, centered
            x_half = (X_RANGE[1] - X_RANGE[0]) / (2 * bev_w)  # 0.2
            y_half = (Y_RANGE[1] - Y_RANGE[0]) / (2 * bev_h)  # 0.2

            xs = torch.linspace(
                X_RANGE[0] + x_half,
                X_RANGE[1] - x_half,
                bev_w
            )   # (200,) — X = forward axis
            ys = torch.linspace(
                Y_RANGE[0] + y_half,
                Y_RANGE[1] - y_half,
                bev_h
            )   # (200,) — Y = left axis

            # indexing='ij': grid_x varies along dim-1 (columns = X)
            #                grid_y varies along dim-0 (rows = Y)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            # Both: (bev_h, bev_w) = (200, 200)

            # Z=0 in ego frame = ground plane
            # Valid for 2D BEV occupancy baseline
            grid_z = torch.zeros_like(grid_x)   # (200, 200)

            # Stack homogeneous coords: [X, Y, Z, 1]
            bev_pts = torch.stack([
                grid_x,
                grid_y,
                grid_z,
                torch.ones_like(grid_x)
            ], dim=-1)   # (200, 200, 4)

            # Flatten to (N, 4) where N = 200*200 = 40000
            bev_pts = bev_pts.reshape(-1, 4)

            # register_buffer → moves to GPU with model.to(device)
            # NOT a trainable parameter
            self.register_buffer('bev_pts', bev_pts)

            # ── Lightweight channel refinement after fusion ───────────
            self.channel_reduce = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

            logger.info(
                f"BEVFormerLite initialized | "
                f"BEV={bev_h}×{bev_w} | "
                f"N_pts={bev_h * bev_w} | "
                f"C={in_channels} | "
                f"Z=0 ground plane assumption"
            )

        except Exception as e:
            raise BEVException(
                "Failed to init BEVFormerLite", e
            ) from e

    def forward(
        self,
        feats:      torch.Tensor,   # (B, N_cam, C, fH, fW)
        intrinsics: torch.Tensor,   # (B, N_cam, 3, 3)
        extrinsics: torch.Tensor    # (B, N_cam, 4, 4) — T_cam→ego
    ) -> torch.Tensor:
        """
        Returns bev_feat: (B, C, BEV_H, BEV_W)
        """
        try:
            B, N, C, fH, fW = feats.shape
            N_pts  = self.bev_h * self.bev_w   # 40000
            device = feats.device

            # ── Step 1: invert extrinsics to get T_ego→cam ────────────
            # preprocess_extrinsic stores T_cam→ego (camera-to-ego).
            # We need T_ego→cam to project ego-frame BEV points
            # into each camera's coordinate frame.
            E_cam2ego = extrinsics                          # (B, N, 4, 4)
            E_ego2cam = torch.linalg.inv(
                E_cam2ego.reshape(B * N, 4, 4).cpu()
            ).to(device).reshape(B, N, 4, 4)                     # (B, N, 4, 4)

            R_ego2cam = E_ego2cam[:, :, :3, :3]            # (B, N, 3, 3)
            t_ego2cam = E_ego2cam[:, :, :3,  3]            # (B, N, 3)

            # ── Step 2: BEV points (ego frame) ────────────────────────
            # bev_pts: (N_pts, 4) — registered buffer, already on device
            pts_ego = self.bev_pts[:, :3]   # (N_pts, 3) — drop homogeneous

            # ── Step 3: accumulate features — camera loop ─────────────
            # Memory-efficient: O(B × C × N) peak, NOT O(B × N_cam × C × N)
            bev_feat_sum = torch.zeros(B, C, N_pts, device=device)
            valid_sum    = torch.zeros(B, 1, N_pts, device=device)

            for cam_idx in range(N):

                # ── 3a: ego → camera frame ────────────────────────────
                # P_cam = R_ego2cam @ P_ego + t_ego2cam
                R = R_ego2cam[:, cam_idx]   # (B, 3, 3)
                t = t_ego2cam[:, cam_idx]   # (B, 3)

                # pts_ego: (N, 3) → expand to (B, N, 3) → bmm
                pts_ego_b = pts_ego.unsqueeze(0).expand(B, -1, -1)
                # (B, N, 3) @ (B, 3, 3).T → (B, N, 3)
                pts_cam = torch.bmm(pts_ego_b, R.transpose(1, 2)) \
                          + t.unsqueeze(1)
                # pts_cam: (B, N_pts, 3) = [X_cam, Y_cam, Z_cam]

                depth = pts_cam[:, :, 2]   # (B, N_pts) — Z in camera frame

                # ── 3b: camera frame → image pixels ───────────────────
                # [u*d, v*d, d] = K @ [X_cam, Y_cam, Z_cam]
                K = intrinsics[:, cam_idx]   # (B, 3, 3)

                # pts_cam: (B, N, 3) → (B, N, 3)
                p_img = torch.bmm(pts_cam, K.transpose(1, 2))
                # p_img: (B, N_pts, 3)

                # Perspective divide — eps prevents /0
                u = p_img[:, :, 0] / (p_img[:, :, 2] + 1e-6)
                v = p_img[:, :, 1] / (p_img[:, :, 2] + 1e-6)
                # u, v: (B, N_pts) — pixel coords at RESIZED image resolution
                # (704×256 after preprocess_image resize)

                # ── 3c: scale to feature map resolution ───────────────
                # Backbone outputs stride-8 features
                # fH=32, fW=88 for input 256×704
                # scale_u = 88/704 = 0.125, scale_v = 32/256 = 0.125
                scale_u = fW / IMG_W   # feature_w / resized_img_w
                scale_v = fH / IMG_H   # feature_h / resized_img_h

                u_feat = u * scale_u   # coords in feature map space
                v_feat = v * scale_v

                # ── 3d: normalize to [-1, 1] for grid_sample ──────────
                # align_corners=True convention:
                #   pixel 0       → -1.0
                #   pixel (W-1)   → +1.0
                # Formula: norm = (pixel / (size-1)) * 2 - 1
                u_norm = (u_feat / (fW - 1.0)) * 2.0 - 1.0  # (B, N_pts)
                v_norm = (v_feat / (fH - 1.0)) * 2.0 - 1.0  # (B, N_pts)

                # ── 3e: validity mask ──────────────────────────────────
                # A BEV cell is VALID for this camera if:
                #   1. It's in FRONT (depth > 0) — behind-camera points
                #      can have valid (u,v) after perspective divide
                #   2. Its projection lands within the image bounds
                valid = (
                    (depth    >  0.1)  &   # in front of camera
                    (u_norm   >= -1.0) &   # within left bound
                    (u_norm   <=  1.0) &   # within right bound
                    (v_norm   >= -1.0) &   # within top bound
                    (v_norm   <=  1.0)     # within bottom bound
                ).float()   # (B, N_pts)

                # ── 3f: grid_sample ────────────────────────────────────
                feat_map = feats[:, cam_idx]   # (B, C, fH, fW)

                # grid_sample expects grid: (B, H_out, W_out, 2)
                # Treat N_pts as a 1D sequence: (B, 1, N_pts, 2)
                # grid[..., 0] = u_norm (x/horizontal)
                # grid[..., 1] = v_norm (y/vertical)
                grid = torch.stack([u_norm, v_norm], dim=-1)
                grid = grid.unsqueeze(1)   # (B, 1, N_pts, 2)

                sampled = F.grid_sample(
                    feat_map,
                    grid,
                    mode         = 'bilinear',
                    padding_mode = 'zeros',    # OOV → 0, not edge garbage
                    align_corners= True        # matches normalization above
                )
                # sampled: (B, C, 1, N_pts) → squeeze → (B, C, N_pts)
                sampled = sampled.squeeze(2)

                # padding_mode='zeros' handles spatial OOV.
                # But depth<0 (behind camera) needs explicit masking —
                # those points can still have (u,v) inside image bounds
                valid_bc = valid.unsqueeze(1)        # (B, 1, N_pts)
                sampled  = sampled * valid_bc        # zero out invalid

                # ── 3g: accumulate ────────────────────────────────────
                bev_feat_sum += sampled     # (B, C, N_pts)
                valid_sum    += valid_bc    # (B, 1, N_pts)

            # ── Step 4: mean pool over valid cameras ──────────────────
            # Cells with 0 valid cameras stay 0 (unknown region)
            bev_feat_avg = bev_feat_sum / (valid_sum + 1e-6)
            # (B, C, N_pts)

            # ── Step 5: reshape flat → spatial ────────────────────────
            bev_feat = bev_feat_avg.reshape(B, C, self.bev_h, self.bev_w)
            # (B, 128, 200, 200)

            # ── Step 6: refine ────────────────────────────────────────
            bev_feat = self.channel_reduce(bev_feat)
            # (B, 128, 200, 200)

            logger.debug(
                f"BEVFormerLite forward | bev: {tuple(bev_feat.shape)}"
            )
            return bev_feat

        except Exception as e:
            raise BEVException(
                "BEVFormerLite forward failed", e
            ) from e