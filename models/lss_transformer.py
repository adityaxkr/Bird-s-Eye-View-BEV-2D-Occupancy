# models/lss_transformer.py
# ══════════════════════════════════════════════════════
# LSS View Transformer — mathematically correct
# Based on: Eq 6 from "Delving into Devils of BEV"
# Implements: Lift → Splat using real camera geometry
#
# HACKATHON GOAL: Image → BEV 2D Occupancy Grid
# ══════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import (
    IMG_CHANNELS, DEPTH_BINS,
    BEV_H, BEV_W,
    X_RANGE, Y_RANGE, Z_RANGE,
    IMG_H, IMG_W
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class LSSViewTransformer(nn.Module):
    """
    Lift-Splat-Shoot View Transformer.

    Implements Eq 6 from "Delving into Devils of BEV":
        F3D(x,y,z) = F*2D(û,v̂) ⊗ D*(û,v̂)|xyz

    Steps:
        LIFT  — predict depth per pixel + outer product
                → frustum of 3D features per camera
        SPLAT — project 3D points onto BEV grid
                using real camera K and E matrices
                → unified (B, C, BEV_H, BEV_W)

    Input:
        feats:      (B, 6, C, fH, fW)
        intrinsics: (B, 6, 3, 3)   K matrix
        extrinsics: (B, 6, 4, 4)   E matrix [R|t]

    Output:
        bev: (B, C, BEV_H, BEV_W)
    """

    def __init__(self,
                 in_channels: int = IMG_CHANNELS,
                 depth_bins:  int = DEPTH_BINS,
                 bev_h:       int = BEV_H,
                 bev_w:       int = BEV_W,
                 d_min:       float = 2.0,    # min depth (metres)
                 d_max:       float = 58.0):  # max depth (metres)
        super().__init__()

        try:
            self.C     = in_channels
            self.D     = depth_bins
            self.bev_h = bev_h
            self.bev_w = bev_w
            self.d_min = d_min
            self.d_max = d_max

            # ── Depth bin values ─────────────────────────
            # D evenly spaced depth values: 2m → 58m
            # Shape: (D,)
            self.register_buffer(
                'depth_values',
                torch.linspace(d_min, d_max, depth_bins)
            )

            # ── BEV grid resolution ───────────────────────
            # How many metres each BEV pixel represents
            self.bev_res_x = (
                (X_RANGE[1] - X_RANGE[0]) / bev_w
            )  # 0.4 m/pixel
            self.bev_res_y = (
                (Y_RANGE[1] - Y_RANGE[0]) / bev_h
            )  # 0.4 m/pixel

            # ── Depth prediction network ──────────────────
            # Input: image features (C channels)
            # Output: D depth logits + C context features
            # Paper: "predict grid-wise depth distribution
            #         on 2D features" (Section 3.1.2)
            self.depth_net = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels,
                          depth_bins + in_channels,
                          kernel_size=1)
            )

            # ── Channel reducer ───────────────────────────
            # After outer product: D×C → C
            self.reduce = nn.Sequential(
                nn.Conv2d(depth_bins * in_channels,
                          in_channels,
                          kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

            # ── BEV feature aggregation ───────────────────
            # Combines features from all cameras on BEV grid
            self.bev_aggregator = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

            logger.info(
                f"LSSViewTransformer initialized | "
                f"D={depth_bins} bins "
                f"[{d_min}m → {d_max}m] | "
                f"BEV={bev_h}×{bev_w} | "
                f"res={self.bev_res_x:.1f}m/px"
            )

        except Exception as e:
            raise BEVException(
                "Failed to init LSSViewTransformer", e
            ) from e

    def forward(self,
                feats:      torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            feats:      (B, N, C, fH, fW)
            intrinsics: (B, N, 3, 3)
            extrinsics: (B, N, 4, 4)

        Returns:
            bev: (B, C, bev_h, bev_w)
        """
        try:
            B, N, C, fH, fW = feats.shape
            device = feats.device

            # ── LIFT ─────────────────────────────────────
            # Process all cameras together
            x = feats.view(B * N, C, fH, fW)

            # Predict depth distribution + context
            out     = self.depth_net(x)
            # Eq 6: D*(û,v̂) = softmax(depth_logits)
            depth   = out[:, :self.D].softmax(dim=1)
            # (B*N, D, fH, fW)
            context = out[:, self.D:]
            # (B*N, C, fH, fW)

            # Outer product: depth × context → 3D features
            # Eq 6: F3D = F*2D ⊗ D*
            # depth:   (B*N, D, 1, fH, fW)
            # context: (B*N, 1, C, fH, fW)
            lifted = (
                depth.unsqueeze(2) *
                context.unsqueeze(1)
            )  # (B*N, D, C, fH, fW)

            # Flatten D×C → single channel dim
            lifted = lifted.view(
                B * N, self.D * self.C, fH, fW
            )

            # Reduce D*C → C channels
            lifted = self.reduce(lifted)
            # (B*N, C, fH, fW)

            # ── SPLAT ─────────────────────────────────────
            # Project 3D points onto BEV grid using K and E
            bev = self._splat_to_bev(
                lifted, intrinsics, extrinsics,
                B, N, fH, fW, device
            )

            # Final aggregation
            bev = self.bev_aggregator(bev)

            logger.info(
                f"LSS forward complete | "
                f"bev shape: {tuple(bev.shape)}"
            )

            return bev  # (B, C, bev_h, bev_w)

        except Exception as e:
            raise BEVException(
                "LSSViewTransformer forward failed", e
            ) from e

    def _splat_to_bev(self,
                      lifted:     torch.Tensor,
                      intrinsics: torch.Tensor,
                      extrinsics: torch.Tensor,
                      B: int, N: int,
                      fH: int, fW: int,
                      device: torch.device
                      ) -> torch.Tensor:
        """
        Project lifted 3D features onto BEV grid.

        Uses real camera geometry:
            P_cam = K⁻¹ × [u, v, 1] × depth
            P_ego = R × P_cam + t

        Then maps P_ego (X,Y) → BEV grid (col, row)

        Returns:
            bev: (B, C, bev_h, bev_w)
        """
        # ── Build BEV accumulator ─────────────────────────
        bev_feat  = torch.zeros(
            B, self.C, self.bev_h, self.bev_w,
            device=device
        )
        bev_count = torch.zeros(
            B, 1, self.bev_h, self.bev_w,
            device=device
        )

        # Feature scale relative to original image
        scale_w = fW / IMG_W
        scale_h = fH / IMG_H

        for b in range(B):
            for n in range(N):
                cam_feat = lifted[b * N + n]
                # (C, fH, fW)

                K = intrinsics[b, n]   # (3, 3)
                E = extrinsics[b, n]   # (4, 4)

                # ── Build pixel grid ──────────────────────
                # All pixel coordinates in feature map
                us = torch.linspace(
                    0, fW - 1, fW, device=device
                )
                vs = torch.linspace(
                    0, fH - 1, fH, device=device
                )
                vv, uu = torch.meshgrid(vs, us, indexing='ij')
                # Both: (fH, fW)

                # Scale back to original image coords
                us_orig = uu / scale_w
                vs_orig = vv / scale_h

                # ── For each depth bin ────────────────────
                for di, depth_val in enumerate(
                    self.depth_values
                ):
                    d = depth_val.item()

                    # Unproject pixel → camera frame
                    # P_cam = K⁻¹ × [u,v,1] × depth
                    fx = K[0, 0].item()
                    fy = K[1, 1].item()
                    cx = K[0, 2].item()
                    cy = K[1, 2].item()

                    X_cam = (us_orig - cx) / fx * d
                    Y_cam = (vs_orig - cy) / fy * d
                    Z_cam = torch.full_like(X_cam, d)
                    # All: (fH, fW)

                    # Stack into homogeneous coords (4, fH*fW)
                    ones = torch.ones_like(X_cam)
                    P_cam = torch.stack(
                        [X_cam, Y_cam, Z_cam, ones], dim=0
                    ).view(4, -1)  # (4, fH*fW)

                    # Transform to ego frame
                    # P_ego = E × P_cam
                    P_ego = (E @ P_cam)[:3]
                    # (3, fH*fW): X_ego, Y_ego, Z_ego

                    # ── Map to BEV grid ───────────────────
                    X_ego = P_ego[0]  # (fH*fW)
                    Y_ego = P_ego[1]  # (fH*fW)

                    # Convert real-world metres → grid index
                    col = (
                        (X_ego - X_RANGE[0]) / self.bev_res_x
                    ).long()
                    row = (
                        (Y_ego - Y_RANGE[0]) / self.bev_res_y
                    ).long()

                    # Filter: only keep points inside BEV
                    mask = (
                        (col >= 0) & (col < self.bev_w) &
                        (row >= 0) & (row < self.bev_h)
                    )  # (fH*fW)

                    if mask.sum() == 0:
                        continue

                    col_valid = col[mask]
                    row_valid = row[mask]

                    # Flatten spatial dims of feature map
                    feat_flat = cam_feat.view(
                        self.C, -1
                    )[:, mask]
                    # (C, valid_points)

                    # Accumulate features on BEV grid
                    bev_feat[b, :, row_valid, col_valid] += \
                        feat_flat
                    bev_count[b, :, row_valid, col_valid] += 1

        # Average: divide accumulated features by count
        bev_count = bev_count.clamp(min=1)
        bev_feat  = bev_feat / bev_count

        return bev_feat  # (B, C, bev_h, bev_w)