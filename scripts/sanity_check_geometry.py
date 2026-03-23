# scripts/sanity_check_geometry.py
# ═══════════════════════════════════════════════════════
# RUN THIS BEFORE TRAINING. DO NOT SKIP.
# Validates: projection math, coverage map, GT quality
# ═══════════════════════════════════════════════════════
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ──────────────────────────────────────────────────────

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from data.preprocess import (
    preprocess_image,
    preprocess_intrinsic,
    preprocess_extrinsic,
    build_bev_occupancy
)
from config.config import (
    DATAROOT, VERSION, CAM_NAMES,
    IMG_H, IMG_W, BEV_H, BEV_W,
    X_RANGE, Y_RANGE
)

os.makedirs('sanity_output', exist_ok=True)

nusc  = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
sample = nusc.sample[0]   # use first sample

# ── Check 1: GT occupancy after Z-filter ──────────────────────────────────
lidar_token = sample['data']['LIDAR_TOP']
lidar_data  = nusc.get('sample_data', lidar_token)
lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))

occ_gt = build_bev_occupancy(pc.points, lidar_calib['rotation'], lidar_calib['translation'])
occ_pct = occ_gt.sum().item() / (BEV_H * BEV_W) * 100

print(f"[CHECK 1] GT occupancy: {occ_gt.sum().item():.0f}/{BEV_H*BEV_W} cells ({occ_pct:.1f}%)")
print(f"          ✓ PASS if 2% < {occ_pct:.1f}% < 15%")
print(f"          ✗ FAIL if >20%: Z-filter not working — check lidar→ego transform")

plt.figure(figsize=(6,6))
plt.imshow(occ_gt.numpy(), cmap='hot', origin='lower')
plt.title(f'BEV GT Occupancy ({occ_pct:.1f}% occupied)')
plt.colorbar()
plt.savefig('sanity_output/check1_gt_occupancy.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: sanity_output/check1_gt_occupancy.png")

# ── Check 2: Single-camera projection onto image ───────────────────────────
cam_name   = 'CAM_FRONT'
cam_token  = sample['data'][cam_name]
cam_data   = nusc.get('sample_data', cam_token)
calib      = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
img_path   = os.path.join(nusc.dataroot, cam_data['filename'])

import cv2
img_orig   = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
orig_h, orig_w = img_orig.shape[:2]   # 900, 1600

K_scaled   = preprocess_intrinsic(calib['camera_intrinsic'], orig_w, orig_h)
E_cam2ego  = preprocess_extrinsic(calib['rotation'], calib['translation'])

# Build BEV grid points (ego frame)
x_half = (X_RANGE[1] - X_RANGE[0]) / (2 * BEV_W)
y_half = (Y_RANGE[1] - Y_RANGE[0]) / (2 * BEV_H)
xs = torch.linspace(X_RANGE[0]+x_half, X_RANGE[1]-x_half, BEV_W)
ys = torch.linspace(Y_RANGE[0]+y_half, Y_RANGE[1]-y_half, BEV_H)
gy, gx = torch.meshgrid(ys, xs, indexing='ij')
gz  = torch.zeros_like(gx)
bev_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (N, 3)

# Invert extrinsic: T_ego→cam
E_ego2cam = torch.inverse(E_cam2ego)
R = E_ego2cam[:3, :3]  # (3, 3)
t = E_ego2cam[:3,  3]  # (3,)
K = K_scaled            # (3, 3)

# Project
pts_cam = (R @ bev_pts.T) + t.unsqueeze(-1)  # (3, N)
depth   = pts_cam[2]                          # (N,)
p_img   = K @ pts_cam                         # (3, N)
u = (p_img[0] / (p_img[2] + 1e-6)).numpy()
v = (p_img[1] / (p_img[2] + 1e-6)).numpy()
d = depth.numpy()

# Scale u,v back to original image resolution (K was already scaled)
# K was scaled to 704×256, so u,v are in [0,704]×[0,256]
# Rescale to original image for overlay
u_orig = u * (orig_w / IMG_W)
v_orig = v * (orig_h / IMG_H)

valid = (d > 0.1) & (u_orig >= 0) & (u_orig < orig_w) & (v_orig >= 0) & (v_orig < orig_h)
u_valid = u_orig[valid].astype(int)
v_valid = v_orig[valid].astype(int)

# Draw projection points on image
img_proj = img_orig.copy()
for pu, pv in zip(u_valid[::10], v_valid[::10]):  # every 10th point
    cv2.circle(img_proj, (pu, pv), 2, (0, 255, 0), -1)

plt.figure(figsize=(12, 5))
plt.imshow(img_proj)
plt.title(f'BEV Grid Projected onto {cam_name} ({valid.sum()} valid pts)')
plt.axis('off')
plt.savefig('sanity_output/check2_projection_front.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"\n[CHECK 2] Front camera projection: {valid.sum()} points land on image")
print(f"          ✓ PASS if 5000–15000 valid points")
print(f"          ✗ FAIL if 0: extrinsic inversion wrong")
print(f"          ✗ FAIL if all points cluster in one corner: K scaling wrong")
print(f"  Saved: sanity_output/check2_projection_front.png")

# ── Check 3: Coverage map — count cameras visible per BEV cell ────────────
coverage = torch.zeros(BEV_H * BEV_W)

for cam_name in CAM_NAMES:
    cam_token  = sample['data'][cam_name]
    cam_data   = nusc.get('sample_data', cam_token)
    calib      = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    orig_h_c   = cam_data['height']
    orig_w_c   = cam_data['width']

    K_c = preprocess_intrinsic(calib['camera_intrinsic'], orig_w_c, orig_h_c)
    E_c = torch.inverse(preprocess_extrinsic(calib['rotation'], calib['translation']))
    R_c = E_c[:3, :3]
    t_c = E_c[:3,  3]

    pts_c = (R_c @ bev_pts.T) + t_c.unsqueeze(-1)
    d_c   = pts_c[2]
    pi_c  = K_c @ pts_c
    u_c   = pi_c[0] / (pi_c[2] + 1e-6)
    v_c   = pi_c[1] / (pi_c[2] + 1e-6)

    v_mask = (d_c > 0.1) & (u_c >= 0) & (u_c < IMG_W) & (v_c >= 0) & (v_c < IMG_H)
    coverage += v_mask.float()

coverage_map = coverage.reshape(BEV_H, BEV_W)
print(f"\n[CHECK 3] Coverage map statistics:")
print(f"  Cells with 0 cameras : {(coverage_map == 0).sum().item()}")
print(f"  Cells with 1 camera  : {(coverage_map == 1).sum().item()}")
print(f"  Cells with 2 cameras : {(coverage_map == 2).sum().item()}")
print(f"  Cells with 3+ cameras: {(coverage_map >= 3).sum().item()}")
print(f"  ✓ PASS: coverage >0 for >80% of cells, arc-shaped regions per camera")
print(f"  ✗ FAIL if coverage=0 everywhere: projection math wrong")
print(f"  ✗ FAIL if coverage=6 everywhere: valid mask not filtering OOV")

plt.figure(figsize=(7, 7))
plt.imshow(coverage_map.numpy(), cmap='hot', origin='lower',
           extent=[X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]])
plt.colorbar(label='# cameras visible')
plt.xlabel('X (forward, m)')
plt.ylabel('Y (left, m)')
plt.title('BEV Camera Coverage Map')
plt.savefig('sanity_output/check3_coverage_map.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"  Saved: sanity_output/check3_coverage_map.png")

print("\n═══════════════════════════════════════════")
print("All checks complete. Review the 3 saved images.")
print("DO NOT START TRAINING until all 3 checks pass.")
print("═══════════════════════════════════════════")