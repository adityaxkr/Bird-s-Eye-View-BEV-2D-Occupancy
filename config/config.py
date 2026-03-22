# config/config.py
# ══════════════════════════════════════════════════════
# Central configuration for BEV Occupancy Project
# Updated: ResNet101, 800×800, 50 epochs
# ══════════════════════════════════════════════════════

import os
import torch


# ── Dataset ────────────────────────────────────────────
DATAROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dataset', 'nuscenes_data'
)
# ↑ This auto-resolves to correct path on both machines:
# Mac    : /Users/nirajmac/Downloads/HACKATHON/dataset/nuscenes_data
# Windows: C:\Users\adity\Bird-s-Eye-View-BEV-2D-Occupancy\dataset\nuscenes_data

VERSION  = 'v1.0-mini'

# ── Camera ─────────────────────────────────────────────
CAM_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]
NUM_CAMS = len(CAM_NAMES)

IMG_H = 256
IMG_W = 704

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── BEV Grid ───────────────────────────────────────────
# 10cm per pixel — matches hackathon requirement exactly
BEV_H = 800
BEV_W = 800

X_RANGE = (-40.0, 40.0)
Y_RANGE = (-40.0, 40.0)
Z_RANGE = (-1.0,  5.4)

# 80m / 800px = 0.1m = 10cm per pixel ✓
BEV_RES = (X_RANGE[1] - X_RANGE[0]) / BEV_W

# ── Model ──────────────────────────────────────────────
IMG_CHANNELS = 128
DEPTH_BINS   = 64
BEV_CHANNELS = 64
BACKBONE     = 'resnet101'

# ── Training ───────────────────────────────────────────
BATCH_SIZE   = 1
EPOCHS       = 50
LR           = 2e-4
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT  = 0.8
NUM_WORKERS  = 2

# ── Device ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# ── Paths ──────────────────────────────────────────────
CKPT_DIR    = 'checkpoints'
RESULTS_DIR = 'results'
LOGS_DIR    = 'logs'

os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)