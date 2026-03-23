# config/config.py
# ══════════════════════════════════════════════════════
# Central configuration for BEV Occupancy Project
# ALL settings live here — import from here everywhere
# Same role as config.yaml in Document Portal
# ══════════════════════════════════════════════════════

import os
import torch


# ── Dataset ────────────────────────────────────────────
DATAROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dataset',
    'nuscenes_data'
)
VERSION  = 'v1.0-mini'

# ── Camera ─────────────────────────────────────────────
# nuScenes has exactly 6 cameras around the car
CAM_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]
NUM_CAMS = len(CAM_NAMES)   # 6

# Resize every camera image to this fixed size
# Original nuScenes: 1600×900 → we resize to save memory
IMG_H = 256    # height in pixels
IMG_W = 704    # width  in pixels

# ImageNet normalization — standard for pretrained ResNet
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── BEV Grid ───────────────────────────────────────────
# The 2D top-down map we want to predict
# Think of it as a 200×200 satellite view centered on the car
BEV_H = 200    # grid height (pixels)
BEV_W = 200    # grid width  (pixels)

# Real-world range around the ego vehicle (in metres)
X_RANGE = (-40.0, 40.0)   # forward ↔ behind = 80m longitudinal
Y_RANGE = (-40.0, 40.0)   # left    ↔ right  = 80m lateral
Z_RANGE = (-1.0,  5.4)    # below ↔ above  = 6.4m tall

# Resolution: how many real-world metres each pixel represents
# 80m / 200px = 0.4 metres per pixel
BEV_RES = (X_RANGE[1] - X_RANGE[0]) / BEV_W   # 0.4 m/px

# ── Model Architecture ─────────────────────────────────
IMG_CHANNELS  = 128    # feature channels from backbone
BEV_CHANNELS  = 64     # channels inside BEV decoder

# ── Training ───────────────────────────────────────────
BATCH_SIZE   = 2       # small — Mac has limited RAM
EPOCHS       = 20
LR           = 2e-4    # learning rate
WEIGHT_DECAY = 1e-4

# Train / validation split
TRAIN_SPLIT  = 0.8     # 80% train, 20% val
NUM_WORKERS  = 0       # 0 = no multiprocessing (safe for Mac)

# ── Device ─────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Returns best available device:
      - CUDA  → Windows friend's NVIDIA GPU
      - MPS   → Your Mac Apple Silicon GPU
      - CPU   → fallback
    """
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

# Create dirs if they don't exist
os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)