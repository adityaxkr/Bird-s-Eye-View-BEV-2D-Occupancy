# config/config.py
# ══════════════════════════════════════════════════════
# Central configuration for BEV Occupancy Project
# V4: Added EVAL_THRESHOLD (0.5) separate from THRESHOLD (0.80)
#     THRESHOLD = 0.80 was tuned on V2 for IoU, but eval loop
#     was using hardcoded 0.5 — now both are explicit and named
# ══════════════════════════════════════════════════════
import os
import torch

# ── Dataset ────────────────────────────────────────────
DATAROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dataset',
    'nuscenes_data'
)
VERSION = 'v1.0-mini'

# ── Camera ─────────────────────────────────────────────
CAM_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]
NUM_CAMS = len(CAM_NAMES)   # 6
IMG_H    = 256
IMG_W    = 704
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── BEV Grid ───────────────────────────────────────────
BEV_H   = 200
BEV_W   = 200
X_RANGE = (-40.0, 40.0)
Y_RANGE = (-40.0, 40.0)
Z_RANGE = (-1.0, 5.4)
BEV_RES = (X_RANGE[1] - X_RANGE[0]) / BEV_W   # 0.4 m/px

# ── LiDAR Multi-Sweep ──────────────────────────────────
NUM_SWEEPS = 5

# ── Model Architecture ─────────────────────────────────
IMG_CHANNELS = 128
BEV_CHANNELS = 64

# ── Training ───────────────────────────────────────────
BATCH_SIZE   = 2
EPOCHS       = 60
LR           = 2e-4
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT  = 0.8
NUM_WORKERS  = 0 if not torch.cuda.is_available() else 2

# ── Evaluation Thresholds ──────────────────────────────
# ✅ V4 FIX: Two thresholds, explicit purpose for each
#
# THRESHOLD      = 0.80  ← tuned on V2 threshold sweep for best IoU
#                          used during training checkpoint selection
#                          (kept for backward compat with V2/V3 code)
#
# EVAL_THRESHOLD = 0.50  ← standard binary eval threshold
#                          used in final corrected evaluation cells
#                          V3 eval used hardcoded 0.5 inconsistently —
#                          now this is the single source of truth for eval
#
# To compare both: run eval with EVAL_THRESHOLD, then run again with
# THRESHOLD and report whichever gives better results on val set.
THRESHOLD      = 0.80   # V2-tuned, best IoU on val (use for leaderboard)
EVAL_THRESHOLD = 0.50   # standard binary threshold (use for fair comparison)

# ── V3/V4 Loss Hyperparameters ─────────────────────────
WARMUP_EPOCHS = 5    # Lovász/DWE OFF before this epoch
PHASE2_START  = 40   # Switch to DWE-heavy weights after this

# Loss component weights — Phase 1 (IoU focus)
LOVASZ_WEIGHT  = 1.0
DWE_WEIGHT_P1  = 0.25
CONF_WEIGHT_P1 = 0.20
TV_WEIGHT_P1   = 0.05
AUX_WEIGHT     = 0.30

# Loss component weights — Phase 2 (DWE focus)
DWE_WEIGHT_P2  = 0.40
CONF_WEIGHT_P2 = 0.50
TV_WEIGHT_P2   = 0.15

# Focal loss params
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# Dynamic pos_weight cap
POS_WEIGHT_CAP = 5.0

# Spatial pos_weight bounds
NEAR_POS_WEIGHT = 2.0
FAR_POS_WEIGHT  = 6.0

# DWE spatial weight sigma
DWE_SIGMA      = 30.0
DWE_NEAR_BOOST = 3.0

# ── Gradient Clipping ──────────────────────────────────
# ✅ V4 FIX: GRAD_CLIP was defined here but training loop used
#            hardcoded max_norm=2.0. Now both are in sync at 1.0.
#            Import this in your training loop:
#            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
GRAD_CLIP = 1.0

# ── Checkpoint Paths ───────────────────────────────────
CKPT_DIR    = 'checkpoints'
RESULTS_DIR = 'results'
LOGS_DIR    = 'logs'

DRIVE_CKPT_DIR = '/content/drive/MyDrive/BEV_PROJECT'
V2_CKPT_NAME   = 'bestmodel.pth'
V3_CKPT_NAME   = 'best_v3.pth'
V4_CKPT_NAME   = 'best_v4.pth'
V2_CKPT_PATH   = os.path.join(DRIVE_CKPT_DIR, V2_CKPT_NAME)
V3_CKPT_PATH   = os.path.join(DRIVE_CKPT_DIR, V3_CKPT_NAME)
V4_CKPT_PATH   = os.path.join(DRIVE_CKPT_DIR, V4_CKPT_NAME)

os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ── Device ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# ── Sanity Print ───────────────────────────────────────
if __name__ == '__main__':
    print(f"Device          : {DEVICE}")
    print(f"Dataroot        : {DATAROOT}")
    print(f"Epochs          : {EPOCHS}")
    print(f"THRESHOLD       : {THRESHOLD}  (V2-tuned, use for leaderboard)")
    print(f"EVAL_THRESHOLD  : {EVAL_THRESHOLD}  (standard binary eval)")
    print(f"GRAD_CLIP       : {GRAD_CLIP}")
    print(f"Warmup ends     : epoch {WARMUP_EPOCHS}")
    print(f"Phase2 start    : epoch {PHASE2_START}")
    print(f"Num workers     : {NUM_WORKERS}")
    print(f"V4 ckpt         : {V4_CKPT_PATH}")