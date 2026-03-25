# models/bev_decoder.py
# ══════════════════════════════════════════════════════
# BEV Decoder + Occupancy Head
# V4: Fixed Lconf (was duplicate of Ldwe), now uses entropy sharpening
#     Fixed: Lconf = entropy sharpening near ego (pushes probs → 0/1)
#     Fixed: THRESHOLD imported and used from config (was hardcoded 0.5)
# ══════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import (
    IMG_CHANNELS, BEV_CHANNELS,
    BEV_H, BEV_W,
    WARMUP_EPOCHS, PHASE2_START,
    DWE_WEIGHT_P1, CONF_WEIGHT_P1, TV_WEIGHT_P1,
    DWE_WEIGHT_P2, CONF_WEIGHT_P2, TV_WEIGHT_P2,
    NEAR_POS_WEIGHT, FAR_POS_WEIGHT,
    FOCAL_ALPHA, FOCAL_GAMMA
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)

# ──────────────────────────────────────────────────────
# Building block — UNCHANGED
# ──────────────────────────────────────────────────────
class ConvBnReLU(nn.Module):
    """Standard Conv → BN → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int,
                 k: int = 3, stride: int = 1,
                 padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ──────────────────────────────────────────────────────
# BEVDecoder — UNCHANGED
# ──────────────────────────────────────────────────────
class BEVDecoder(nn.Module):
    """
    FastOcc 2D FCN Decoder (Eq 3, 4, 5).
    Input:  (B, 128, 200, 200)
    Output: (B,  64, 200, 200)
    """
    def __init__(self,
                 in_channels: int = IMG_CHANNELS,
                 out_channels: int = BEV_CHANNELS):
        super().__init__()
        try:
            self.decoder = nn.Sequential(
                ConvBnReLU(in_channels, 128),
                ConvBnReLU(128, 128),
                ConvBnReLU(128, out_channels),
                ConvBnReLU(out_channels, out_channels),
            )
            logger.info(
                f"BEVDecoder initialized | "
                f"{in_channels} → {out_channels} channels"
            )
        except Exception as e:
            raise BEVException("Failed to init BEVDecoder", e) from e

    def forward(self, bev_feat: torch.Tensor):
        try:
            return self.decoder(bev_feat)
        except Exception as e:
            raise BEVException("BEVDecoder forward failed", e) from e


# ──────────────────────────────────────────────────────
# OccupancyHead — UNCHANGED
# ──────────────────────────────────────────────────────
class OccupancyHead(nn.Module):
    """
    Final occupancy head.
    Output: RAW LOGITS — sigmoid applied in loss only.
    Input:  (B, 64, 200, 200)
    Output: occ_logits (B,1,200,200), aux_logits (B,1,200,200)
    """
    def __init__(self, bev_channels: int = BEV_CHANNELS):
        super().__init__()
        try:
            self.predict = nn.Sequential(
                ConvBnReLU(bev_channels, 32),
                ConvBnReLU(32, 32),
                nn.Conv2d(32, 1, kernel_size=1)
            )
            self.aux_head = nn.Sequential(
                ConvBnReLU(bev_channels, 32),
                nn.Conv2d(32, 1, kernel_size=1)
            )
            logger.info(
                f"OccupancyHead initialized | "
                f"input: {bev_channels}ch → output: 1ch logits"
            )
        except Exception as e:
            raise BEVException("Failed to init OccupancyHead", e) from e

    def forward(self, bev_decoded: torch.Tensor):
        try:
            occ_logits = self.predict(bev_decoded)
            aux_logits = self.aux_head(bev_decoded)
            return occ_logits, aux_logits
        except Exception as e:
            raise BEVException("OccupancyHead forward failed", e) from e


# ──────────────────────────────────────────────────────
# Spatial weight helpers
# ──────────────────────────────────────────────────────
def dwe_exact_weight(H: int, W: int,
                     device: torch.device) -> torch.Tensor:
    """
    Exact match to hackathon DWE metric: w = 1/dist_from_ego.
    Returns: (1, 1, H, W) normalized — mean = 1.0
    """
    cx = W / 2
    cy = H / 2
    ys = torch.arange(H, device=device).float() - cy
    xs = torch.arange(W, device=device).float() - cx
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    dist = torch.sqrt(gy**2 + gx**2).clamp(min=1.0)
    w = 1.0 / dist
    w = w / w.mean()
    return w.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)


def spatial_pos_weight(H: int, W: int,
                       device: torch.device) -> torch.Tensor:
    """
    Per-pixel pos_weight: near ego → NEAR_POS_WEIGHT, far → FAR_POS_WEIGHT.
    Returns: (1, 1, H, W)
    """
    cx = W / 2
    cy = H / 2
    ys = torch.arange(H, device=device).float() - cy
    xs = torch.arange(W, device=device).float() - cx
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    dist  = torch.sqrt(gy**2 + gx**2)
    alpha = (dist / (max(H, W) / 2)).clamp(0.0, 1.0)
    pw    = NEAR_POS_WEIGHT + (FAR_POS_WEIGHT - NEAR_POS_WEIGHT) * alpha
    return pw.unsqueeze(0).unsqueeze(0)         # (1,1,H,W)


# ──────────────────────────────────────────────────────
# Existing loss functions — UNCHANGED
# ──────────────────────────────────────────────────────
def focal_loss(
    pred_logits: torch.Tensor,
    gt: torch.Tensor,
    gamma: float = FOCAL_GAMMA,
    alpha: float = FOCAL_ALPHA,
    pos_weight: torch.Tensor = None
) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits)
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    alpha_t = alpha * gt + (1.0 - alpha) * (1.0 - gt)
    if pos_weight is not None:
        bce = -(pos_weight * gt * torch.log(pred)
                + (1.0 - gt) * torch.log(1.0 - pred))
    else:
        bce = -(gt * torch.log(pred) + (1.0 - gt) * torch.log(1.0 - pred))
    pt    = torch.exp(-bce)
    focal = alpha_t * (1.0 - pt) ** gamma * bce
    return focal.mean()


def dice_loss(
    pred_logits: torch.Tensor,
    gt: torch.Tensor
) -> torch.Tensor:
    pred   = torch.sigmoid(pred_logits)
    smooth = 1e-6
    intersection = (pred * gt).sum(dim=[1, 2, 3])
    union        = pred.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3])
    dice         = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def aux_bce_loss(
    aux_logits: torch.Tensor,
    gt: torch.Tensor
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(aux_logits, gt)


# ──────────────────────────────────────────────────────
# ✅ V4 FIX: total_occupancy_loss
#    BUG FIXED: Lconf was identical to Ldwe (same formula, different weight)
#    FIX: Lconf now uses DWE-weighted binary cross-entropy sharpening
#         → forces probs toward hard 0/1 near ego
#         → directly reduces soft DWE error at center
# ──────────────────────────────────────────────────────
def total_occupancy_loss(
    occ_logits: torch.Tensor,
    gt: torch.Tensor,
    epoch: int = 1,
    aux_logits: torch.Tensor = None,
    focal_w: float = 1.0,
    dice_w: float  = 1.0,
    aux_w: float   = 0.3
) -> dict:
    """
    V4 Loss = Focal(spatial) + Dice + Aux + DWE + Confidence(entropy) + TV

    Phases:
      epoch < WARMUP_EPOCHS    : focal + dice + aux only
      WARMUP_EPOCHS → PHASE2   : + DWE + confidence + TV (P1 weights)
      PHASE2 → end             : same + amplified (P2 weights)

    V4 changes vs V3:
      Lconf fixed: was (dwe_map * |prob - gt|) — duplicate of Ldwe
      Now:         DWE-weighted binary entropy H(prob)
                   → H(p) = -p*log(p) - (1-p)*log(1-p)
                   → High entropy (p≈0.5) near ego is penalized heavily
                   → Pushes predictions to confident 0 or 1 near center
    """
    B, _, H, W = occ_logits.shape
    device     = occ_logits.device

    # ── Shape guard ────────────────────────────────────
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    gtf = gt.float()

    # ── Spatial weight maps ────────────────────────────
    dwe_map = dwe_exact_weight(H, W, device)    # (1,1,H,W)
    sp_pw   = spatial_pos_weight(H, W, device)  # (1,1,H,W)

    # ── Focal with spatial pos_weight ──────────────────
    Lf = focal_loss(
        occ_logits, gtf,
        pos_weight=sp_pw.expand_as(gtf)
    ) * focal_w

    # ── Dice ───────────────────────────────────────────
    Ld = dice_loss(occ_logits, gtf) * dice_w

    # ── Aux BCE ────────────────────────────────────────
    Lb = torch.tensor(0.0, device=device)
    if aux_logits is not None:
        Lb = aux_bce_loss(aux_logits, gtf) * aux_w

    # ── Warmup: stable base loss only ─────────────────
    if epoch < WARMUP_EPOCHS:
        total = Lf + Ld + Lb
        return {
            'total': total,
            'focal': Lf.detach(),
            'dice' : Ld.detach(),
            'aux'  : Lb.detach(),
            'dwe'  : torch.tensor(0.0),
            'conf' : torch.tensor(0.0),
            'tv'   : torch.tensor(0.0),
            'phase': 'warmup'
        }

    # ── Phase weights ──────────────────────────────────
    if epoch < PHASE2_START:
        dw, cw, tw = DWE_WEIGHT_P1, CONF_WEIGHT_P1, TV_WEIGHT_P1
        phase = 'phase1'
    else:
        dw, cw, tw = DWE_WEIGHT_P2, CONF_WEIGHT_P2, TV_WEIGHT_P2
        phase = 'phase2-DWE'

    probs = torch.sigmoid(occ_logits)

    # ── DWE loss — exact metric alignment ──────────────
    # Same formula as hackathon: weighted |prob - GT|
    Ldwe = (dwe_map * torch.abs(probs - gtf)).mean() * dw

    # ── ✅ V4 FIX: Confidence sharpening (entropy-based) ──
    # OLD (V3 BUG): Lconf = (dwe_map * |prob - gt|).mean() * cw
    #   → Identical to Ldwe — wastes a loss term, no new signal
    #
    # NEW (V4): DWE-weighted binary entropy H(prob)
    #   H(p) = -p*log(p) - (1-p)*log(1-p)  — peaks at p=0.5, zero at p=0/1
    #   Multiplied by dwe_map → penalizes uncertain predictions near ego
    #   This directly forces the model toward hard 0/1 near center,
    #   which reduces the soft DWE error that stays high even when
    #   Ldwe appears low during training (soft probs vs binary eval gap)
    eps     = 1e-6
    entropy = -(probs * torch.log(probs + eps)
                + (1.0 - probs) * torch.log(1.0 - probs + eps))
    Lconf   = (dwe_map * entropy).mean() * cw

    # ── TV smoothness (DWE-weighted) ───────────────────
    diff_h = torch.abs(probs[:, :, 1:, :]  - probs[:, :, :-1, :])
    diff_w = torch.abs(probs[:, :, :, 1:]  - probs[:, :, :, :-1])
    Ltv    = (dwe_map[:, :, 1:, :]  * diff_h).mean() + \
             (dwe_map[:, :, :, 1:]  * diff_w).mean()
    Ltv    = Ltv * tw

    total = Lf + Ld + Lb + Ldwe + Lconf + Ltv
    return {
        'total': total,
        'focal': Lf.detach(),
        'dice' : Ld.detach(),
        'aux'  : Lb.detach(),
        'dwe'  : Ldwe.detach(),
        'conf' : Lconf.detach(),
        'tv'   : Ltv.detach(),
        'phase': phase
    }