# ══════════════════════════════════════════════════════════════════
# V3 FINAL TRAINING — BEV 2D Occupancy
# Fixed: clone guard, DWE resume, dual plateau tracking
# ══════════════════════════════════════════════════════════════════

import os, sys, torch, torch.optim as optim
import numpy as np
from tqdm import tqdm

# ── Mount Drive ───────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Clone only if not already cloned ──────────────────────────────
REPO = '/content/Bird-s-Eye-View-BEV-2D-Occupancy'
if not os.path.exists(REPO):
    os.system(
        'git clone https://github.com/nirajj12/Bird-s-Eye-View-BEV-2D-Occupancy.git '
        '/content/Bird-s-Eye-View-BEV-2D-Occupancy'
    )
    print("✅ Repo cloned")
else:
    os.system(f'cd {REPO} && git pull')   # pull latest changes
    print("✅ Repo pulled latest")

os.chdir(REPO)
sys.path.insert(0, REPO)
os.system('pip install nuscenes-devkit pyquaternion -q')

# ── Config override ───────────────────────────────────────────────
import config.config as cfg
cfg.DATAROOT = '/content/dataset'

from data.nuscenes_loader import get_dataloaders
from models.bev_model     import BEVOccupancyModel
from utils.metrics        import compute_metrics

# ── Device ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {torch.cuda.get_device_name(0)}")

# ── Drive paths ───────────────────────────────────────────────────
DRIVE_SAVE = '/content/drive/MyDrive/BEV_PROJECT_V3'
os.makedirs(DRIVE_SAVE, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────
train_loader, val_loader, val_ds, full_dataset = get_dataloaders(
    dataroot='/content/dataset'
)
print(f"Train  : {len(train_loader.dataset)}")
print(f"Val    : {len(val_ds)}")

# ── Hyperparams ───────────────────────────────────────────────────
TOTAL_EP      = 60
WARMUP_EP     = 5
PATIENCE_IoU  = 12   # stop if smooth IoU flat for 12 epochs
PATIENCE_DWE  = 15   # stop if DWE flat for 15 epochs (longer — DWE improves slower)

# ── Model + Optimizer ─────────────────────────────────────────────
model     = BEVOccupancyModel(pretrained=True).to(device)
optimizer = optim.AdamW(
    model.parameters(), lr=2e-4, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=55, eta_min=1e-6
)

# ── State trackers ────────────────────────────────────────────────
best_iou        = 0.0
best_dwe        = float('inf')
best_smooth_iou = 0.0
best_smooth_dwe = float('inf')
history         = []
start_epoch     = 1
no_improve_iou  = 0
no_improve_dwe  = 0

# ── Auto-resume ───────────────────────────────────────────────────
ckpt_files = sorted([
    f for f in os.listdir(DRIVE_SAVE)
    if f.startswith('epoch_') and f.endswith('.pth')
])

if ckpt_files:
    latest_path = os.path.join(DRIVE_SAVE, ckpt_files[-1])
    print(f"\n🔄 Resuming from : {ckpt_files[-1]}")

    ckpt        = torch.load(latest_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1
    history     = ckpt.get('history', [])

    if history:
        best_iou        = max(h['val_iou'] for h in history)
        best_dwe        = min(h['val_dwe'] for h in history)  # ✅ FIX: was missing
        best_smooth_iou = best_iou
        best_smooth_dwe = best_dwe
    else:
        best_iou = ckpt['val_iou']
        best_dwe = ckpt['val_dwe']

    # Restore scheduler state
    for _ in range(max(0, ckpt['epoch'] - WARMUP_EP)):
        scheduler.step()

    print(f"✅ Start epoch  : {start_epoch}")
    print(f"✅ Best IoU     : {best_iou:.4f}")
    print(f"✅ Best DWE     : {best_dwe:.4f}")
else:
    print("\n🆕 No checkpoint — starting fresh from epoch 1")

print(f"📁 Saving to    : {DRIVE_SAVE}")
print(f"📊 Epochs       : {start_epoch} → {TOTAL_EP}")
print("─" * 85)
print(f"{'Ep':>4} | {'Loss':>7} | {'Focal':>6} | {'Dice':>6} | "
      f"{'DWELoss':>7} | {'Phase':>6} | {'IoU':>7} | {'DWE':>7} | Note")
print("─" * 85)

# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════
for epoch in range(start_epoch, TOTAL_EP + 1):

    # ── Warmup LR schedule ────────────────────────────────────────
    if epoch <= WARMUP_EP:
        for pg in optimizer.param_groups:
            pg['lr'] = 2e-4 * epoch / WARMUP_EP

    # ────────────────────────────────────────────────────────────
    # TRAIN
    # ────────────────────────────────────────────────────────────
    model.train()
    ep_loss, ep_focal, ep_dice, ep_dwe_l, n_b = 0, 0, 0, 0, 0

    pbar = tqdm(
        train_loader,
        desc  = f"Ep {epoch:02d}/{TOTAL_EP} [Train]",
        ncols = 120
    )

    for batch in pbar:
        imgs       = batch['imgs'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        occ_gt     = batch['occ_gt'].to(device)

        occ_logits, aux_logits = model(imgs, intrinsics, extrinsics)

        # ✅ epoch passed → warmup / phase1 / phase2 DWE
        losses = model.compute_loss(
            occ_logits, aux_logits, occ_gt,
            epoch = epoch
        )
        loss = losses['total']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=2.0
        )
        optimizer.step()

        ep_loss  += loss.item()
        ep_focal += losses['focal'].item() if hasattr(losses['focal'], 'item') else losses['focal']
        ep_dice  += losses['dice'].item()  if hasattr(losses['dice'],  'item') else losses['dice']
        ep_dwe_l += losses['dwe'].item()   if hasattr(losses['dwe'],   'item') else losses['dwe']
        n_b      += 1

        pbar.set_postfix({
            'L'  : f"{loss.item():.3f}",
            'F'  : f"{losses['focal']:.3f}",
            'D'  : f"{losses['dice']:.3f}",
            'DWE': f"{losses['dwe']:.4f}",
            'Ph' : losses['phase'][:4],
        })

    avg_loss  = ep_loss  / n_b
    avg_focal = ep_focal / n_b
    avg_dice  = ep_dice  / n_b
    avg_dwe_l = ep_dwe_l / n_b

    if epoch > WARMUP_EP:
        scheduler.step()

    # ────────────────────────────────────────────────────────────
    # VALIDATE — per-sample (accurate DWE)
    # ────────────────────────────────────────────────────────────
    model.eval()
    iou_sum, dwe_sum, v_b = 0, 0, 0

    with torch.no_grad():
        for idx in val_ds.indices:
            s    = full_dataset[idx]
            imgs = s['imgs'].unsqueeze(0).to(device)
            K    = s['intrinsics'].unsqueeze(0).to(device)
            E    = s['extrinsics'].unsqueeze(0).to(device)
            gt   = s['occ_gt'].to(device)

            occ_logits, _ = model(imgs, K, E)
            m        = compute_metrics(occ_logits, gt.unsqueeze(0))
            iou_sum += m['occ_iou']
            dwe_sum += m['dwe']
            v_b     += 1

    avg_iou = iou_sum / v_b
    avg_dwe = dwe_sum / v_b

    history.append({
        'epoch'   : epoch,
        'loss'    : avg_loss,
        'focal'   : avg_focal,
        'dice'    : avg_dice,
        'dwe_loss': avg_dwe_l,
        'val_iou' : avg_iou,
        'val_dwe' : avg_dwe,
        'phase'   : losses['phase']
    })

    # ── Rolling averages (last 3 epochs) ─────────────────────────
    recent_ious = [h['val_iou'] for h in history[-3:]]
    recent_dwes = [h['val_dwe'] for h in history[-3:]]
    smooth_iou  = sum(recent_ious) / len(recent_ious)
    smooth_dwe  = sum(recent_dwes) / len(recent_dwes)

    # ────────────────────────────────────────────────────────────
    # SAVE BEST IoU MODEL
    # ────────────────────────────────────────────────────────────
    note = ""
    if avg_iou > best_iou:
        best_iou = avg_iou
        note    += " ★IoU"
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'val_iou'    : avg_iou,
            'val_dwe'    : avg_dwe,
            'history'    : history
        }, f'{DRIVE_SAVE}/best_iou_model.pth')

    # ── Save best DWE model (main target) ────────────────────────
    if avg_dwe < best_dwe:
        best_dwe = avg_dwe
        note    += " ★DWE"
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'val_iou'    : avg_iou,
            'val_dwe'    : avg_dwe,
            'history'    : history
        }, f'{DRIVE_SAVE}/best_dwe_model.pth')

    # ────────────────────────────────────────────────────────────
    # PLATEAU DETECTION (dual — IoU + DWE)
    # ────────────────────────────────────────────────────────────
    if smooth_iou > best_smooth_iou:
        best_smooth_iou = smooth_iou
        no_improve_iou  = 0
    else:
        no_improve_iou += 1

    if smooth_dwe < best_smooth_dwe:
        best_smooth_dwe = smooth_dwe
        no_improve_dwe  = 0
    else:
        no_improve_dwe += 1

    # ────────────────────────────────────────────────────────────
    # EPOCH LOG (table row)
    # ────────────────────────────────────────────────────────────
    print(f"{epoch:>4} | {avg_loss:>7.4f} | {avg_focal:>6.3f} | "
          f"{avg_dice:>6.3f} | {avg_dwe_l:>7.4f} | "
          f"{losses['phase']:>6} | {avg_iou:>7.4f} | "
          f"{avg_dwe:>7.4f} |{note} "
          f"iou_pat={no_improve_iou}/{PATIENCE_IoU} "
          f"dwe_pat={no_improve_dwe}/{PATIENCE_DWE}")

    # ────────────────────────────────────────────────────────────
    # EPOCH CHECKPOINT (every epoch — safe for Colab crashes)
    # ────────────────────────────────────────────────────────────
    torch.save({
        'epoch'      : epoch,
        'model_state': model.state_dict(),
        'optimizer'  : optimizer.state_dict(),
        'val_iou'    : avg_iou,
        'val_dwe'    : avg_dwe,
        'history'    : history
    }, f'{DRIVE_SAVE}/epoch_{epoch:02d}.pth')

    # ────────────────────────────────────────────────────────────
    # EARLY STOP — both IoU AND DWE must plateau
    # ────────────────────────────────────────────────────────────
    iou_plateau = no_improve_iou >= PATIENCE_IoU
    dwe_plateau = no_improve_dwe >= PATIENCE_DWE

    if iou_plateau and dwe_plateau:
        print(f"\n⏹  EARLY STOP")
        print(f"   IoU flat for {no_improve_iou} epochs "
              f"(patience={PATIENCE_IoU})")
        print(f"   DWE flat for {no_improve_dwe} epochs "
              f"(patience={PATIENCE_DWE})")
        break

    if iou_plateau and not dwe_plateau:
        print(f"   ⚠️  IoU plateau but DWE still improving "
              f"— continuing for DWE")

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print(f"""
╔══════════════════════════════════════════════╗
║          V3 TRAINING COMPLETE               ║
╠══════════════════════════════════════════════╣
║  Best IoU : {best_iou:.4f}  (V2: 0.3011)         ║
║  Best DWE : {best_dwe:.4f}  (V2: 0.2323)         ║
╠══════════════════════════════════════════════╣
║  Δ IoU    : {best_iou - 0.3011:+.4f}                      ║
║  Δ DWE    : {best_dwe - 0.2323:+.4f}  ← main target      ║
╠══════════════════════════════════════════════╣
║  Checkpoints saved to BEV_PROJECT_V3/       ║
║    best_iou_model.pth  ← highest IoU        ║
║    best_dwe_model.pth  ← lowest DWE         ║
║    epoch_XX.pth        ← every epoch        ║
╚══════════════════════════════════════════════╝
""")