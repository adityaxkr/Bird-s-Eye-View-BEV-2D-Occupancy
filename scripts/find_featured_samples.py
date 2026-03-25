import os
import sys
import torch
from tqdm import tqdm

# Ensure the root project directory is in the python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import config as cfg
from models.bev_model import BEVOccupancyModel
from data.nuscenes_loader import get_dataloaders
from utils.metrics import compute_metrics

def find_best_samples():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    model = BEVOccupancyModel(pretrained=False).to(device)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "best_iou_model.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"✅ Model loaded (from Epoch {ckpt['epoch']})")

    # 2. Load Validation Data
    # Make sure your cfg.DATAROOT is pointing to the right place locally!
    _, _, val_ds, full_dataset = get_dataloaders(dataroot=cfg.DATAROOT)
    print(f"✅ Loaded validation set: {len(val_ds)} samples")

    results = []

    # 3. Evaluate all validation samples
    with torch.no_grad():
        for idx in tqdm(val_ds.indices, desc="Evaluating Samples"):
            sample = full_dataset[idx]
            imgs = sample['imgs'].unsqueeze(0).to(device)
            K = sample['intrinsics'].unsqueeze(0).to(device)
            E = sample['extrinsics'].unsqueeze(0).to(device)
            gt = sample['occ_gt'].to(device)

            occ_logits, _ = model(imgs, K, E)
            
            # Compute metrics
            metrics = compute_metrics(occ_logits, gt.unsqueeze(0))
            
            results.append({
                'index': idx,
                'iou': metrics['occ_iou'],
                'dwe': metrics['dwe']
            })

    # 4. Sort by IoU (descending) and extract Top 5
    results.sort(key=lambda x: x['iou'], reverse=True)
    top_5 = results[:5]

    print("\n🏆 TOP 5 FEATURED SAMPLES (Highest IoU) 🏆")
    print("-" * 55)
    featured_indices = []
    for i, res in enumerate(top_5):
        print(f"#{i+1} | Index: {res['index']:<4} | IoU: {res['iou']:.4f} | DWE: {res['dwe']:.4f}")
        featured_indices.append(res['index'])
        
    print("-" * 55)
    print(f"\n👉 Copy this exact line into your app/main.py:")
    print(f"FEATURED_INDICES = {featured_indices}\n")

if __name__ == "__main__":
    find_best_samples()