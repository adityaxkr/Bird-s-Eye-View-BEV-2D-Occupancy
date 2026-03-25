import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import torchvision.transforms.functional as TF

from config import config as cfg
from models.bev_model import BEVOccupancyModel
from data.nuscenes_loader import get_dataloaders
from utils.metrics import compute_metrics

app = FastAPI(title="BEV Occupancy Hackathon API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Load Model
model = BEVOccupancyModel(pretrained=False).to(device)
ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "best_iou_model.pth")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
model.eval()

# 2. Load Validation Data
_, _, val_ds, full_dataset = get_dataloaders(dataroot=cfg.DATAROOT)

# 3. Load Fixed Calibration for Uploads
try:
    FIXED_K = np.load(os.path.join(ROOT_DIR, "fixed_K.npy"))
    FIXED_E = np.load(os.path.join(ROOT_DIR, "fixed_E.npy"))
except:
    print("Warning: fixed_K.npy or fixed_E.npy not found.")
    FIXED_K, FIXED_E = None, None

FEATURED_INDICES = [80, 103, 82, 104, 116]

def tensor_to_b64(t):
    img_np = t.cpu().numpy()
    mean, std = np.array(cfg.IMG_MEAN).reshape(3,1,1), np.array(cfg.IMG_STD).reshape(3,1,1)
    img_np = np.clip((img_np * std) + mean, 0, 1) * 255.0
    pil_img = Image.fromarray(img_np.astype(np.uint8).transpose(1, 2, 0))
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/samples")
async def get_samples():
    samples = []
    for idx in val_ds.indices:
        sample_meta = full_dataset.nusc.sample[idx]
        scene = full_dataset.nusc.get('scene', sample_meta['scene_token'])
        samples.append({
            "index": idx,
            "scene_name": scene['name'],
            "description": scene['description'],
            "featured": idx in FEATURED_INDICES
        })
    return samples

@app.get("/api/sample-preview/{index}")
async def get_sample_preview(index: int):
    data = full_dataset[index]
    return {"cam_images": [tensor_to_b64(img) for img in data['imgs']]}

@app.post("/api/predict/sample/{index}")
async def predict_sample(index: int):
    data = full_dataset[index]
    imgs, K, E = data['imgs'].unsqueeze(0).to(device), data['intrinsics'].unsqueeze(0).to(device), data['extrinsics'].unsqueeze(0).to(device)
    gt = data['occ_gt']

    with torch.no_grad():
        occ_logits, _ = model(imgs, K, E)
    
    pred_prob = torch.sigmoid(occ_logits).squeeze().cpu().numpy()
    gt_np = gt.numpy()
    metrics = compute_metrics(occ_logits, gt.unsqueeze(0).to(device), threshold=cfg.THRESHOLD)
    
    pred_bin = (pred_prob > cfg.THRESHOLD).astype(float)
    tp = (pred_bin * gt_np).sum()
    fp = (pred_bin * (1 - gt_np)).sum()
    fn = ((1 - pred_bin) * gt_np).sum()
    
    return {
        "pred_grid": pred_prob.tolist(),
        "gt_grid": gt_np.tolist(),
        "metrics": {
            "iou": metrics['occ_iou'], "dwe": metrics['dwe'],
            "precision": float(tp / (tp + fp + 1e-6)), "recall": float(tp / (tp + fn + 1e-6))
        }
    }

@app.post("/api/predict/upload")
async def predict_upload(
    cam_front: UploadFile = File(...), cam_front_left: UploadFile = File(...), cam_front_right: UploadFile = File(...),
    cam_back: UploadFile = File(...), cam_back_left: UploadFile = File(...), cam_back_right: UploadFile = File(...)
):
    files = [cam_front, cam_front_left, cam_front_right, cam_back, cam_back_left, cam_back_right]
    img_tensors = []
    mean, std = torch.tensor(cfg.IMG_MEAN).view(3, 1, 1), torch.tensor(cfg.IMG_STD).view(3, 1, 1)
    
    for f in files:
        pil_img = Image.open(io.BytesIO(await f.read())).convert("RGB").resize((cfg.IMG_W, cfg.IMG_H))
        img_tensors.append((TF.to_tensor(pil_img) - mean) / std)
        
    imgs_batch = torch.stack(img_tensors).unsqueeze(0).to(device)
    K_batch, E_batch = torch.tensor(FIXED_K).unsqueeze(0).to(device), torch.tensor(FIXED_E).unsqueeze(0).to(device)
    
    with torch.no_grad():
        occ_logits, _ = model(imgs_batch, K_batch, E_batch)
        
    return {"pred_grid": torch.sigmoid(occ_logits).squeeze().cpu().numpy().tolist()}
