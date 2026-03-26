# 🌟 BEV-NET - Complete Pipeline Documentation

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![nuScenes](https://img.shields.io/badge/Dataset-nuScenes%20mini-6c63ff?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/MAHE%20Mobility%20Challenge-2026-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Project%20Ready-success?style=for-the-badge)

> **"We take six cameras from a self-driving car and predict, in real time, which parts of the ground around the vehicle are occupied — with a custom loss function that makes the model prioritise accuracy exactly where it matters most for safety."**

📍 **Project Type**: End-to-end BEV occupancy prediction system  
🔗 **GitHub Repository**: [https://github.com/nirajj12/Bird-s-Eye-View-BEV-2D-Occupancy](https://github.com/nirajj12/Bird-s-Eye-View-BEV-2D-Occupancy)  
📊 **Dataset Used**: **nuScenes mini**  
🧠 **Core Goal**: Predict a 2D Bird's-Eye View occupancy map from 6 surround-view cameras  
🎯 **Best Validation IoU**: **0.3649**  
📉 **DWE**: **0.1137** (51% better than baseline)  
🏁 **Built for**: MAHE Mobility Challenge – Centre of Excellence in Autonomous Mobility, MIT Bengaluru

---

## 1. Project Overview

BEV-NET is an end-to-end autonomous-driving perception project that converts six surround-view RGB images into a top-down occupancy map. It combines multi-camera feature extraction, geometry-aware BEV projection, BEV decoding, occupancy prediction, evaluation metrics, and an interactive FastAPI demo interface.

The system is built around a custom **BEVOccupancyModel** with four main components: **ImageBackbone**, **BEVFormerLite**, **BEVDecoder**, and **OccupancyHead**. It outputs a `200 x 200` BEV occupancy grid and supports both dataset-based inference and fixed-calibration custom uploads.

---

## 2. Model Architecture — The Engineering Journey

This is not just a model. It is the result of three distinct attempts, each failing in a specific way that pointed to the next improvement. Here is exactly what we tried, what went wrong, and what we built instead.

---

### Attempt 1 — LSS-Style Lift-Splat Transformer

Our first approach was a **Lift-Splat-Shoot (LSS)** style pipeline — lift each pixel into 3D using predicted depth, splat features into BEV voxels, then apply a transformer over the BEV grid.

Simple, fast, and well-understood in the literature.

**What happened:**

Training appeared to progress normally for the first few epochs — then **IoU plateaued at epoch 7 and never improved further**.

**Why it failed:**

- LSS requires a depth prediction module — another learned component needing supervision not available in nuScenes mini
- With only 404 samples, a depth network never converges; it outputs near-uniform depth distributions
- Uncertain depth spreads feature mass across voxels, producing blurry BEV representations
- The transformer was learning on top of geometrically noisy, incorrect features
- IoU was stuck around **0.15–0.20** regardless of training duration

LSS works at scale with tens of thousands of samples. For 404, it is the wrong tool.

---

### Attempt 2 — Geometry-Aware BEV Projection (BEVFormer-Style)

> "Instead of predicting depth and then projecting, what if we use **known** camera geometry directly?"

We know the intrinsic matrix **K** and extrinsic matrix **E** for every camera exactly. There is no need to *learn* where things are in 3D — we just need to sample features at geometrically correct locations.

**The shift in thinking:**

| LSS Approach | Geometry-Aware Projection |
|---|---|
| Image → predict depth → lift to 3D → splat to BEV | Known K, E → define BEV grid points → project back to image → sample features |

For every BEV cell at position `(x, y, z)` in ego coordinates:
```
F_BEV(x,y,z) = sample(F_2D, project(P_ego → camera → image))
```

We ask: *"For this BEV cell, which pixel in which camera should I look at?"* — and we know the **exact answer** from K and E. No learned depth. No additional supervision needed.

**Result:** IoU jumped to **0.30**, training was stable, no plateau.

**But a new problem emerged.** DWE was **0.23** — the model was making too many errors near the ego vehicle, the most safety-critical zone.

---

### Attempt 3 — Spatial Geometry + DWE-Aware Training

**What is DWE and why does it matter?**
> DWE weights errors by `1/distance` from ego. A false positive 2 metres away hurts **10× more** than one 20 metres away. A ghost obstacle near you causes an emergency brake; one far away barely matters.

We diagnosed three specific problems in the geometry-aware projection model:

| Problem | Root Cause |
|---|---|
| Blurry near-ego predictions | Sampling only at `Z=0` missed bumper and roof-height features |
| Loss / metric mismatch | Training minimised uniform error; evaluation penalised near-ego errors heavily |
| Soft probabilities near ego | Model outputting `0.6–0.7` instead of decisive `0.05` or `0.95` |

**Our solutions — each maps directly to a diagnosed problem:**

**Solution 1 — Multi-Height Spatial Geometry Sampling**
```python
Z_HEIGHTS = [0.0, 0.5, 1.5]  # ground plane, bumper height, roof height
```
A car is not just on the ground. We sample at three heights and mean-pool — capturing the full vertical extent of objects in the scene.

**Solution 2 — DWE-Aligned Loss Function**
```python
L_dwe = (dwe_map * abs(prob - GT)).mean()
# dwe_map = 1/dist — mathematically identical to the evaluation metric
```
We made the training loss **mathematically identical** to the evaluation metric. The model now explicitly optimises what it is being judged on.

**Solution 3 — Phased Training Curriculum**
```
Epochs  1– 5:  Warmup   → focal + dice only        (learn basic occupancy shapes)
Epochs  6–40:  Phase 1  → + DWE + confidence + TV  (learn spatial priority)
Epochs 41–60:  Phase 2  → amplified DWE weights     (refine near-ego accuracy)
```
Adding all losses from epoch 1 caused collapse. The curriculum gives the model a stable foundation before applying spatial pressure.

**Result:** IoU **0.3649**, DWE **0.1137** — 21% IoU gain and 51% DWE reduction over the geometry projection baseline.

---

### Final Architecture Components

- **ImageBackbone** — Shared CNN backbone applied to each of the 6 camera views
- **BEVFormerLite** — Geometry-aware view transformer, samples at Z = [0.0, 0.5, 1.5]
- **BEVDecoder** — 2D BEV refinement that upsamples and smooths the BEV feature map
- **OccupancyHead** — Produces main occupancy logits and auxiliary logits

### Inputs and Outputs

- **Inputs**: Images `B×6×3×H×W` · Intrinsics `B×6×3×3` · Extrinsics `B×6×4×4`
- **Outputs**: Main logits `B×1×200×200` · Auxiliary logits `B×1×200×200` · Binary BEV map

### System Overview Diagram

![BEV-NET architecture diagram](assets/readme/bevnet-architecture-diagram.png)

```mermaid
flowchart TB
    A[6 Camera Inputs] --> B[Preprocessing]
    B --> C[ImageBackbone]
    C --> D[BEVFormerLite]
    D --> E[BEVDecoder]
    E --> F[OccupancyHead]
    F --> G[Occupancy Logits]
    G --> H[Metrics + Visualization]
    H --> I[FastAPI Demo UI]
```

### Training Flow Diagram

```mermaid
flowchart LR
    A[Warmup] --> B[Phase 1]
    B --> C[Phase 2]
    C --> D[Validation]
    D --> E[Best Checkpoint]
```

**Code Modules**
- `models/backbone.py` · `models/bev_former_lite.py` · `models/bev_decoder.py` · `models/bev_model.py`

---

## 3. Dataset Used

This project uses the **nuScenes mini** dataset for training and evaluation.

📂 **nuScenes mini used in this project**:  
[https://drive.google.com/drive/folders/1g5KgxG0p8-MmTiXkNtCpoYSIkdBQprEm](https://drive.google.com/drive/folders/1g5KgxG0p8-MmTiXkNtCpoYSIkdBQprEm)

| Split | Samples |
|---|---|
| Total | 404 |
| Training | 323 |
| Validation | 81 |

Each sample provides:
- 6 synchronized surround-view RGB camera images
- Camera intrinsics and extrinsics for each camera
- BEV occupancy ground-truth built from LiDAR-based occupancy mapping

The BEV grid is `200 × 200` cells at `0.4 m/cell` → **80 m × 80 m** range around the ego vehicle.

**Relevant Code**
- `data/nuscenes_loader.py` – dataset and dataloaders
- `data/preprocess.py` – image resizing/normalization and calibration preprocessing
- `scripts/extracted_fixed_calib.py`, `scripts/sanity_check_geometry.py` – fixed calibration and geometry sanity checks

---

## 4. Setup & Installation Instructions

### 4.1 Clone the Repository

```bash
git clone https://github.com/nirajj12/Bird-s-Eye-View-BEV-2D-Occupancy.git
cd Bird-s-Eye-View-BEV-2D-Occupancy
```

### 4.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Prepare the Dataset

1. Download **nuScenes mini** from the Drive link above or from the official nuScenes website.
2. Place it in the dataset directory configured in your project (see `config/config.py`).
3. Adjust `cfg.DATAROOT` in the config if needed.

---

## 5. How to Run the Code

### 5.1 Training

```bash
python train.py
```

**Training configuration**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Weight decay | `1e-4` |
| Scheduler | Cosine Annealing LR |
| Epochs | 60 |
| Gradient clipping | Enabled |

**Loss components**: Focal · Dice · Auxiliary BCE · DWE · Confidence regularization · TV regularization

### 5.2 Running the FastAPI Demo

```bash
uvicorn main:app --reload
```

Three inference modes available:
- **Dataset browser** — select any of the 81 validation samples
- **Featured scenes** — curated interesting scenarios
- **Custom upload** — upload your own 6 camera images (fixed nuScenes calibration applied)

---

## 6. Example Outputs / Results

### 6.1 Demo Preview

The FastAPI demo includes a cinematic BEV-style interface for scene selection, multi-camera input review, inference execution, and occupancy evaluation.

**Landing screen**

![BEV-NET landing screen](assets/readme/ui-landing.png)

**Dataset browser with six camera feeds**

![BEV-NET dataset browser](assets/readme/ui-dataset-browser.png)

**Occupancy results and live performance metrics**

![BEV-NET results and metrics panel](assets/readme/ui-results-metrics.png)

### 6.2 Results as a Story

> **IoU improved across each iteration: LSS (~0.17, plateau ep.7) → Geometry Projection (0.30) → Spatial Geometry + DWE Loss (0.36, +21%). The most critical improvement was DWE dropping from 0.23 to 0.11 — a 51% reduction in safety-critical near-ego error.**

| | LSS Transformer | Geometry Projection | Spatial Geometry + DWE Loss (Ours) |
|---|---:|---:|---:|
| **Overall IoU** | ~0.17 *(plateau ep.7)* | 0.3011 | **0.3649** |
| **DWE** | — | 0.2323 | **0.1137** |
| **Precision** | — | 0.4369 | **0.4520** |
| **Recall** | — | 0.5218 | **0.6110** |
| **F1 Score** | — | 0.4734 | **0.5146** |
| **Near-ego IoU** | — | — | **0.6278** |
| **Far-field IoU** | — | — | **0.3150** |

> **What DWE = 0.11 means**: Our average spatially-weighted prediction error is 11% — with errors near the vehicle penalised up to 100× more than distant ones.

> **Near-ego IoU = 0.63**: In the safety-critical zone within 12 metres, our model is correct nearly **two-thirds of the time**. That is the number an autonomous driving engineer actually cares about.

### 6.3 Qualitative Outputs

#### Image-to-BEV Pipeline Visualization

![Image-to-BEV transformation pipeline](assets/readme/viz1_pipeline_v3.png)

#### Validation Showcase Summary

![Best, mid, and worst validation showcase](assets/readme/viz5_bev_showcase.png)

#### Best Validation Sample

![Best validation sample](assets/readme/best_1.png)

#### Random Validation Sample

![Random validation sample](assets/readme/random_1.png)

#### Worst Validation Sample

![Worst validation sample](assets/readme/worst_1.png)

---

## 7. Demo Interface (FastAPI Frontend)

**Stack**: FastAPI + HTML + CSS + JavaScript

### Features

- Scene browser for nuScenes mini validation data
- Featured scenario selection
- Upload mode with fixed nuScenes intrinsics/extrinsics
- BEV probability heatmap
- Binary occupancy map with TP / FP / FN color coding
- Hover-based cell inspection (distance from ego, probability, GT vs prediction)
- Threshold slider with live metric updates
- Metrics panel (IoU, DWE, precision, recall, F1)

### Main API Routes

| Route | Description |
|---|---|
| `GET /api/samples` | List available validation scenes |
| `GET /api/sample-preview/{index}` | Return camera previews for a sample |
| `POST /api/predict-sample/{index}` | Run inference on a dataset sample |
| `POST /api/predict-upload` | Run inference on 6 user-uploaded images |

---

## 8. Project Structure

```bash
.
├── app/
│   └── main.py
├── assets/
│   └── readme/
├── checkpoints/
├── config/
│   └── config.py
├── data/
│   ├── nuscenes_loader.py
│   └── preprocess.py
├── dataset/
│   └── nuscenes_data/
├── exception/
│   └── custom_exception.py
├── logger/
│   └── custom_logger.py
├── logs/
├── models/
│   ├── backbone.py
│   ├── bev_decoder.py
│   ├── bev_former_lite.py
│   └── bev_model.py
├── notebooks/
├── results/
├── sanity_output/
├── scripts/
│   ├── extracted_fixed_calib.py
│   ├── find_featured_samples.py
│   └── sanity_check_geometry.py
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── utils/
│   ├── metrics.py
│   └── visualize.py
├── fixed_E.npy
├── fixed_K.npy
├── requirements.txt
├── train.py
└── README.md
```

---

## 9. Technologies Used

| Category | Tools Used |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Dataset | nuScenes mini |
| Geometry | Camera intrinsics, extrinsics, BEV projection |
| Model Stack | ImageBackbone, BEVFormerLite, BEVDecoder, OccupancyHead |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Matplotlib, custom canvas rendering |
| Utilities | NumPy, tqdm |
| Version Control | Git, GitHub |

---

## 10. Future Enhancements

- Add temporal fusion over multiple frames
- Improve far-field occupancy accuracy (currently IoU 0.31)
- Add Docker support for easy deployment
- Integrate experiment tracking dashboards
- Add explainability overlays for camera→BEV feature contributions
- Extend from binary occupancy to richer BEV semantics (drivable area, lanes, object classes)

---

## 11. Acknowledgments

- **Hackathon**: Built for the **MAHE Mobility Challenge 2026** — Centre of Excellence in Autonomous Mobility, Manipal Institute of Technology Bengaluru
- **Partners**: HARMAN · VTS · Department of Electronics & Communication Engineering
- **Dataset**: nuScenes mini (Motional)
- **Frameworks**: PyTorch and FastAPI
- **Focus Area**: Multi-camera BEV occupancy prediction and visualization

---

## Contributors

<table>
  <tr>
    <td align="center" width="50%">
      <h3>🧑‍🎓 Niraj Kumar</h3>
      <p>AI/ML Engineer</p>
      <p>
        <a href="https://github.com/nirajj12">
          <img src="https://img.shields.io/badge/GitHub-nirajj12-181717?style=for-the-badge&logo=github" alt="Niraj Kumar GitHub" />
        </a>
        <br />
        <a href="https://www.linkedin.com/in/niraj-kumar-8255111b8/">
          <img src="https://img.shields.io/badge/LinkedIn-Niraj%20Kumar-0A66C2?style=for-the-badge&logo=linkedin" alt="Niraj Kumar LinkedIn" />
        </a>
      </p>
    </td>
    <td align="center" width="50%">
      <h3>🧑‍🎓 Aditya Kumar</h3>
      <p>AI/ML Engineer</p>
      <p>
        <a href="https://github.com/adityaxkr">
          <img src="https://img.shields.io/badge/GitHub-adityaxkr-181717?style=for-the-badge&logo=github" alt="Aditya Kumar GitHub" />
        </a>
        <br />
        <a href="https://www.linkedin.com/in/adityaxkr/">
          <img src="https://img.shields.io/badge/LinkedIn-Aditya%20Kumar-0A66C2?style=for-the-badge&logo=linkedin" alt="Aditya Kumar LinkedIn" />
        </a>
      </p>
    </td>
  </tr>
</table>

---

⭐ If this project helped you, consider starring the repository!
