#  FloorPlan Generator (CGAN + Pix2PixHD)
> **Generate architectural floor plan images from custom room constraints using a two-stage GAN pipeline.**

---

##  Overview

This project combines **two deep-learning models**:

| Stage | Model | Framework | Purpose |
|--------|--------|------------|----------|
| **Stage 1: Constraint-based Layout Generation** | Conditional GAN (CGAN) | TensorFlow / Keras | Generates rough layout based on room counts |
| **Stage 2: Refinement** | Pix2PixHD | PyTorch | Converts rough layout into realistic architectural floorplan |

The user specifies:
rooms, bedrooms, kitchen, bathrooms, balcony, etc.

The system outputs a **final, high-resolution floorplan**.

---

##  Features

✔ User-controlled generation (based on room count constraints)  
✔ Two deep learning models combined into one automated pipeline  
✔ Supports GPU acceleration (TensorFlow + PyTorch CUDA)  
✔ Lightweight CLI interaction — no UI required for demonstration  
✔ Suitable for research demos, papers, and engineering expos  

---

##  Folder Structure

floorplan_generator/
│── datasets/
│ └── floorplans/
│ └── test_A/ # CGAN writes rough layout here
│
│── pix2pixHD/ # Pix2PixHD repo (PyTorch)
│ └── checkpoints/
│ └── floorGAN_finetune_v3/
│ ├── latest_net_G.pth # Generator weights
│ ├── latest_net_D.pth # Discriminator weights
│
│── training_output/
│ ├── generator_final.keras # SavedModel (CGAN)
│ ├── scaler_data.gz # MinMaxScaler & metadata
│
│── generate_floorplan.py # Runs CGAN only
│── run_floorplan.sh # Runs full pipeline (CGAN + Pix2PixHD)
│── requirements.txt
│── README.md
└── ...

---

##  Setup Instructions

### 1️⃣ Clone Repository (requires Git LFS for large model files)

> ⚠️ Install Git LFS first  
> https://git-lfs.github.com/

```bash
git clone https://github.com/AbhinavRaj24/floorplan_generator.git
cd floorplan_generator
git lfs pull
```
### 2️⃣ Setup Virtual Environment
```
python3 -m venv venv
source venv/bin/activate     # Linux/macOS
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
##  Usage

###  Run Full Pipeline (CGAN → Pix2PixHD)
`/run_floorplan.sh`
The script will ask:
Enter count for 'total_rooms': 8 Enter count for 'Bedroom': 3 Enter count for 'Kitchen': 1 Enter count for 'Bathroom': 2 ...

Then:

CGAN generates rough output → `datasets/floorplans/test_A/...png`

Pix2PixHD refines it → `pix2pixHD/results/.../final.png`

---

## ⚙ Model Details

### CGAN — Conditional Generator (Keras)

Given:

* `z ∼ N(0, 1)` (latent noise)
* `c ∈ ℝⁿ` (user-specified room constraints)

The generator learns mapping:
`G(z, c) → X`
Where `X` is a generated layout image.

**Losses:**

* **Generator**: `BinaryCrossentropy( real_label )`
* **Discriminator**: label smoothing + noise regularization

### Pix2PixHD — Refinement Model (PyTorch)

**Architecture:**

* Global generator + local enhancer networks
* Multi-scale discriminators
* Instance Normalization
* Residual blocks

Refines noisy CGAN image into clearer, high-resolution floorplan.

---

## 🔧 Troubleshooting
| Issue | Solution |
| :--- | :--- |
| File not found: `.keras` | Ensure `git lfs pull` ran properly |
| CGAN generates same output repeatedly | Delete cached noise or retrain |
| Pix2PixHD fails with `UnpicklingError` | Use `Python 3.10` + `PyTorch ≤ 2.1`, NOT `2.6` |
| Images not appearing in results folder | Ensure `dataroot` path is correct in `run_floorplan.sh` |

---
