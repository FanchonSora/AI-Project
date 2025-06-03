# Galaxy10 Classification Project

A full‑stack, end‑to‑end playground for **classifying galaxy morphologies** in the Galaxy10‑DECals data set.

* **Backend** – two independent PyTorch models (EfficientNet‑B0 and EffNet‑B2), plus a light‑weight ensemble and a shared evaluation pipeline.
* **Frontend** – (optional) React UI to upload an image and receive the predicted class.
* **Data** – single `Galaxy10_DECals.h5` containing \~14 000 64×64 RGB images and labels.

---

## Table of Contents

1. [Project Layout](#project-layout)
2. [Quick Start](#quick-start)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Running Evaluations](#running-evaluations)
6. [REST Inference API (optional)](#rest-inference-api-optional)
7. [Frontend](#frontend)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

---

## Project Layout

```
.
├── backend/
│   ├── backend_old_model.py        # EfficientNet‑B0 (128×128) + h5 dataset wrapper
│   ├── backend_new_model.py        # EfficientNet‑B2 (260→224) – higher capacity
│   ├── combined_backend.py         # Toy ensemble logic
│   ├── evaluate.py                 # Single entry point – evaluates old, new & ensemble
│   ├── galaxy10_b0.pth             # Weights for old model
│   ├── galaxy10_ema.pth            # Weights for new model
│   └── Galaxy10_DECals.h5          # Dataset (≈270 MB)
├── frontend/                       # React app (optional)
│   ├── src/
│   └── ...
├── requirements.txt                # Python deps (Torch, timm, sklearn, h5py …)
└── README.md                       # This file
```

> **Note** – the repository deliberately keeps dataset & checkpoints inside *backend/* for simplicity; feel free to move them and adjust paths / CLI flags.

---

## Quick Start

**1️⃣ Clone & Install**

```bash
# Clone
$ git clone https://github.com/<you>/galaxy10-classification.git
$ cd galaxy10-classification

# Create Python virtualenv (recommended)
$ python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install all runtime dependencies
$ pip install -r requirements.txt
```

**2️⃣ Download the dataset (if missing)**

```bash
$ wget -O backend/Galaxy10_DECals.h5 \
  https://zenodo.org/record/2530925/files/Galaxy10_DECals.h5?download=1
```

*(or copy the file into `backend/` from another location — size ≈ 270 MB)*

**3️⃣ Run evaluation**

```bash
$ cd backend
# Evaluate old (B0), new (B2) and ensemble in one go
$ python evaluate.py                                    # default batch=64, CUDA if available

# Custom paths / batch size
$ python evaluate.py \
    --h5_path ../data/Galaxy10_DECals.h5 \
    --old_ckpt weights/my_old.pth \
    --new_ckpt weights/my_new.pth \
    --batch 32
```

The script prints **Confusion Matrix**, **per‑class metrics**, **Top‑1 accuracy** and **Macro‑F1** for both *validation* and *test* splits.

---

## Dataset

* **Name**: Galaxy10 DECals
* **Source**: [Kaggle](https://www.kaggle.com/datasets/alanmrsp/galaxy10decals) / [Zenodo 2530925](https://doi.org/10.5281/zenodo.2530925)
* **Classes (10)**: Smooth, Features/Disk, Star, Edge‑On Disk, Spiral, Irregular, Lens/Arc, Ring, Disturbed, Merger
* **Pre‑processing**: images are kept at the native 64×64 RGB resolution and wrapped into a single HDF5 file (`Galaxy10_DECals.h5`).

---

## Models

| Model        | Backbone        | Input Size |                                               Normalisation | Checkpoint            |
| ------------ | --------------- | ---------: | ----------------------------------------------------------: | --------------------- |
| **old**      | EfficientNet‑B0 |    128×128 |                                            mean = std = 0.5 | `galaxy10_b0.pth`     |
| **new**      | EfficientNet‑B2 |    260→224 | ImageNet (μ = *0.485 0.456 0.406*, σ = *0.229 0.224 0.225*) | `galaxy10_ema.pth`    |
| **ensemble** | rule‑based      |          — |                                                           — | n/a (uses both above) |

Both models output 10‑class logits; the ensemble simply trusts the *old* predictions when the two models disagree, otherwise averages the confidences.

---

## Running Evaluations

The central script is **`backend/evaluate.py`**. Key CLI flags:

| Flag         | Default              | Description             |
| ------------ | -------------------- | ----------------------- |
| `--h5_path`  | `Galaxy10_DECals.h5` | Path to HDF5 dataset    |
| `--old_ckpt` | `galaxy10_b0.pth`    | Weights for *old* model |
| `--new_ckpt` | `galaxy10_ema.pth`   | Weights for *new* model |
| `--batch`    | `64`                 | Evaluation batch size   |

Example — evaluate *only* the new model:

```bash
python evaluate.py --new_ckpt my_B2.pth --old_ckpt "" --batch 128
```

If you leave a ckpt flag empty (`""`) that model will be skipped.

---

## REST Inference API (optional)

Both **`backend_old_model.py`** and **`backend_new_model.py`** expose a tiny Flask API. Run e.g.:

```bash
$ python backend_old_model.py --host 0.0.0.0 --port 8000
```

**POST** an image (`multipart/form-data`) to `/predict` and receive JSON:

```json
{
  "class_id": 4,
  "class_name": "Spiral",
  "probability": 0.92
}
```

See comments in each file for details and cURL examples.

---

## Frontend
![image](https://github.com/user-attachments/assets/c4c528dd-360a-467e-99ae-3e168c347c50)
![image](https://github.com/user-attachments/assets/2404c320-72bf-4f71-aa5e-559a3784700b)

The *frontend* folder hosts a minimal React + Tailwind app that can:

1. Upload an image, call the REST endpoint, and show the predicted morphology.
2. Render a short description of the class, with example PNGs from the dataset.

```bash
$ cd frontend
$ pnpm install           # or yarn / npm
$ pnpm dev               # Vite/React dev server on :5173
```

Configure the backend URL in `src/config.ts`.

---

## Troubleshooting

| Problem                                                       | Fix                                                                 |
| ------------------------------------------------------------- | ------------------------------------------------------------------- |
| **`OSError: Unable to open file (file signature not found)`** | Wrong `--h5_path` or corrupted download. Re‑download the dataset.   |
| **`RuntimeError: size mismatch for classifier.1.weight`**     | Check that the checkpoint you pass matches the backbone (B0 vs B2). |
| Windows + h5py slowdown                                       | Keep `num_workers=0` in `evaluate.py` or use WSL/Linux.             |

---

## License

Code is released under the **MIT License** (see `LICENSE`). Galaxy10 data is licensed under [CC‑BY‑4.0](https://creativecommons.org/licenses/by/4.0/); please cite *"Galaxy Zoo: Probabilistic Morphologies for Millions of Galaxies"* if you use the dataset.
