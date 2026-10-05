# Color-Invariant Saree Design Recognition System

An end-to-end PyTorch metric-learning framework for color-invariant saree design retrieval, verification, and identification.

Developed for **DeepLure AI Assessment**.

---

## 📌 Problem Overview

Given an RGB image of a saree:
1. **Retrieval:** Retrieve the most similar saree designs from a reference gallery using Cosine Similarity.
2. **Verification:** Given two saree images, determine whether they contain the *same* design, even when their color palettes differ.
3. **Color Invariance:** The same saree design should match across different colorways, while different designs must remain distant in embedding space even if they share similar colors.

---

## 🏗️ Model Architecture & Methodology

```
Input Image (224x224 RGB)
       │
       ▼
ResNet-50 Backbone (Pretrained ImageNet)
       │ (2048-D features)
       ▼
Projection Head (2048 → 512 → 128)
  - Linear(2048, 512)
  - BatchNorm1d(512)
  - ReLU()
  - Linear(512, 128)
       │
       ▼
L2 Normalization (Unit Hyper-sphere: ||z||₂ = 1)
       │
       ▼
128-Dimensional Normalized Embedding Space
       │
       ▼
Cosine Similarity (Dot Product)
```

### 🎯 Loss Function: Supervised Contrastive Loss (SupCon)
Rather than relying on triplet loss with heuristic semi-hard mining, we utilize **Supervised Contrastive Loss**:
- Simultaneously pulls all positive colorways of a design ID together in embedding space.
- Pushes different design motifs apart.
- Operates on normalized 128-D vectors with temperature parameter $\tau=0.07$.

### 🎨 Color-Invariant Data Augmentations
- `ColorJitter` (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
- `RandomGrayscale` (p=0.25): Decouples motif feature extraction from chromatic cues.
- `RandomResizedCrop` (scale 0.6–1.0) & `RandomRotation` (±15°): Scale and orientation invariance while preserving weave structure.

---

## 📊 Empirical Results

Tested on an **unseen Test split** (zero design leakage from training):

### **1. Identification & Retrieval**
- **Top-1 Accuracy:** `100.00%`
- **Top-5 Accuracy:** `100.00%`

### **2. Pairwise Verification Protocol**
- **Optimal Verification Threshold ($\tau^*$):** `0.760`
- **Accuracy:** `99.98%`
- **Precision:** `98.53%`
- **Recall:** `100.00%`
- **F1 Score:** `99.26%`
- **ROC-AUC:** `1.0000`

```
Confusion Matrix (12,328 Test Pairs):
                      Predicted Different (0)    Predicted Same (1)
Actual Different (0)          12,192 (TN)               2 (FP)
Actual Same (1)                    0 (FN)             134 (TP)
```

### **3. Efficiency Metrics**
- **Total Parameters:** `24,623,808` (~24.6M)
- **Embedding Dimension:** `128`
- **Checkpoint File Size:** `94.24 MB`
- **Inference Latency:** `~65 ms` / image (CPU) / `~4.2 ms` / image (GPU)

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/kondraguntayugandhar/DeepLure_AIE_case_colour.git
cd DeepLure_AIE_case_colour

pip install torch torchvision scikit-learn matplotlib pillow numpy tqdm
```

### 2. Dataset Setup
Place your dataset under `./archive` containing craft/pattern subdirectories (`Banarasi`, `Bandhani`, `Ikat`, `Pichwai`).

### 3. Run End-to-End Pipeline
```bash
python train_and_evaluate.py
```

---

## 📁 Repository Structure

```
├── train_and_evaluate.py   # Complete PyTorch training, retrieval & verification pipeline
├── splits_metadata.json    # Deterministic split mappings
├── README.md               # Documentation and benchmark results
└── .gitignore              # Ignores large datasets and checkpoints
```

---

## 📝 License & Confidentiality Notice
The proprietary DeepLure corpus is confidential and not included in this repository.
