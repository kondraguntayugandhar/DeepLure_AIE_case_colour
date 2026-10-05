# ResNet-50 Evaluation Metrics Report

## System Specifications & Categories
- **Categories (2 Classes):** `normal_sarees` vs `handloom_sarees`
- **Total Images:** 1,468 images across 610 unique design identities
- **Train Split:** 1,027 images (427 designs)
- **Valid Split:** 215 images (91 designs)
- **Test Split:** 226 images (92 designs)
- **Backbone Architecture:** Pretrained ResNet-50
- **Projection Head:** Linear(2048, 512) -> BatchNorm1d -> ReLU -> Linear(512, 128) -> L2 Normalization
- **Embedding Dimension:** 128 (L2 Normalized)
- **Loss Function:** Supervised Contrastive Loss (SupCon, $\tau=0.07$)

---

## 1. Identification & Retrieval Metrics (Unseen Test Set)
- **Top-1 Accuracy:** 76.30%
- **Top-5 Accuracy:** 80.00%

---

## 2. Confusion Matrix & Verification Metrics (normal_sarees vs handloom_sarees)
- **Optimal Verification Threshold ($\tau^*$):** `0.880`
- **Total Test Pairs Evaluated:** 13,535
- **Accuracy:** `99.74%`
- **Precision:** `98.08%`
- **Recall:** `75.56%`
- **F1 Score:** `85.36%`

### Binary Confusion Matrix Table

| | **Predicted normal_sarees** | **Predicted handloom_sarees** |
| :--- | :--- | :--- |
| **Actual normal_sarees** | **True Negative (TN): 13,498** | **False Positive (FP): 2** |
| **Actual handloom_sarees** | **False Negative (FN): 33** | **True Positive (TP): 102** |

---

## 3. Computational Efficiency & Model Footprint
- **Total Parameters:** `24,623,808` (~24.6 Million)
- **Trainable Parameters:** `24,623,808`
- **Inference Latency (CPU):** `68.80 ms` / image
- **Inference Latency (GPU T4):** `~4.2 ms` / image
- **Model Checkpoint Size:** `94.24 MB`
