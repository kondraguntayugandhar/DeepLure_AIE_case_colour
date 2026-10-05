# ResNet-50 Evaluation Metrics Report (2-Class Dataset: Normal & Handloom Sarees)

## Dataset Structure (`./dataset`)
- **Top-Level Categories (2 Classes):** `normal_sarees` and `handloom_sarees`
- **Splits:** `train/`, `valid/`, `test/`
- **Total Images:** 1,468 images across 610 unique design identities
- **Train Split:** 1,027 images (427 designs)
- **Valid Split:** 215 images (91 designs)
- **Test Split:** 226 images (92 designs)

---

## 1. Identification & Retrieval Metrics (Unseen Test Set)
- **Top-1 Accuracy:** 100.00%
- **Top-5 Accuracy:** 100.00%

---

## 2. Pairwise Verification Protocol Metrics
- **Optimal Verification Threshold ($\tau^*$):** `0.760`
- **Total Test Pairs Evaluated:** 12,328
- **Accuracy:** `99.98%`
- **Precision:** `98.53%`
- **Recall:** `100.00%`
- **F1 Score:** `99.26%`

### Confusion Matrix (Standard Binary Layout)

| | **Predicted Positive (Same Design)** | **Predicted Negative (Different Design)** |
| :--- | :--- | :--- |
| **Actual Positive (Same Design)** | **True Positive (TP): 134** | **False Negative (FN): 0** |
| **Actual Negative (Different Design)** | **False Positive (FP): 2** | **True Negative (TN): 12,192** |

---

## 3. Computational Efficiency & Model Footprint
- **Total Parameters:** `24,623,808` (~24.6 Million)
- **Trainable Parameters:** `24,623,808`
- **Inference Latency (CPU):** `65.94 ms` / image
- **Inference Latency (GPU T4):** `~4.2 ms` / image
- **Model Checkpoint Size:** `94.24 MB`
