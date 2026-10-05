# ResNet-50 Evaluation Metrics Report (5 Epochs Training)

## System Specifications & Categories
- **Epochs Trained:** 5
- **Loss Progression:** Epoch 1: 4.8872 -> Epoch 2: 4.0934 -> Epoch 3: 3.4930 -> Epoch 4: 3.0180 -> Epoch 5: 2.6288
- **Categories (2 Classes):** `normal_sarees` vs `handloom_sarees`
- **Total Images:** 1,391 images across 421 unique design identities
- **Train Split:** 978 images (294 designs)
- **Valid Split:** 206 images (63 designs)
- **Test Split:** 207 images (64 designs)
- **Backbone Architecture:** Pretrained ResNet-50
- **Projection Head:** Linear(2048, 512) -> BatchNorm1d -> ReLU -> Linear(512, 128) -> L2 Normalization
- **Embedding Dimension:** 128 (L2 Normalized)
- **Loss Function:** Supervised Contrastive Loss (SupCon, $\tau=0.07$)

---

## 1. Identification & Retrieval Metrics (Unseen Test Set)
- **Top-1 Accuracy:** 100.00%
- **Top-5 Accuracy:** 100.00%

---

## 2. Confusion Matrix & Verification Metrics (normal_sarees vs handloom_sarees)
- **Optimal Verification Threshold ($\tau^*$):** `0.770`
- **Total Test Pairs Evaluated:** 12,328
- **Accuracy:** `99.98%`
- **Precision:** `98.53%`
- **Recall:** `100.00%`
- **F1 Score:** `99.26%`

### Binary Confusion Matrix Table

| | **Predicted normal_sarees** | **Predicted handloom_sarees** |
| :--- | :--- | :--- |
| **Actual normal_sarees** | **True Positive (TP): 134** | **False Negative (FN): 0** |
| **Actual handloom_sarees** | **False Positive (FP): 2** | **True Negative (TN): 12,192** |

---

## 3. Computational Efficiency & Model Footprint
- **Total Parameters:** `24,623,808` (~24.6 Million)
- **Trainable Parameters:** `24,623,808`
- **Inference Latency (CPU):** `68.80 ms` / image
- **Inference Latency (GPU T4):** `~4.2 ms` / image
- **Model Checkpoint Size:** `94.24 MB`
