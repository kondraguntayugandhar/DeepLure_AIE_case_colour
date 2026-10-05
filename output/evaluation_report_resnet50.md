# ResNet-50 Empirical Evaluation Report (Merged Corpus: Archive + Handlooms)

## System Specifications
- **Combined Dataset:** `archive` (1,468 images) + `handlooms` (165 images) = **1,633 Total Images**
- **Total Unique Design Identities:** 775 designs (542 Train / 116 Val / 117 Test)
- **Backbone Architecture:** Pretrained ResNet-50
- **Projection Head:** Linear(2048, 512) -> BatchNorm1d -> ReLU -> Linear(512, 128) -> L2 Normalization
- **Embedding Dimension:** 128 (L2 Normalized)
- **Loss Function:** Supervised Contrastive Loss (SupCon, $\tau=0.07$)

---

## 1. Identification & Retrieval Metrics (Unseen Test Set)
- **Top-1 Accuracy:** 100.00%
- **Top-5 Accuracy:** 100.00%

---

## 2. Pairwise Verification Protocol Metrics
- **Optimal Verification Threshold ($\tau^*$):** `0.820`
- **Total Test Pairs Evaluated:** 14,274
- **Accuracy:** `99.99%`
- **Precision:** `99.19%`
- **Recall:** `100.00%`
- **F1 Score:** `99.59%`
- **ROC-AUC:** `1.0000`

### Confusion Matrix
```
                      Predicted Different (0)    Predicted Same (1)
Actual Different (0)          14,151 (TN)               1 (FP)
Actual Same (1)                    0 (FN)             122 (TP)
```

---

## 3. Computational Efficiency & Model Footprint
- **Total Parameters:** `24,623,808` (~24.6 Million)
- **Trainable Parameters:** `24,623,808`
- **Inference Latency (CPU):** `54.86 ms` / image
- **Inference Latency (GPU T4):** `~4.2 ms` / image
- **Model Checkpoint Size:** `94.24 MB`
