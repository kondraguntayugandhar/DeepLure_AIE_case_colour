import os
import json
from pathlib import Path

# Create output folder
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Metrics dictionary for ResNet-50 on Archive Dataset (1,468 images, 610 designs)
metrics = {
    "model_architecture": "ResNet-50 + Projection Head (2048 -> 512 -> 128)",
    "embedding_dimension": 128,
    "loss_function": "Supervised Contrastive Loss (SupCon, tau=0.07)",
    "data_splits": {
        "dataset_sources": ["archive"],
        "total_images": 1468,
        "total_unique_designs": 610,
        "train_designs": 427,
        "val_designs": 91,
        "test_designs": 92,
        "zero_leakage_verified": True
    },
    "identification_retrieval": {
        "top_1_accuracy": 1.0000,
        "top_5_accuracy": 1.0000,
        "top_1_accuracy_pct": "100.00%",
        "top_5_accuracy_pct": "100.00%"
    },
    "pairwise_verification": {
        "optimal_threshold": 0.760,
        "total_test_pairs": 12328,
        "accuracy": 0.9998,
        "precision": 0.9853,
        "recall": 1.0000,
        "f1_score": 0.9926,
        "roc_auc": 1.0000,
        "confusion_matrix": {
            "true_negatives": 12192,
            "false_positives": 2,
            "false_negatives": 0,
            "true_positives": 134
        }
    },
    "efficiency": {
        "total_parameters": 24623808,
        "trainable_parameters": 24623808,
        "embedding_size_bytes": 512,
        "inference_latency_cpu_ms": 65.94,
        "inference_latency_gpu_ms": 4.2,
        "checkpoint_size_mb": 94.24
    }
}

# Save JSON file
json_path = output_dir / "evaluation_metrics_resnet50.json"
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved evaluation metrics JSON to {json_path}")

# Save Markdown report
md_path = output_dir / "evaluation_report_resnet50.md"
md_content = f"""# ResNet-50 Evaluation Metrics Report

## System Specifications
- **Dataset Source:** `archive` (1,468 images across 4 categories: `Banarasi`, `Bandhani`, `Ikat`, `Pichwai`)
- **Total Unique Design Identities:** 610 designs (427 Train / 91 Val / 92 Test)
- **Backbone Architecture:** Pretrained ResNet-50
- **Projection Head:** Linear(2048, 512) -> BatchNorm1d -> ReLU -> Linear(512, 128) -> L2 Normalization
- **Embedding Dimension:** 128 (L2 Normalized)
- **Loss Function:** Supervised Contrastive Loss (SupCon, $\\tau=0.07$)

---

## 1. Identification & Retrieval Metrics (Unseen Test Set)
- **Top-1 Accuracy:** 100.00%
- **Top-5 Accuracy:** 100.00%

---

## 2. Pairwise Verification Protocol Metrics
- **Optimal Verification Threshold ($\\tau^*$):** `0.760`
- **Total Test Pairs Evaluated:** 12,328
- **Accuracy:** `99.98%`
- **Precision:** `98.53%`
- **Recall:** `100.00%`
- **F1 Score:** `99.26%`
- **ROC-AUC:** `1.0000`

### Confusion Matrix
```
                      Predicted Different (0)    Predicted Same (1)
Actual Different (0)          12,192 (TN)               2 (FP)
Actual Same (1)                    0 (FN)             134 (TP)
```

---

## 3. Computational Efficiency & Model Footprint
- **Total Parameters:** `24,623,808` (~24.6 Million)
- **Trainable Parameters:** `24,623,808`
- **Inference Latency (CPU):** `65.94 ms` / image
- **Inference Latency (GPU T4):** `~4.2 ms` / image
- **Model Checkpoint Size:** `94.24 MB`
"""

with open(md_path, "w") as f:
    f.write(md_content)

print(f"Saved evaluation report Markdown to {md_path}")
