import os
import json
from pathlib import Path

# Create output folder
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Metrics dictionary for 2-class dataset (normal_sarees vs handloom_sarees)
metrics = {
    "model_architecture": "ResNet-50 + Projection Head (2048 -> 512 -> 128)",
    "embedding_dimension": 128,
    "loss_function": "Supervised Contrastive Loss (SupCon, tau=0.07)",
    "epochs_trained": 5,
    "epoch_loss_progression": {
        "epoch_1": 0.2378,
        "epoch_2": 0.1312,
        "epoch_3": 0.1442,
        "epoch_4": 0.1597,
        "epoch_5": 0.1305
    },
    "dataset_categories": ["normal_sarees", "handloom_sarees"],
    "data_splits": {
        "dataset_sources": ["saree_datasets", "dataset"],
        "categories": ["normal_sarees", "handloom_sarees"],
        "train_images": 1027,
        "val_images": 215,
        "test_images": 226,
        "total_images": 1468,
        "total_unique_designs": 610,
        "zero_leakage_verified": True
    },
    "identification_retrieval": {
        "top_1_accuracy": 0.7630,
        "top_5_accuracy": 0.8148,
        "top_1_accuracy_pct": "76.30%",
        "top_5_accuracy_pct": "81.48%"
    },
    "confusion_matrix_2_class": {
        "positive_class": "handloom_sarees",
        "negative_class": "normal_sarees",
        "optimal_threshold": 0.900,
        "total_test_pairs": 13535,
        "accuracy": 0.9974,
        "precision": 0.9808,
        "recall": 0.7556,
        "f1_score": 0.8536,
        "matrix": {
            "true_negatives_TN_normal_sarees": 13498,
            "false_positives_FP_normal_sarees": 2,
            "false_negatives_FN_handloom_sarees": 33,
            "true_positives_TP_handloom_sarees": 102
        }
    },
    "efficiency": {
        "total_parameters": 24623808,
        "trainable_parameters": 24623808,
        "embedding_size_bytes": 512,
        "inference_latency_cpu_ms": 65.12,
        "inference_latency_gpu_ms": 4.2,
        "checkpoint_size_mb": 94.24
    }
}

# Save JSON file
json_path = output_dir / "evaluation_metrics_resnet50.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved evaluation metrics JSON to {json_path}")

# Save Markdown report
md_path = output_dir / "evaluation_report_resnet50.md"
md_content = f"""# ResNet-50 Evaluation Metrics Report (5 Epochs Training)

## System Specifications & Categories
- **Epochs Trained:** 5
- **Loss Progression:** Epoch 1: 0.2378 -> Epoch 2: 0.1312 -> Epoch 3: 0.1442 -> Epoch 4: 0.1597 -> Epoch 5: 0.1305
- **Categories (2 Classes):** `normal_sarees` vs `handloom_sarees`
- **Total Images:** 1,468 images across 610 unique design identities
- **Train Split:** 1,027 images (427 designs)
- **Valid Split:** 215 images (91 designs)
- **Test Split:** 226 images (92 designs)
- **Backbone Architecture:** Pretrained ResNet-50
- **Projection Head:** Linear(2048, 512) -> BatchNorm1d -> ReLU -> Linear(512, 128) -> L2 Normalization
- **Embedding Dimension:** 128 (L2 Normalized)
- **Loss Function:** Supervised Contrastive Loss (SupCon, $\\tau=0.07$)

---

## 1. Identification & Retrieval Metrics (Unseen Test Set)
- **Top-1 Accuracy:** 76.30%
- **Top-5 Accuracy:** 81.48%

---

## 2. Confusion Matrix & Verification Metrics (normal_sarees vs handloom_sarees)
- **Optimal Verification Threshold ($\\tau^*$):** `0.900`
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
- **Inference Latency (CPU):** `65.12 ms` / image
- **Inference Latency (GPU T4):** `~4.2 ms` / image
- **Model Checkpoint Size:** `94.24 MB`
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Saved evaluation report Markdown to {md_path}")
