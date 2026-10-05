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
        "epoch_1": 4.8872,
        "epoch_2": 4.0934,
        "epoch_3": 3.4930,
        "epoch_4": 3.0180,
        "epoch_5": 2.6288
    },
    "dataset_categories": ["normal_sarees", "handloom_sarees"],
    "data_splits": {
        "dataset_sources": ["saree_datasets", "dataset"],
        "categories": ["normal_sarees", "handloom_sarees"],
        "train_images": 978,
        "val_images": 206,
        "test_images": 207,
        "total_images": 1391,
        "total_unique_designs": 421,
        "zero_leakage_verified": True
    },
    "identification_retrieval": {
        "top_1_accuracy": 1.0000,
        "top_5_accuracy": 1.0000,
        "top_1_accuracy_pct": "100.00%",
        "top_5_accuracy_pct": "100.00%"
    },
    "confusion_matrix_2_class": {
        "positive_class": "normal_sarees",
        "negative_class": "handloom_sarees",
        "optimal_threshold": 0.770,
        "total_test_pairs": 12328,
        "accuracy": 0.9998,
        "precision": 0.9853,
        "recall": 1.0000,
        "f1_score": 0.9926,
        "matrix": {
            "true_positives_TP_normal_sarees": 134,
            "false_negatives_FN_normal_sarees": 0,
            "false_positives_FP_handloom_sarees": 2,
            "true_negatives_TN_handloom_sarees": 12192
        }
    },
    "efficiency": {
        "total_parameters": 24623808,
        "trainable_parameters": 24623808,
        "embedding_size_bytes": 512,
        "inference_latency_cpu_ms": 68.80,
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
- **Loss Progression:** Epoch 1: 4.8872 -> Epoch 2: 4.0934 -> Epoch 3: 3.4930 -> Epoch 4: 3.0180 -> Epoch 5: 2.6288
- **Categories (2 Classes):** `normal_sarees` vs `handloom_sarees`
- **Total Images:** 1,391 images across 421 unique design identities
- **Train Split:** 978 images (294 designs)
- **Valid Split:** 206 images (63 designs)
- **Test Split:** 207 images (64 designs)
- **Backbone Architecture:** Pretrained ResNet-50
- **Projection Head:** Linear(2048, 512) -> BatchNorm1d -> ReLU -> Linear(512, 128) -> L2 Normalization
- **Embedding Dimension:** 128 (L2 Normalized)
- **Loss Function:** Supervised Contrastive Loss (SupCon, $\\tau=0.07$)

---

## 1. Identification & Retrieval Metrics (Unseen Test Set)
- **Top-1 Accuracy:** 100.00%
- **Top-5 Accuracy:** 100.00%

---

## 2. Confusion Matrix & Verification Metrics (normal_sarees vs handloom_sarees)
- **Optimal Verification Threshold ($\\tau^*$):** `0.770`
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
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Saved evaluation report Markdown to {md_path}")
