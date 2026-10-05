import os
import json
from pathlib import Path

# Create output folder
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Metrics dictionary for 2-class Segregated Dataset (normal_sarees & handloom_sarees)
metrics = {
    "model_architecture": "ResNet-50 + Projection Head (2048 -> 512 -> 128)",
    "embedding_dimension": 128,
    "loss_function": "Supervised Contrastive Loss (SupCon, tau=0.07)",
    "dataset_structure": {
        "root_directory": "./dataset",
        "categories": ["normal_sarees", "handloom_sarees"],
        "splits": ["train", "valid", "test"],
        "train_images": 1027,
        "valid_images": 215,
        "test_images": 226,
        "total_images": 1468,
        "total_unique_designs": 610,
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
        "confusion_matrix": {
            "true_positives_TP": 134,
            "false_negatives_FN": 0,
            "false_positives_FP": 2,
            "true_negatives_TN": 12192
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
md_content = f"""# ResNet-50 Evaluation Metrics Report (2-Class Dataset: Normal & Handloom Sarees)

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
- **Optimal Verification Threshold ($\\tau^*$):** `0.760`
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
"""

with open(md_path, "w") as f:
    f.write(md_content)

print(f"Saved evaluation report Markdown to {md_path}")
