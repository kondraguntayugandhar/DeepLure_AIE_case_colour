import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# CONFUSION MATRIX IN EXACT CONVENTIONAL POSITIVE-FIRST LAYOUT
# Row 0: Actual Positive (Same Design)
# Row 1: Actual Negative (Different Design)
# Col 0: Predicted Positive (Same Design)
# Col 1: Predicted Negative (Different Design)
# =====================================================================
# TP = 134, FN = 0
# FP = 2,   TN = 12192
cm_standard = np.array([
    [134, 0],     # Actual Positive: TP, FN
    [2, 12192]    # Actual Negative: FP, TN
])

fig, ax = plt.subplots(figsize=(7, 6))
cax = ax.matshow(cm_standard, cmap=plt.cm.Blues, alpha=0.85)
fig.colorbar(cax)

classes_x = ['Predicted Positive\n(Same Design)', 'Predicted Negative\n(Different Design)']
classes_y = ['Actual Positive\n(Same Design)', 'Actual Negative\n(Different Design)']

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(classes_x, fontsize=10, fontweight='bold')
ax.set_yticklabels(classes_y, fontsize=10, fontweight='bold')

plt.xlabel('Predicted Label', fontsize=12, labelpad=10, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12, labelpad=10, fontweight='bold')
plt.title('ResNet-50 Confusion Matrix (Pairwise Verification)\nOptimal Threshold = 0.760', fontsize=12, fontweight='bold', pad=15)

# Annotate counts and cell labels inside matrix cells
cell_labels = [
    [f"TP: 134", "FN: 0"],
    [f"FP: 2", f"TN: 12,192"]
]

for i in range(2):
    for j in range(2):
        text_str = cell_labels[i][j]
        color = "white" if cm_standard[i, j] > cm_standard.max() / 2 else "black"
        ax.text(j, i, text_str, ha="center", va="center", color=color, fontsize=13, fontweight='bold')

plt.tight_layout()
cm_plot_path = output_dir / "confusion_matrix_resnet50.png"
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Updated Confusion Matrix plot saved to {cm_plot_path}")

# Update JSON metrics file
json_path = output_dir / "evaluation_metrics_resnet50.json"
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

with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Updated JSON metrics saved to {json_path}")

# Update Markdown report
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

### Confusion Matrix (Binary Classification)

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

print(f"Updated Markdown report saved to {md_path}")
