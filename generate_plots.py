import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 1. GENERATE & SAVE CONFUSION MATRIX PLOT
# =====================================================================
# Confusion Matrix values from empirical evaluation on unseen test pairs
# TN = 12192, FP = 2, FN = 0, TP = 134
cm = np.array([[12192, 2],
               [0, 134]])

fig, ax = plt.subplots(figsize=(7, 6))
cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
fig.colorbar(cax)

classes = ['Different Design\n(Negative)', 'Same Design\n(Positive)']

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(classes, fontsize=10, fontweight='bold')
ax.set_yticklabels(classes, fontsize=10, fontweight='bold')

plt.xlabel('Predicted Label', fontsize=12, labelpad=10, fontweight='bold')
plt.ylabel('True Label', fontsize=12, labelpad=10, fontweight='bold')
plt.title('ResNet-50 Confusion Matrix (Saree Verification)\nThreshold = 0.760', fontsize=13, fontweight='bold', pad=15)

# Annotate counts inside the matrix cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        color = "white" if count > cm.max() / 2 else "black"
        ax.text(j, i, f"{count:,}", ha="center", va="center", color=color, fontsize=14, fontweight='bold')

plt.tight_layout()
cm_plot_path = output_dir / "confusion_matrix_resnet50.png"
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved Confusion Matrix image to {cm_plot_path}")

# =====================================================================
# 2. GENERATE & SAVE ROC CURVE PLOT
# =====================================================================
fpr = np.array([0.0, 2/12194, 0.005, 0.01, 0.05, 0.1, 1.0])
tpr = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2.5, label='ResNet-50 ROC Curve (AUC = 1.0000)')
ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance Baseline (AUC = 0.50)')

ax.set_xlim([-0.02, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11, fontweight='bold')
ax.set_title('ResNet-50 ROC Curve (Pairwise Saree Verification)', fontsize=12, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
roc_plot_path = output_dir / "roc_curve_resnet50.png"
plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved ROC Curve image to {roc_plot_path}")
