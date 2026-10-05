import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# GENERATE & SAVE CONFUSION MATRIX PLOT
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
plt.title('ResNet-50 Confusion Matrix (Saree Verification)\nOptimal Threshold = 0.760', fontsize=13, fontweight='bold', pad=15)

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
