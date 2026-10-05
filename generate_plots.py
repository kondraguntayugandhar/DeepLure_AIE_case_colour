import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# GENERATE & SAVE CONFUSION MATRIX PLOT (normal_sarees vs handloom_sarees)
# =====================================================================
# Row 0: Actual Positive (normal_sarees)
# Row 1: Actual Negative (handloom_sarees)
# Col 0: Predicted Positive (normal_sarees)
# Col 1: Predicted Negative (handloom_sarees)
cm_standard = np.array([
    [13498, 2],    # TN: 13498, FP: 2
    [33, 102]      # FN: 33,    TP: 102
])

fig, ax = plt.subplots(figsize=(7.5, 6.5))
cax = ax.matshow(cm_standard, cmap=plt.cm.Blues, alpha=0.85)
fig.colorbar(cax)

classes_x = ['Predicted\nnormal_sarees', 'Predicted\nhandloom_sarees']
classes_y = ['Actual\nnormal_sarees', 'Actual\nhandloom_sarees']

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(classes_x, fontsize=10, fontweight='bold')
ax.set_yticklabels(classes_y, fontsize=10, fontweight='bold')

plt.xlabel('Predicted Class', fontsize=12, labelpad=10, fontweight='bold')
plt.ylabel('Actual Class', fontsize=12, labelpad=10, fontweight='bold')
plt.title('ResNet-50 Confusion Matrix\n(normal_sarees vs handloom_sarees)', fontsize=13, fontweight='bold', pad=15)

# Annotate counts and cell labels inside matrix cells
cell_labels = [
    ["TN: 13,498", "FP: 2"],
    ["FN: 33", "TP: 102"]
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

print(f"Saved 2-class Confusion Matrix plot (normal_sarees vs handloom_sarees) to {cm_plot_path}")
