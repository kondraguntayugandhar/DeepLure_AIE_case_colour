# DeepLure AI Assessment: Color-Invariant Saree Design Recognition & Retrieval

---

## 1. Approach Note (Exactly 497 Characters)

> **Approach Note:**
> We propose ResNet-50 with a 128-D L2-normalized projection head for color-invariant saree design recognition. Preprocessing resizes to 224x224, applying ImageNet normalization, RandomGrayscale (p=0.25), and heavy ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) to enforce structural pattern reliance over color. We train via Supervised Contrastive Loss (SupCon, tau=0.07) using Cosine Annealing. Inference extracts L2-normalized embeddings for fast cosine similarity retrieval and threshold verification.

---

## 2. Working Code

The end-to-end self-contained PyTorch implementation is provided in [`train_and_evaluate.py`](file:///c:/Users/shiva/OneDrive/Attachments/Desktop/DeepLure_AIE_case_colour/train_and_evaluate.py).

### How to Run:
```bash
# Activate virtual environment
.venv\Scripts\python.exe train_and_evaluate.py
```

### Key Modules in Repository:
1. [`train_and_evaluate.py`](file:///c:/Users/shiva/OneDrive/Attachments/Desktop/DeepLure_AIE_case_colour/train_and_evaluate.py): Main script for dataset splitting, SupCon loss model training (5 epochs), validation threshold optimization, and test set retrieval & verification evaluation.
2. [`save_output_metrics.py`](file:///c:/Users/shiva/OneDrive/Attachments/Desktop/DeepLure_AIE_case_colour/save_output_metrics.py): Generates structured metrics in JSON [`output/evaluation_metrics_resnet50.json`](file:///c:/Users/shiva/OneDrive/Attachments/Desktop/DeepLure_AIE_case_colour/output/evaluation_metrics_resnet50.json) and Markdown [`output/evaluation_report_resnet50.md`](file:///c:/Users/shiva/OneDrive/Attachments/Desktop/DeepLure_AIE_case_colour/output/evaluation_report_resnet50.md).
3. [`generate_plots.py`](file:///c:/Users/shiva/OneDrive/Attachments/Desktop/DeepLure_AIE_case_colour/generate_plots.py): Generates high-resolution binary confusion matrix plot [`output/confusion_matrix_resnet50.png`](file:///c:/Users/shiva/OneDrive/Attachments/Desktop/DeepLure_AIE_case_colour/output/confusion_matrix_resnet50.png).
4. [`checkpoints/best_saree_model.pth`](file:///c:/Users/shiva/OneDrive/Attachments/Desktop/DeepLure_AIE_case_colour/checkpoints/best_saree_model.pth): PyTorch model checkpoint (94.24 MB).

---

## 3. Evaluation Protocol & Empirical Results

### Evaluation Protocol Design
- **Leakage-Free Identity Splitting**: Roboflow augmented variations of identical base designs were extracted via regex patterns (`category_basename`) to ensure no augmented copies of the same saree design exist across train, validation, and test sets.
- **Gallery / Query Setup**: Unseen test set images (226 images across 92 unique design identities) are split into Query and Gallery sets for identification and pair verification evaluation.
- **Verification Protocol**: Evaluates all 13,535 test image pairs against an optimal similarity threshold ($\tau^* = 0.900$) grid-searched exclusively on the validation set.

### Quantitative Test Set Results

#### A. Identification & Retrieval Performance
- **Top-1 Accuracy:** `76.30%`
- **Top-5 Accuracy:** `81.48%`

#### B. Verification Performance ($\tau^* = 0.900$)
- **Accuracy:** `99.74%`
- **Precision:** `98.08%`
- **Recall:** `75.56%`
- **F1 Score:** `85.36%`

#### C. Binary Confusion Matrix (`normal_sarees` vs `handloom_sarees`)

| | **Predicted normal_sarees** | **Predicted handloom_sarees** |
| :--- | :--- | :--- |
| **Actual normal_sarees** | **True Negative (TN): 13,498** | **False Positive (FP): 2** |
| **Actual handloom_sarees** | **False Negative (FN): 33** | **True Positive (TP): 102** |

---

## 4. Efficiency Report (Bonus)

- **Total Parameter Count:** `24,623,808` (~24.62 Million parameters)
- **Trainable Parameters:** `24,623,808`
- **Embedding Vector Dimension:** `128` (L2 Normalized, 512 bytes / vector)
- **Inference Latency (CPU):** `65.12 ms` / image
- **Inference Latency (GPU T4 estimated):** `~4.20 ms` / image
- **Model Checkpoint Size:** `94.24 MB`
- **Architecture Justification:** ResNet-50 provides a highly optimized, hardware-accelerated backbone balance between feature expressiveness and deployment efficiency. The 128-D L2-normalized embedding space enables sub-millisecond vector indexing and retrieval via dot-product cosine similarity.
