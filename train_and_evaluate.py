import functools
print = functools.partial(print, flush=True)
import re
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

# =====================================================================
# 1. CONFIGURATION & REPRODUCIBILITY
# =====================================================================
class Config:
    DATA_ROOT = Path("./dataset")
    SPLIT_METADATA = Path("splits_metadata.json")
    CHECKPOINT_DIR = Path("checkpoints")
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_saree_model.pth"
    
    SEED = 42
    IMAGE_SIZE = (224, 224)
    EMBEDDING_DIM = 128
    
    BATCH_SIZE = 32
    EPOCHS = 2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.07  # SupCon loss temperature
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)
Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 2. DATASET SPLITTING & METADATA PREPARATION
# =====================================================================
def extract_base_design_id(image_path):
    filename = Path(image_path).name
    category = Path(image_path).parent.name
    # Strip Roboflow augmentation suffix: _jpg.rf.<hash>.jpg
    base_name = re.sub(r'_(jpg|jpeg|png)\.rf\.[a-f0-9]+\.(jpg|jpeg|png)$', '', filename, flags=re.IGNORECASE)
    return f"{category}_{base_name}"

def prepare_data_splits(data_root):
    candidate_paths = [Path(data_root), Path("./dataset"), Path("./saree_datasets"), Path("./archive")]
    data_path = None
    for p in candidate_paths:
        if p.exists() and len(list(p.rglob('*.*'))) > 0:
            data_path = p
            break
            
    if data_path is None:
        raise FileNotFoundError(f"Could not locate dataset in any of {candidate_paths}")
        
    design_to_images = defaultdict(list)
    image_paths = [p for p in data_path.rglob('*.*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    
    for p in image_paths:
        design_id = extract_base_design_id(p)
        design_to_images[design_id].append(str(p.resolve()))
        
    unique_designs = sorted(list(design_to_images.keys()))
    random.seed(Config.SEED)
    random.shuffle(unique_designs)
    
    num_designs = len(unique_designs)
    train_end = int(0.70 * num_designs)
    val_end = int(0.85 * num_designs)
    
    train_designs = set(unique_designs[:train_end])
    val_designs = set(unique_designs[train_end:val_end])
    test_designs = set(unique_designs[val_end:])
    
    # Map design IDs to numerical labels for train set
    design2label = {d: idx for idx, d in enumerate(sorted(list(train_designs)))}
    
    splits = {"train": [], "val": [], "test": []}
    for design_id, paths in design_to_images.items():
        if design_id in train_designs:
            target_split = "train"
            label = design2label[design_id]
        elif design_id in val_designs:
            target_split = "val"
            label = -1  # Validation/Test designs are open-set (unseen)
        else:
            target_split = "test"
            label = -1
            
        for path in paths:
            splits[target_split].append({
                "image_path": path,
                "design_id": design_id,
                "label": label
            })
            
    # Build Gallery / Query protocol for Val & Test sets
    def make_gallery_query(eval_items):
        d_map = defaultdict(list)
        for item in eval_items:
            d_map[item["design_id"]].append(item)
        gallery, query = [], []
        for d_id, items in d_map.items():
            if len(items) >= 2:
                gallery.append(items[0])
                query.extend(items[1:])
            else:
                gallery.append(items[0])
        return gallery, query
        
    val_g, val_q = make_gallery_query(splits["val"])
    test_g, test_q = make_gallery_query(splits["test"])
    
    return splits, val_g, val_q, test_g, test_q

# =====================================================================
# 3. TRANSFORMS & PYTORCH DATASET
# =====================================================================
# Augmentation strategy: Preserves motif/geometric structure while destroying color dependency.
# - ColorJitter & RandomGrayscale: Force the backbone to rely on motif texture/shapes, not color.
# - RandomResizedCrop & Flips: Provide spatial invariance without altering directional weave symmetry.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.25),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize(Config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SareeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["design_id"], item["label"], item["image_path"]

# =====================================================================
# 4. MODEL ARCHITECTURE
# =====================================================================
class ColorInvariantSareeNet(nn.Module):
    """
    Backbone: Pretrained ResNet-50
    Projection Head: 2048 -> 512 (BatchNorm + ReLU) -> 128-D
    Output: L2-normalized 128-dimensional embedding for Cosine Similarity.
    """
    def __init__(self, embedding_dim=128, pretrained=True):
        super(ColorInvariantSareeNet, self).__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Extract feature extractor backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.projection_head(features)
        # L2 Normalization ensures cosine similarity equals dot product
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

# =====================================================================
# 5. LOSS FUNCTION: SUPERVISED CONTRASTIVE LOSS
# =====================================================================
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    Pulls embeddings of the same saree design together (positive pairs)
    and pushes embeddings of different saree designs apart (negative pairs),
    regardless of color palette changes.
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Mask out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log-probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss

# =====================================================================
# 6. INFERENCE & EMBEDDING EXTRACTION
# =====================================================================
@torch.no_grad()
def get_embedding(model, image_tensor, device=Config.DEVICE):
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    embeddings = model(image_tensor)
    return embeddings.squeeze(0).cpu()

@torch.no_grad()
def build_gallery(model, dataloader, device=Config.DEVICE):
    model.eval()
    all_embeddings = []
    all_design_ids = []
    all_paths = []
    
    for images, design_ids, _, paths in dataloader:
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_design_ids.extend(design_ids)
        all_paths.extend(paths)
        
    gallery_embeddings = torch.cat(all_embeddings, dim=0)
    return gallery_embeddings, all_design_ids, all_paths

def retrieve(query_embedding, gallery_embeddings, top_k=5):
    """
    Cosine similarity retrieval between query embedding and gallery embeddings.
    """
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
        
    # Dot product of L2-normalized vectors equals Cosine Similarity
    similarities = torch.mm(query_embedding, gallery_embeddings.T).squeeze(0)
    top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
    return top_indices.tolist(), top_scores.tolist()

def verify(image1_embed, image2_embed, threshold):
    """
    Verifies whether two saree images contain the same design.
    """
    similarity = torch.dot(image1_embed, image2_embed).item()
    is_same = similarity >= threshold
    return is_same, similarity

# =====================================================================
# 7. TRAINING & VALIDATION PIPELINE
# =====================================================================
def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    
    for images, _, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                embeddings = model(images)
                loss = criterion(embeddings, labels)
        else:
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * images.size(0)
        
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate_retrieval(model, gallery_loader, query_loader, device):
    gallery_embeds, gallery_ids, _ = build_gallery(model, gallery_loader, device)
    
    top1_correct = 0
    top5_correct = 0
    total_queries = 0
    
    for images, query_ids, _, _ in query_loader:
        images = images.to(device)
        query_embeds = model(images).cpu()
        
        for i in range(len(query_ids)):
            q_id = query_ids[i]
            q_embed = query_embeds[i]
            
            top_indices, _ = retrieve(q_embed, gallery_embeds, top_k=5)
            retrieved_ids = [gallery_ids[idx] for idx in top_indices]
            
            if retrieved_ids[0] == q_id:
                top1_correct += 1
            if q_id in retrieved_ids[:5]:
                top5_correct += 1
                
            total_queries += 1
            
    top1_acc = top1_correct / max(total_queries, 1)
    top5_acc = top5_correct / max(total_queries, 1)
    return top1_acc, top5_acc

@torch.no_grad()
def find_optimal_threshold(model, gallery_loader, query_loader, device):
    """
    Finds optimal verification threshold on validation set maximizing F1-score.
    Pairs are drawn between queries and gallery items.
    """
    gallery_embeds, gallery_ids, _ = build_gallery(model, gallery_loader, device)
    
    sims, targets = [], []
    for images, query_ids, _, _ in query_loader:
        images = images.to(device)
        query_embeds = model(images).cpu()
        
        for i in range(len(query_ids)):
            q_id = query_ids[i]
            q_embed = query_embeds[i]
            
            for j in range(len(gallery_ids)):
                g_id = gallery_ids[j]
                g_embed = gallery_embeds[j]
                
                sim = torch.dot(q_embed, g_embed).item()
                is_same = 1 if q_id == g_id else 0
                sims.append(sim)
                targets.append(is_same)
                
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        preds = [1 if s >= thresh else 0 for s in sims]
        _, _, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

# =====================================================================
# 8. FULL EVALUATION SUITE
# =====================================================================
def run_full_evaluation(model, test_gallery_loader, test_query_loader, best_thresh, device):
    print("\n" + "="*60)
    print("      FINAL EVALUATION RESULTS ON UNSEEN TEST SET")
    print("="*60)
    
    # 1. Identification & Retrieval
    top1_acc, top5_acc = evaluate_retrieval(model, test_gallery_loader, test_query_loader, device)
    print(f"\n--- IDENTIFICATION & RETRIEVAL ---")
    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc * 100:.2f}%")
    
    # 2. Verification Protocol (Query vs Gallery pairs)
    gallery_embeds, gallery_ids, _ = build_gallery(model, test_gallery_loader, device)
    sims, targets = [], []
    
    for images, query_ids, _, _ in test_query_loader:
        images = images.to(device)
        query_embeds = model(images).cpu()
        
        for i in range(len(query_ids)):
            q_id = query_ids[i]
            q_embed = query_embeds[i]
            
            for j in range(len(gallery_ids)):
                g_id = gallery_ids[j]
                g_embed = gallery_embeds[j]
                
                sim = torch.dot(q_embed, g_embed).item()
                is_same = 1 if q_id == g_id else 0
                sims.append(sim)
                targets.append(is_same)
                
    preds = [1 if s >= best_thresh else 0 for s in sims]
    prec, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
    acc = np.mean(np.array(preds) == np.array(targets))
    cm = confusion_matrix(targets, preds)
    
    print(f"\n--- VERIFICATION PROTOCOL (Optimal Threshold = {best_thresh:.3f}) ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    
    # 3. Efficiency & Model Metrics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Latency benchmark
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    avg_latency = (time.time() - start_time) / 100 * 1000  # ms
    
    print(f"\n--- EFFICIENCY MEASUREMENTS ---")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Embedding Dimension:   {Config.EMBEDDING_DIM}")
    print(f"Inference Latency:     {avg_latency:.2f} ms / image (Device: {device})")
    if Config.BEST_MODEL_PATH.exists():
        size_mb = Config.BEST_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Checkpoint File Size:  {size_mb:.2f} MB")

# =====================================================================
# 9. MAIN EXECUTION PIPELINE
# =====================================================================
def main():
    print(f"Using Device: {Config.DEVICE}")
    
    # 1. Prepare leak-free splits
    splits, val_g, val_q, test_g, test_q = prepare_data_splits(Config.DATA_ROOT)
    
    # 2. DataLoaders
    train_dataset = SareeDataset(splits["train"], transform=train_transforms)
    val_g_dataset = SareeDataset(val_g, transform=eval_transforms)
    val_q_dataset = SareeDataset(val_q, transform=eval_transforms)
    test_g_dataset = SareeDataset(test_g, transform=eval_transforms)
    test_q_dataset = SareeDataset(test_q, transform=eval_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_g_loader = DataLoader(val_g_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    val_q_loader = DataLoader(val_q_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_g_loader = DataLoader(test_g_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_q_loader = DataLoader(test_q_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 3. Model, Loss, Optimizer
    model = ColorInvariantSareeNet(embedding_dim=Config.EMBEDDING_DIM, pretrained=True).to(Config.DEVICE)
    criterion = SupervisedContrastiveLoss(temperature=Config.TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print("\nStarting Training Pipeline...")
    best_val_top1 = 0.0
    
    for epoch in range(1, Config.EPOCHS + 1):
        start_t = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, Config.DEVICE)
        scheduler.step()
        
        val_top1, val_top5 = evaluate_retrieval(model, val_g_loader, val_q_loader, Config.DEVICE)
        elapsed = time.time() - start_t
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Val Top-1: {val_top1*100:.2f}% | Val Top-5: {val_top5*100:.2f}% | LR: {lr:.6f} | Time: {elapsed:.1f}s")
        
        if val_top1 >= best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            
    print(f"\nTraining Complete. Best Validation Top-1: {best_val_top1*100:.2f}%")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    best_thresh = find_optimal_threshold(model, val_g_loader, val_q_loader, Config.DEVICE)
    print(f"Determined Optimal Verification Threshold: {best_thresh:.3f}")
    
    # Run evaluation
    run_full_evaluation(model, test_g_loader, test_q_loader, best_thresh, Config.DEVICE)

if __name__ == "__main__":
    main()
