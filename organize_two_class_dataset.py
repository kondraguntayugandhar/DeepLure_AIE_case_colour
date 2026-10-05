import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

source_dataset = Path("saree_datasets")
if not source_dataset.exists():
    source_dataset = Path("archive")

target_dataset = Path("dataset")
if target_dataset.exists():
    shutil.rmtree(target_dataset)

# Create train, valid, test for 2 classes: handloom_sarees & normal_sarees
classes = ["normal_sarees", "handloom_sarees"]
splits = ["train", "valid", "test"]

for s in splits:
    for c in classes:
        (target_dataset / s / c).mkdir(parents=True, exist_ok=True)

# 1. Collect all images from source_dataset (Roboflow patterns -> normal_sarees)
def get_base_id(p):
    filename = p.name
    base_name = re.sub(r'_(jpg|jpeg|png)\.rf\.[a-f0-9]+\.(jpg|jpeg|png)$', '', filename, flags=re.IGNORECASE)
    return f"{p.parent.name}_{base_name}"

normal_design_map = defaultdict(list)
normal_imgs = [p for p in source_dataset.rglob('*.*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

for p in normal_imgs:
    d_id = get_base_id(p)
    normal_design_map[d_id].append(p)

unique_normal_designs = sorted(list(normal_design_map.keys()))
random.shuffle(unique_normal_designs)

num_normal = len(unique_normal_designs)
n_train_end = int(0.70 * num_normal)
n_val_end = int(0.85 * num_normal)

normal_train_ids = set(unique_normal_designs[:n_train_end])
normal_val_ids = set(unique_normal_designs[n_train_end:n_val_end])
normal_test_ids = set(unique_normal_designs[n_val_end:])

normal_counts = {"train": 0, "valid": 0, "test": 0}

for d_id, paths in normal_design_map.items():
    if d_id in normal_train_ids:
        target_split = "train"
    elif d_id in normal_val_ids:
        target_split = "valid"
    else:
        target_split = "test"
        
    for p in paths:
        dest_path = target_dataset / target_split / "normal_sarees" / p.name
        shutil.copy2(p, dest_path)
        normal_counts[target_split] += 1

print(f"Segregated Normal Sarees ({len(normal_imgs)} images, {num_normal} design IDs):")
print(f"  Train: {normal_counts['train']} images")
print(f"  Valid: {normal_counts['valid']} images")
print(f"  Test:  {normal_counts['test']} images")

# 2. Check for handloom_sarees folder if present
handloom_source = Path("handlooms")
if not handloom_source.exists():
    handloom_source = Path("handloom_sarees-20261005T110039Z-1-001")

handloom_counts = {"train": 0, "valid": 0, "test": 0}
if handloom_source.exists():
    handloom_imgs = [p for p in handloom_source.rglob('*.*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    random.shuffle(handloom_imgs)
    
    h_total = len(handloom_imgs)
    h_train_end = int(0.70 * h_total)
    h_val_end = int(0.85 * h_total)
    
    h_train_imgs = handloom_imgs[:h_train_end]
    h_val_imgs = handloom_imgs[h_train_end:h_val_end]
    h_test_imgs = handloom_imgs[h_val_end:]
    
    for p in h_train_imgs:
        shutil.copy2(p, target_dataset / "train" / "handloom_sarees" / p.name)
        handloom_counts["train"] += 1
    for p in h_val_imgs:
        shutil.copy2(p, target_dataset / "valid" / "handloom_sarees" / p.name)
        handloom_counts["valid"] += 1
    for p in h_test_imgs:
        shutil.copy2(p, target_dataset / "test" / "handloom_sarees" / p.name)
        handloom_counts["test"] += 1
        
    print(f"\nSegregated Handloom Sarees ({h_total} images):")
    print(f"  Train: {handloom_counts['train']} images")
    print(f"  Valid: {handloom_counts['valid']} images")
    print(f"  Test:  {handloom_counts['test']} images")
else:
    print("\nNote: Handloom source directory not found locally. Created empty class folder for handloom_sarees.")

print("\nDataset segregation complete! Target folder: './dataset'")
