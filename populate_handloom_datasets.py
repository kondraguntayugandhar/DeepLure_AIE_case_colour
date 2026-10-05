import os
import shutil
import random
from pathlib import Path

random.seed(42)

# Locate handloom source directory
handloom_source = Path("handloom_sarees")
if not handloom_source.exists():
    handloom_source = Path("handlooms")

if handloom_source.exists():
    handloom_imgs = [p for p in handloom_source.rglob('*.*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    random.shuffle(handloom_imgs)
    
    total_h = len(handloom_imgs)
    h_train_end = int(0.70 * total_h)
    h_val_end = int(0.85 * total_h)
    
    train_h = handloom_imgs[:h_train_end]
    val_h = handloom_imgs[h_train_end:h_val_end]
    test_h = handloom_imgs[h_val_end:]
    
    dataset_roots = [Path("saree_datasets"), Path("dataset")]
    
    for root in dataset_roots:
        if root.exists():
            (root / "train" / "handloom_sarees").mkdir(parents=True, exist_ok=True)
            (root / "valid" / "handloom_sarees").mkdir(parents=True, exist_ok=True)
            (root / "test" / "handloom_sarees").mkdir(parents=True, exist_ok=True)
            
            for p in train_h:
                shutil.copy2(p, root / "train" / "handloom_sarees" / p.name)
            for p in val_h:
                shutil.copy2(p, root / "valid" / "handloom_sarees" / p.name)
            for p in test_h:
                shutil.copy2(p, root / "test" / "handloom_sarees" / p.name)
                
            print(f"Populated {root.name} with {total_h} handloom_sarees images:")
            print(f"  Train: {len(train_h)} images")
            print(f"  Valid: {len(val_h)} images")
            print(f"  Test:  {len(test_h)} images")

print("Handloom dataset population complete!")
