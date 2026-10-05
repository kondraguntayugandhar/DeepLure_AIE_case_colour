import os
import shutil
from pathlib import Path

root = Path("saree_datasets")
if not root.exists():
    root = Path("archive")

if root.exists():
    splits = ["train", "valid", "test"]
    for s in splits:
        split_dir = root / s
        if split_dir.exists():
            normal_dir = split_dir / "normal_sarees"
            handloom_dir = split_dir / "handloom_sarees"
            
            normal_dir.mkdir(parents=True, exist_ok=True)
            handloom_dir.mkdir(parents=True, exist_ok=True)
            
            # Move all craft subfolders (Banarasi, Bandhani, Ikat, Pichwai) into normal_sarees
            subdirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name not in ["normal_sarees", "handloom_sarees"]]
            
            moved_count = 0
            for d in subdirs:
                imgs = [p for p in d.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                for img in imgs:
                    dest = normal_dir / img.name
                    shutil.move(img, dest)
                    moved_count += 1
                # Remove empty subfolder after moving images
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
            print(f"Reorganized {root.name}/{s}: Moved {moved_count} images into normal_sarees/")

print("Reorganization of dataset folders complete!")
