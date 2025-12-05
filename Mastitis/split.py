import os
import shutil
import random

# Input & Output paths
DATASET_DIR = "dataset"     
OUTPUT_DIR = "dataset_split"
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

# Create split folders
for split in ["train", "val", "test"]:
    for cls in ["mastitis", "healthy"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# Process each class
for cls in ["mastitis", "healthy"]:
    cls_folder = os.path.join(DATASET_DIR, cls)
    images = os.listdir(cls_folder)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(SPLIT_RATIOS[0] * n_total)
    n_val   = int(SPLIT_RATIOS[1] * n_total)

    # Split
    train_files = images[:n_train]
    val_files   = images[n_train:n_train+n_val]
    test_files  = images[n_train+n_val:]

    # Copy files
    for fname in train_files:
        shutil.copy(os.path.join(cls_folder, fname),
                    os.path.join(OUTPUT_DIR, "train", cls, fname))

    for fname in val_files:
        shutil.copy(os.path.join(cls_folder, fname),
                    os.path.join(OUTPUT_DIR, "val", cls, fname))

    for fname in test_files:
        shutil.copy(os.path.join(cls_folder, fname),
                    os.path.join(OUTPUT_DIR, "test", cls, fname))

print(" Dataset split complete! Check 'dataset_split/'")
