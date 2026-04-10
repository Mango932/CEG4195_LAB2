import os
import numpy as np
from datasets import load_dataset
from PIL import Image

# ==============================
# 1. Paths (EDIT THIS)
# ==============================
BASE_DIR = "dataset"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train/images")
TRAIN_LBL_DIR = os.path.join(BASE_DIR, "train/labels")
VAL_IMG_DIR   = os.path.join(BASE_DIR, "val/images")
VAL_LBL_DIR   = os.path.join(BASE_DIR, "val/labels")

# Create folders if they don’t exist
for path in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
    os.makedirs(path, exist_ok=True)

# ==============================
# 2. Load dataset
# ==============================
ds = load_dataset("keremberke/satellite-building-segmentation", name="full")
train_data = ds["train"]

# ==============================
# 3. Shuffle + select 100 samples
# ==============================
shuffled = train_data.shuffle(seed=42)
subset = shuffled.select(range(100))

train_subset = subset.select(range(90))
val_subset   = subset.select(range(90, 100))

# ==============================
# 4. Save function
# ==============================
def save_sample(sample, idx, img_dir, lbl_dir):
    image = sample["image"]
    bboxes = sample["objects"]["bbox"]

    # Save image
    img_path = os.path.join(img_dir, f"{idx}.png")
    image.save(img_path)

    # Save labels (YOLO format)
    w, h = image.size

    label_path = os.path.join(lbl_dir, f"{idx}.txt")
    with open(label_path, "w") as f:
        for bbox in bboxes:
            x, y, bw, bh = bbox

            # Convert to YOLO format (normalized)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h

            class_id = 0  # building class

            f.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")

# ==============================
# 5. Save train data
# ==============================
for i, sample in enumerate(train_subset):
    save_sample(sample, i, TRAIN_IMG_DIR, TRAIN_LBL_DIR)

# ==============================
# 6. Save validation data
# ==============================
for i, sample in enumerate(val_subset):
    save_sample(sample, i, VAL_IMG_DIR, VAL_LBL_DIR)

print("✅ Done! 90 train + 10 validation images saved.")