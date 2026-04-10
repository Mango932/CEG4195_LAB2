# ==============================
# 1. Imports
# ==============================
import os
import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ==============================
# 2. Folder Paths — matches your structure exactly
# ==============================
TRAIN_IMG_DIR   = "dataset/train/images"
TRAIN_LABEL_DIR = "dataset/train/labels"
VAL_IMG_DIR     = "dataset/val/images"
VAL_LABEL_DIR   = "dataset/val/labels"

# Create label folders if they don't exist yet
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR,   exist_ok=True)

# ==============================
# 3. Load SAM Model
# ==============================
device         = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type     = "vit_h"

print(f"Loading SAM on {device} ...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
print("SAM loaded.")

# ==============================
# 4. Helper Functions
# ==============================

def get_image_paths(folder):
    """Get all image file paths from a folder."""
    supported = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    paths = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(supported)
    ])
    return paths


def save_mask(mask_np, path):
    """Save binary mask as PNG."""
    cv2.imwrite(path, mask_np)


# ==============================
# 5. Process a Folder
# ==============================

def process_folder(img_dir, label_dir, split_name):
    """
    Reads every image from img_dir,
    runs SAM on it,
    saves the combined mask to label_dir
    with the exact same filename.
    """
    image_paths = get_image_paths(img_dir)
    n           = len(image_paths)
    print(f"\n=== {split_name}: {n} images found in {img_dir} ===")

    if n == 0:
        print(f"  No images found — check your folder path: {img_dir}")
        return

    for idx, img_path in enumerate(image_paths):
        # Load image
        filename = os.path.basename(img_path)           # e.g. image001.png
        image    = np.array(Image.open(img_path).convert("RGB"))

        # Run SAM
        sam_masks = mask_generator.generate(image)

        # Combine all SAM masks into one binary mask
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for m in sam_masks:
            seg = m["segmentation"].astype(np.uint8)
            combined_mask = np.maximum(combined_mask, seg * 255)

        # Save mask with the same filename as the image
        label_path = os.path.join(label_dir, filename)
        save_mask(combined_mask, label_path)

        print(f"  [{idx+1}/{n}] {filename} — {len(sam_masks)} SAM masks → {label_path}")

    print(f"\n{split_name} done!")
    print(f"  Images : {img_dir}")
    print(f"  Labels : {label_dir}")


# ==============================
# 6. Run
# ==============================

process_folder(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, "train")
process_folder(VAL_IMG_DIR,   VAL_LABEL_DIR,   "val")

print("\n========== ALL DONE ==========")
print("Masks saved to:")
print(f"  {TRAIN_LABEL_DIR}")
print(f"  {VAL_LABEL_DIR}")