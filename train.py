# ==============================
# 1. Imports
# ==============================
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt

# ==============================
# 2. Config
# ==============================
TRAIN_IMG_DIR   = "dataset/train/images"
TRAIN_LABEL_DIR = "dataset/train/labels"
VAL_IMG_DIR     = "dataset/val/images"
VAL_LABEL_DIR   = "dataset/val/labels"

IMAGE_SIZE  = 512
BATCH_SIZE  = 8    
EPOCHS      = 50
LR          = 1e-4
NUM_CLASSES = 2
DEVICE      = "cuda"

# ==============================
# 3. Dataset
# ==============================

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, label_dir, image_size=512):
        self.image_size  = image_size
        self.img_paths   = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.label_paths = sorted([
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.img_transform(image)

        mask  = Image.open(self.label_paths[idx]).convert("L")
        mask  = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask  = np.array(mask)
        mask  = (mask > 127).astype(np.int64)
        mask  = torch.from_numpy(mask)

        return image, mask

# ==============================
# 4. Metrics
# ==============================

def compute_iou(preds, masks):
    preds = preds.bool()
    masks = masks.bool()
    intersection = (preds & masks).sum().item()
    union        = (preds | masks).sum().item()
    return intersection / union if union > 0 else 1.0

def compute_dice(preds, masks):
    preds = preds.bool()
    masks = masks.bool()
    intersection = (preds & masks).sum().item()
    denom        = preds.sum().item() + masks.sum().item()
    return 2 * intersection / denom if denom > 0 else 1.0

# ==============================
# 5. Main — required on Windows
# ==============================

if __name__ == "__main__":

    print(f"Training on : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, IMAGE_SIZE)
    val_dataset   = SegmentationDataset(VAL_IMG_DIR,   VAL_LABEL_DIR,   IMAGE_SIZE)

    train_loader  = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,      # ← fix for Windows (no multiprocessing)
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,      # ← fix for Windows
        pin_memory  = True
    )

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")

    # ── Model ──────────────────────────────────────────────
    model = deeplabv3_resnet101(weights="DEFAULT")
    model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)  # ← 256 not 10
    model = model.to(DEVICE)
    print("Model ready — DeepLabV3+ ResNet-101")

    # Fix deprecated GradScaler warning
    scaler = torch.amp.GradScaler("cuda")

    print("Model ready — DeepLabV3+ MobileNetV3")

    # ── Loss & Optimiser ───────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {
        "train_loss": [], "val_loss": [],
        "train_iou":  [], "val_iou":  [],
        "train_dice": [], "val_dice": []
    }
    best_val_iou = 0.0

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──────────────────────────────────────────
        model.train()
        t_loss, t_iou, t_dice = [], [], []

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE,  non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images)["out"]
                loss    = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            t_loss.append(loss.item())
            t_iou.append(compute_iou(preds, masks))
            t_dice.append(compute_dice(preds, masks))

        # ── Validate ───────────────────────────────────────
        model.eval()
        v_loss, v_iou, v_dice = [], [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks  = masks.to(DEVICE,  non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(images)["out"]
                    loss    = criterion(outputs, masks)

                preds = outputs.argmax(dim=1)
                v_loss.append(loss.item())
                v_iou.append(compute_iou(preds, masks))
                v_dice.append(compute_dice(preds, masks))

        scheduler.step()

        history["train_loss"].append(np.mean(t_loss))
        history["val_loss"].append(np.mean(v_loss))
        history["train_iou"].append(np.mean(t_iou))
        history["val_iou"].append(np.mean(v_iou))
        history["train_dice"].append(np.mean(t_dice))
        history["val_dice"].append(np.mean(v_dice))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {history['train_loss'][-1]:.4f}/{history['val_loss'][-1]:.4f} | "
            f"IoU {history['train_iou'][-1]:.4f}/{history['val_iou'][-1]:.4f} | "
            f"Dice {history['train_dice'][-1]:.4f}/{history['val_dice'][-1]:.4f}"
        )

        if history["val_iou"][-1] > best_val_iou:
            best_val_iou = history["val_iou"][-1]
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✔ Best model saved (val IoU = {best_val_iou:.4f})")

    # ── Save & Plot ────────────────────────────────────────
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["train_iou"], label="Train")
    axes[1].plot(history["val_iou"],   label="Val")
    axes[1].set_title("IoU"); axes[1].legend()

    axes[2].plot(history["train_dice"], label="Train")
    axes[2].plot(history["val_dice"],   label="Val")
    axes[2].set_title("Dice"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    print(f"\nBest Val IoU : {best_val_iou:.4f}")
    print("Saved: best_model.pth, training_curves.png, training_history.json")