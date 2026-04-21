"""
PHASE 2 — Fine-tune DeepLabV3 on your labeled F1 data
======================================================
Fully debugged and fixed version.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import os
import json

BASE_DIR = Path(__file__).resolve().parent

# CONFIG
TRAIN_IMAGES_DIR = BASE_DIR / "data" / "roboflow_export" / "train"
ANNOTATIONS_PATH = BASE_DIR / "data" / "roboflow_export" / "train" / "_annotations.coco.json"
MODEL_SAVE_PATH  = BASE_DIR / "models" / "finetuned_model.pth"
NUM_CLASSES      = 3
NUM_EPOCHS       = 20
BATCH_SIZE       = 4
LEARNING_RATE    = 1e-4
VAL_SPLIT        = 0.2


class F1SegmentationDataset(Dataset):
    def __init__(self, images_dir, annotations_path, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        with open(annotations_path, encoding="utf-8") as f:
            coco = json.load(f)

        self.images     = coco["images"]
        self.categories = {c["id"]: c["name"] for c in coco["categories"]}

        self.class_map = {}
        for cat_id, cat_name in self.categories.items():
            if "track" in cat_name.lower() or "surface" in cat_name.lower():
                self.class_map[cat_id] = 1
            elif "kerb" in cat_name.lower() or "curb" in cat_name.lower():
                self.class_map[cat_id] = 2
            else:
                self.class_map[cat_id] = 0

        print(f"Categories: {self.categories}")
        print(f"Class map:  {self.class_map}")

        self.ann_by_image = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.ann_by_image:
                self.ann_by_image[img_id] = []
            self.ann_by_image[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.images_dir / img_info["file_name"]
        image    = Image.open(img_path).convert("RGB")
        w, h     = image.size

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for ann in self.ann_by_image.get(img_info["id"], []):
            class_idx = self.class_map.get(ann["category_id"], 0)

            seg_data = ann.get("segmentation", [])

            # Handle both list-of-lists and RLE formats
            if isinstance(seg_data, dict):
                # RLE format — skip, we only handle polygon format
                continue

            for seg in seg_data:
                # Skip if not a list of numbers
                if not isinstance(seg, list):
                    continue
                if len(seg) < 6:
                    continue
                try:
                    polygon = [(int(float(seg[i])), int(float(seg[i+1])))
                               for i in range(0, len(seg), 2)]
                    draw.polygon(polygon, fill=class_idx)
                except (ValueError, TypeError, IndexError):
                    continue

        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))

        if self.transforms:
            image = self.transforms(image)

        return image, mask_tensor


def load_model_for_finetuning(num_classes):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model   = deeplabv3_resnet50(weights=weights)
    model.classifier[4]     = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def get_transforms():
    return T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        outputs    = model(images)
        logits     = outputs["out"]
        aux_logits = outputs["aux"]

        masks_resized = F.interpolate(
            masks.unsqueeze(1).float(),
            size=logits.shape[2:],
            mode="nearest"
        ).squeeze(1).long()

        main_loss = criterion(logits, masks_resized)
        aux_loss  = criterion(aux_logits, masks_resized)
        loss      = main_loss + 0.4 * aux_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 5 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"Epoch {epoch} - avg train loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            logits  = outputs["out"]

            masks_resized = F.interpolate(
                masks.unsqueeze(1).float(),
                size=logits.shape[2:],
                mode="nearest"
            ).squeeze(1).long()

            loss        = criterion(logits, masks_resized)
            total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"              val loss: {avg_loss:.4f}")
    return avg_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cpu":
        print("Warning: CPU training is slow (~30 min/epoch)")
        print("Strongly recommend Google Colab free GPU instead\n")

    if not ANNOTATIONS_PATH.exists():
        print(f"Not found: {ANNOTATIONS_PATH}")
        print("Make sure data/roboflow_export/train/_annotations.coco.json exists")
        return

    transforms   = get_transforms()
    full_dataset = F1SegmentationDataset(TRAIN_IMAGES_DIR, ANNOTATIONS_PATH, transforms)
    print(f"\nTotal images: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("No images found — check TRAIN_IMAGES_DIR path")
        return

    val_size   = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size} | Val: {val_size}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = load_model_for_finetuning(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print("Starting training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n-- Epoch {epoch}/{NUM_EPOCHS} --")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
            print(f"  Best model saved -> {MODEL_SAVE_PATH}")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("Next: replace 3_segment_frames.py with phase2_segment_finetuned.py and re-run pipeline")


if __name__ == "__main__":
    main()