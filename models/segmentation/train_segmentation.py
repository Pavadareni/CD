"""
Training script for U-Net crack segmentation model
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import config
from models.segmentation.unet import UNet, CombinedLoss


# =====================================================
# DATASET
# =====================================================

class CrackDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


# =====================================================
# AUGMENTATIONS
# =====================================================

def get_transforms(train=True):

    if train:
        return A.Compose([
            A.Resize(*config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(*config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


# =====================================================
# METRIC
# =====================================================

def calculate_iou(pred, target, threshold=0.5):

    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    return ((intersection + 1e-6) / (union + 1e-6)).item()


# =====================================================
# TRAIN / VAL
# =====================================================

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss, total_iou = 0, 0

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += calculate_iou(preds, masks)

    return total_loss / len(loader), total_iou / len(loader)


# =====================================================
# MAIN TRAIN FUNCTION
# =====================================================

def train_segmentation_model(
    train_image_dir,
    train_mask_dir,
    num_epochs=None,
    batch_size=None,
    learning_rate=None,
    save_path=None
):

    num_epochs = num_epochs or config.UNET_CONFIG["num_epochs"]
    batch_size = batch_size or config.UNET_CONFIG["batch_size"]
    learning_rate = learning_rate or config.UNET_CONFIG["learning_rate"]

    # ✅ FORCE PATH
    save_path = save_path or os.path.join("models", "saved", "unet_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = CrackDataset(
        train_image_dir,
        train_mask_dir,
        transform=get_transforms(train=True)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    model = UNet(
        in_channels=config.UNET_CONFIG["in_channels"],
        out_channels=config.UNET_CONFIG["out_channels"],
        init_features=config.UNET_CONFIG["init_features"]
    ).to(device)

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):

        loss, iou = train_epoch(model, loader, criterion, optimizer, device)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss:.4f}  IoU: {iou:.4f}")

    # =====================================================
    # ALWAYS SAVE
    # =====================================================

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\n✓ Model saved at: {save_path}")
    print("Training complete!")

    return model


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    
    train_segmentation_model(
        train_image_dir=os.path.join(config.SYNTHETIC_DATA_DIR, "images"),
        train_mask_dir=os.path.join(config.SYNTHETIC_DATA_DIR, "masks")
    )