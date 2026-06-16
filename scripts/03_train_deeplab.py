import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50


IMAGE_SIZE = 512


def read_split(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def find_image(image_dir, stem):
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"找不到图片: {stem}")


class ShadowDataset(Dataset):
    def __init__(self, image_dir, mask_dir, names, train=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.names = names
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        stem = self.names[idx]

        image_path = find_image(self.image_dir, stem)
        mask_path = self.mask_dir / f"{stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

        if self.train:
            if np.random.rand() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 轻微亮度扰动，让模型适应不同光照
            if np.random.rand() < 0.5:
                brightness = 0.8 + np.random.rand() * 0.4
                image = TF.adjust_brightness(image, brightness)

        image = TF.to_tensor(image)

        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def calc_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter

    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    dice = ((2 * inter + 1e-6) / (preds.sum(dim=1) + targets.sum(dim=1) + 1e-6)).mean().item()

    return iou, dice


def compute_pos_weight(mask_dir, names):
    pos = 0
    neg = 0

    for stem in names:
        mask_path = Path(mask_dir) / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = mask > 127

        pos += mask.sum()
        neg += mask.size - mask.sum()

    if pos == 0:
        return 1.0

    weight = neg / pos
    weight = float(np.clip(weight, 1.0, 10.0))
    return weight


def build_model():
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    split_dir = Path(args.split_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_names = read_split(split_dir / "train.txt")
    val_names = read_split(split_dir / "val.txt")

    train_dataset = ShadowDataset(image_dir, mask_dir, train_names, train=True)
    val_dataset = ShadowDataset(image_dir, mask_dir, val_names, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    pos_weight_value = compute_pos_weight(mask_dir, train_names)
    print("pos_weight:", pos_weight_value)

    pos_weight = torch.tensor([pos_weight_value], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_model()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits = outputs["out"]

            loss_main = bce_loss(logits, masks) + dice_loss(logits, masks)

            if "aux" in outputs and outputs["aux"] is not None:
                aux_logits = outputs["aux"]
                loss_aux = bce_loss(aux_logits, masks) + dice_loss(aux_logits, masks)
                loss = loss_main + 0.4 * loss_aux
            else:
                loss = loss_main

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} val"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                logits = outputs["out"]

                loss = bce_loss(logits, masks) + dice_loss(logits, masks)
                iou, dice = calc_metrics(logits, masks)

                val_loss += loss.item()
                val_iou += iou
                val_dice += dice

        val_loss /= max(len(val_loader), 1)
        val_iou /= max(len(val_loader), 1)
        val_dice /= max(len(val_loader), 1)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_iou={val_iou:.4f}, "
            f"val_dice={val_dice:.4f}"
        )

        torch.save(model.state_dict(), save_dir / "last.pth")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_dir / "best.pth")
            print("保存 best model")

    print("训练完成，best val IoU:", best_val_iou)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", default="dataset/images")
    parser.add_argument("--mask_dir", default="dataset/masks")
    parser.add_argument("--split_dir", default="dataset/splits")
    parser.add_argument("--save_dir", default="runs/deeplab_shadow")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train(args)