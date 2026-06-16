"""
python scripts/04_predict.py --input_dir dataset/images --weight runs/deeplab_shadow/best.pth --use_roi
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50


IMAGE_SIZE = 512


def build_model():
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=True
    )

    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    return model


def make_trapezoid_roi(h, w):
    roi = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([
        [int(0.05 * w), h],
        [int(0.95 * w), h],
        [int(0.68 * w), int(0.42 * h)],
        [int(0.32 * w), int(0.42 * h)],
    ], dtype=np.int32)

    cv2.fillPoly(roi, [pts], 255)
    return roi


def overlay_mask(image_bgr, mask, alpha=0.45):
    overlay = image_bgr.copy()

    mask_bool = mask > 0

    # 如果这一张图没有预测出任何影子，直接返回原图
    if not np.any(mask_bool):
        return overlay

    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255

    # 先整张图做融合，再只取 mask 区域
    blended = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        red,
        alpha,
        0
    )

    overlay[mask_bool] = blended[mask_bool]

    return overlay


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    mask_out_dir = output_dir / "masks"
    overlay_out_dir = output_dir / "overlays"

    mask_out_dir.mkdir(parents=True, exist_ok=True)
    overlay_out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    state = torch.load(args.weight, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    image_paths = sorted(
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg")) +
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.bmp")),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    with torch.no_grad():
        for image_path in tqdm(image_paths):
            image_pil = Image.open(image_path).convert("RGB")
            ori_w, ori_h = image_pil.size

            image_resized = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)

            logits = model(image_tensor)["out"]
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prob = cv2.resize(prob, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
            mask = (prob > args.threshold).astype(np.uint8) * 255

            if args.use_roi:
                roi = make_trapezoid_roi(ori_h, ori_w)
                mask = cv2.bitwise_and(mask, roi)

            # 不再使用 cv2.imread，避免读取失败
            image_rgb = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            overlay = overlay_mask(image_bgr, mask)

            cv2.imwrite(str(mask_out_dir / f"{image_path.stem}.png"), mask)
            cv2.imwrite(str(overlay_out_dir / f"{image_path.stem}.jpg"), overlay)

    print("预测完成")
    print("mask 输出:", mask_out_dir)
    print("overlay 输出:", overlay_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default="dataset/images")
    parser.add_argument("--weight", default="runs/deeplab_shadow/best.pth")
    parser.add_argument("--output_dir", default="runs/predict")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_roi", action="store_true")

    args = parser.parse_args()
    predict(args)