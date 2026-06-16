"""
使用方式：

python scripts/05_predict_shadow_with_road.py ^
  --input_dir dataset/images ^
  --shadow_weight runs/deeplab_shadow/best.pth ^
  --output_dir runs/predict_with_road ^
  --shadow_threshold 0.5 ^
  --road_dilate 45 ^
  --road_close 55

说明：
1. shadow 模型：你自己训练好的 DeepLabV3。
2. road 模型：现成 HuggingFace SegFormer Cityscapes 模型。
3. 最终结果：final_shadow = raw_shadow & refined_road_mask。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


IMAGE_SIZE = 512


# -----------------------------
# 1. 你训练好的影子分割模型
# -----------------------------
def build_shadow_model():
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=True
    )

    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    return model


# -----------------------------
# 2. 现成道路分割模型 SegFormer
# -----------------------------
def load_road_model(model_name, device):
    print(f"加载道路分割模型: {model_name}")

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return processor, model


# -----------------------------
# 3. 工具函数
# -----------------------------
def sorted_image_paths(input_dir):
    input_dir = Path(input_dir)

    paths = (
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg")) +
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.bmp"))
    )

    def sort_key(p):
        return int(p.stem) if p.stem.isdigit() else p.stem

    return sorted(paths, key=sort_key)


def overlay_mask(image_bgr, mask, color=(0, 0, 255), alpha=0.45):
    """
    color 默认红色，BGR 格式。
    """
    overlay = image_bgr.copy()
    mask_bool = mask > 0

    if not np.any(mask_bool):
        return overlay

    color_img = np.zeros_like(image_bgr)
    color_img[:, :] = color

    blended = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        color_img,
        alpha,
        0
    )

    overlay[mask_bool] = blended[mask_bool]
    return overlay


def save_prob_heatmap(prob, out_path):
    prob_vis = (prob * 255).clip(0, 255).astype(np.uint8)
    prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), prob_vis)


def keep_largest_component(mask):
    """
    可选：只保留最大连通道路区域。
    有些图里 road 会有小碎片，用这个可以过滤。
    """
    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        return (mask * 255).astype(np.uint8)

    max_area = 0
    max_idx = 1

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_idx = i

    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == max_idx] = 255
    return out


def refine_road_mask(road_mask, close_size=55, dilate_size=45, keep_largest=False):
    """
    道路 mask 后处理。

    重点：
    - close：填道路内部洞，比如阴影、车道线、路面小块漏分。
    - dilate：把 road mask 放宽，避免把真实道路边缘影子裁掉。
    """
    road_mask = (road_mask > 0).astype(np.uint8) * 255

    if keep_largest:
        road_mask = keep_largest_component(road_mask)

    if close_size > 0:
        k = int(close_size)
        if k % 2 == 0:
            k += 1
        kernel_close = np.ones((k, k), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)

    if dilate_size > 0:
        k = int(dilate_size)
        if k % 2 == 0:
            k += 1
        kernel_dilate = np.ones((k, k), np.uint8)
        road_mask = cv2.dilate(road_mask, kernel_dilate, iterations=1)

    return road_mask


# -----------------------------
# 4. 影子预测
# -----------------------------
def predict_shadow(shadow_model, image_pil, device, threshold=0.5, normalize=False):
    ori_w, ori_h = image_pil.size

    image_resized = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image_tensor = TF.to_tensor(image_resized)

    # 注意：
    # 如果你训练 shadow 模型时没有加 normalize，这里就不要加。
    # 当前默认 normalize=False，兼容你已经训练好的 best.pth。
    if normalize:
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = shadow_model(image_tensor)["out"]
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob = cv2.resize(prob, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
    raw_mask = (prob > threshold).astype(np.uint8) * 255

    return prob, raw_mask


# -----------------------------
# 5. 道路预测
# -----------------------------
def predict_road(road_processor, road_model, image_pil, device, road_label_ids):
    """
    Cityscapes 常用类别中：
    0 = road
    1 = sidewalk

    默认只用 road=0。
    如果你想把路肩/人行道也放进来，可以运行时传：
    --road_labels 0 1
    """
    ori_w, ori_h = image_pil.size

    inputs = road_processor(images=image_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = road_model(**inputs)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=False
        )

        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    road_mask = np.zeros((ori_h, ori_w), dtype=np.uint8)

    for label_id in road_label_ids:
        road_mask[pred == label_id] = 255

    return road_mask, pred


# -----------------------------
# 6. 主流程
# -----------------------------
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    raw_shadow_dir = output_dir / "01_raw_shadow_masks"
    road_mask_dir = output_dir / "02_road_masks"
    road_refined_dir = output_dir / "03_road_masks_refined"
    final_mask_dir = output_dir / "04_final_shadow_masks"
    shadow_heatmap_dir = output_dir / "05_shadow_heatmaps"
    road_overlay_dir = output_dir / "06_road_overlays"
    final_overlay_dir = output_dir / "07_final_overlays"
    compare_dir = output_dir / "08_compare"

    for d in [
        raw_shadow_dir,
        road_mask_dir,
        road_refined_dir,
        final_mask_dir,
        shadow_heatmap_dir,
        road_overlay_dir,
        final_overlay_dir,
        compare_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # 加载你训练好的影子模型
    print("加载影子模型:", args.shadow_weight)
    shadow_model = build_shadow_model()
    state = torch.load(args.shadow_weight, map_location=device)
    shadow_model.load_state_dict(state)
    shadow_model.to(device)
    shadow_model.eval()

    # 加载现成道路模型
    road_processor, road_model = load_road_model(args.road_model, device)

    image_paths = sorted_image_paths(input_dir)
    print("图片数量:", len(image_paths))

    if len(image_paths) == 0:
        print("没有找到图片，请检查 input_dir:", input_dir)
        return

    for image_path in tqdm(image_paths):
        image_pil = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        stem = image_path.stem

        # 1. 原始影子预测，不加道路约束
        shadow_prob, raw_shadow_mask = predict_shadow(
            shadow_model=shadow_model,
            image_pil=image_pil,
            device=device,
            threshold=args.shadow_threshold,
            normalize=args.shadow_normalize
        )

        # 2. 现成道路模型预测 road mask
        road_mask, road_pred = predict_road(
            road_processor=road_processor,
            road_model=road_model,
            image_pil=image_pil,
            device=device,
            road_label_ids=args.road_labels
        )

        # 3. 道路 mask 放宽，避免道路阴影被裁掉
        road_mask_refined = refine_road_mask(
            road_mask,
            close_size=args.road_close,
            dilate_size=args.road_dilate,
            keep_largest=args.keep_largest_road
        )

        # 4. 最终道路影子
        final_shadow_mask = cv2.bitwise_and(raw_shadow_mask, road_mask_refined)

        # 5. 可视化
        raw_shadow_overlay = overlay_mask(
            image_bgr,
            raw_shadow_mask,
            color=(0, 0, 255),
            alpha=0.45
        )

        road_overlay = overlay_mask(
            image_bgr,
            road_mask_refined,
            color=(0, 255, 0),
            alpha=0.35
        )

        final_overlay = overlay_mask(
            image_bgr,
            final_shadow_mask,
            color=(0, 0, 255),
            alpha=0.45
        )

        # 对比图：左 原图；中 道路 mask；右 最终影子
        compare = np.concatenate(
            [
                image_bgr,
                road_overlay,
                final_overlay
            ],
            axis=1
        )

        # 6. 保存
        cv2.imwrite(str(raw_shadow_dir / f"{stem}.png"), raw_shadow_mask)
        cv2.imwrite(str(road_mask_dir / f"{stem}.png"), road_mask)
        cv2.imwrite(str(road_refined_dir / f"{stem}.png"), road_mask_refined)
        cv2.imwrite(str(final_mask_dir / f"{stem}.png"), final_shadow_mask)

        save_prob_heatmap(shadow_prob, shadow_heatmap_dir / f"{stem}.jpg")

        cv2.imwrite(str(road_overlay_dir / f"{stem}.jpg"), road_overlay)
        cv2.imwrite(str(final_overlay_dir / f"{stem}.jpg"), final_overlay)
        cv2.imwrite(str(compare_dir / f"{stem}.jpg"), compare)

    print("预测完成")
    print("原始影子 mask:", raw_shadow_dir)
    print("道路 mask:", road_mask_dir)
    print("修正道路 mask:", road_refined_dir)
    print("最终影子 mask:", final_mask_dir)
    print("最终 overlay:", final_overlay_dir)
    print("对比图:", compare_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default="dataset/images")
    parser.add_argument("--shadow_weight", default="runs/deeplab_shadow/best.pth")
    parser.add_argument("--output_dir", default="runs/predict_with_road")

    parser.add_argument(
        "--road_model",
        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    )

    parser.add_argument("--shadow_threshold", type=float, default=0.5)

    # 如果你之前训练 shadow 模型时加了 ImageNet normalize，这里才加这个参数。
    parser.add_argument("--shadow_normalize", action="store_true")

    # Cityscapes:
    # 0 = road
    # 1 = sidewalk
    # 默认只取 road。
    parser.add_argument("--road_labels", type=int, nargs="+", default=[0])

    # 道路 mask 后处理参数
    parser.add_argument("--road_close", type=int, default=55)
    parser.add_argument("--road_dilate", type=int, default=45)
    parser.add_argument("--keep_largest_road", action="store_true")

    args = parser.parse_args()
    predict(args)