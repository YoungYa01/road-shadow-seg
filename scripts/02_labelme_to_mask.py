import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


image_dir = Path("dataset/images")
json_dir = Path("dataset/labels_json")
mask_dir = Path("dataset/masks")
mask_dir.mkdir(parents=True, exist_ok=True)


def find_image(stem):
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


json_paths = sorted(
    json_dir.glob("*.json"),
    key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
)

for json_path in tqdm(json_paths):
    stem = json_path.stem
    image_path = find_image(stem)

    if image_path is None:
        print(f"找不到对应图片: {stem}")
        continue

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    mask = np.zeros((height, width), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip().lower()

        if label != "shadow":
            continue

        points = shape.get("points", [])
        if len(points) < 3:
            continue

        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)

    cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

print("转换完成，mask 保存在:", mask_dir)