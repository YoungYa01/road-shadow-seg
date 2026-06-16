from pathlib import Path
from PIL import Image
import numpy as np

image_dir = Path("dataset/images")
mask_dir = Path("dataset/masks")

image_paths = sorted(
    list(image_dir.glob("*.jpg")) +
    list(image_dir.glob("*.jpeg")) +
    list(image_dir.glob("*.png")),
    key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
)

ok = True

for image_path in image_paths:
    stem = image_path.stem
    mask_path = mask_dir / f"{stem}.png"

    if not mask_path.exists():
        print(f"缺少 mask: {mask_path}")
        ok = False
        continue

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    if image.size != mask.size:
        print(f"尺寸不一致: {image_path.name}, image={image.size}, mask={mask.size}")
        ok = False

    mask_arr = np.array(mask)
    unique_values = np.unique(mask_arr)

    if not set(unique_values.tolist()).issubset({0, 255}):
        print(f"mask 不是标准二值图: {mask_path.name}, values={unique_values[:10]}")
        ok = False

print("图片数量:", len(image_paths))
print("检查结果:", "通过" if ok else "有问题")