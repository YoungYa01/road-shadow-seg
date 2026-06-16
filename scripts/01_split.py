from pathlib import Path
from sklearn.model_selection import train_test_split

image_dir = Path("dataset/images")
split_dir = Path("dataset/splits")
split_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    list(image_dir.glob("*.jpg")) +
    list(image_dir.glob("*.jpeg")) +
    list(image_dir.glob("*.png")),
    key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
)

names = [p.stem for p in image_paths]

train_names, temp_names = train_test_split(
    names,
    test_size=20,
    random_state=42,
    shuffle=True
)

val_names, test_names = train_test_split(
    temp_names,
    test_size=10,
    random_state=42,
    shuffle=True
)

def save_txt(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item + "\n")

save_txt(split_dir / "train.txt", train_names)
save_txt(split_dir / "val.txt", val_names)
save_txt(split_dir / "test.txt", test_names)

print("train:", len(train_names))
print("val:", len(val_names))
print("test:", len(test_names))
print("划分完成:", split_dir)