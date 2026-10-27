import random
from pathlib import Path
from PIL import Image, ImageDraw

random.seed(42)

OUT_DIR = Path("data/dummy_dataset")
CLASSES = ["kubus", "silinder", "botol"]
JUMLAH = {"train": 30, "val": 8, "test": 8}  # gambar per kelas


def gambar_objek(kelas, size=224):
    bg = tuple(random.randint(180, 255) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    warna = tuple(random.randint(0, 150) for _ in range(3))
    cx = size // 2 + random.randint(-25, 25)
    cy = size // 2 + random.randint(-25, 25)

    if kelas == "kubus":
        s = random.randint(35, 55)
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=warna)
    elif kelas == "silinder":
        w, h = random.randint(30, 45), random.randint(40, 55)
        d.rectangle([cx - w, cy - h, cx + w, cy + h], fill=warna)
        d.ellipse([cx - w, cy - h - 12, cx + w, cy - h + 12], fill=warna)
        d.ellipse([cx - w, cy + h - 12, cx + w, cy + h + 12], fill=warna)
    else:  # botol
        w, h = random.randint(25, 35), random.randint(40, 55)
        d.rectangle([cx - w, cy - h, cx + w, cy + h], fill=warna)
        d.rectangle([cx - 10, cy - h - 30, cx + 10, cy - h], fill=warna)
    return img


for split, n in JUMLAH.items():
    for kelas in CLASSES:
        folder = OUT_DIR / split / kelas
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            gambar_objek(kelas).save(folder / f"{kelas}_{i:03d}.jpg")

print("Dataset dummy selesai dibuat di:", OUT_DIR)