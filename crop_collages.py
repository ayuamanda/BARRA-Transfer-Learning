from pathlib import Path
from PIL import Image

COLLAGE_DIR = Path("data/collages")
OUT_DIR = Path("data/barra_dataset")
CLASSES = ["kubus", "silinder", "botol"]

COLS, ROWS = 4, 2      # grid kolase: 4 kolom x 2 baris
INSET = 0.03           # buang 3% pinggiran tiap sel (garis putih)
SPLIT_BY_INDEX = ["train", "train", "val", "test"]  # kolase ke-1..4

# Bersihkan hasil potongan lama supaya tidak menumpuk saat dijalankan ulang
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        folder = OUT_DIR / split / cls
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.png"):
            old.unlink()

for cls in CLASSES:
    files = sorted((COLLAGE_DIR / cls).glob("*.png"))
    assert len(files) == 4, f"{cls}: seharusnya 4 kolase, ditemukan {len(files)}"

    for file, split in zip(files, SPLIT_BY_INDEX):
        img = Image.open(file).convert("RGB")
        W, H = img.size
        cw, ch = W / COLS, H / ROWS

        for r in range(ROWS):
            for c in range(COLS):
                left = c * cw + INSET * cw
                right = (c + 1) * cw - INSET * cw
                top = r * ch + INSET * ch
                bottom = (r + 1) * ch - INSET * ch
                crop = img.crop((round(left), round(top), round(right), round(bottom)))
                crop.save(OUT_DIR / split / cls / f"{file.stem}_r{r + 1}c{c + 1}.png")

        print(f"{file.name} -> {split}")

print("\nSelesai. Hasil ada di:", OUT_DIR)