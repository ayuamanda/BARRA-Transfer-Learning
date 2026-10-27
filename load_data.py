import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # kurangi pesan cuda yang tidak penting

from pathlib import Path
import tensorflow as tf

# Dataset sintetis (buatan AI) hasil potongan kolase
DATA_DIR = Path("data/barra_dataset")

IMG_SIZE = (224, 224)   # ukuran input standar MobileNetV2
BATCH_SIZE = 16
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

class_names = train_ds.class_names

# class_names diteruskan agar urutan label val/test PASTI sama dengan train
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

if __name__ == "__main__":
    print("Nama kelas  :", class_names)
    print("Jumlah kelas:", len(class_names))
    for images, labels in train_ds.take(1):
        print("Bentuk batch gambar:", images.shape)
        print("Bentuk batch label :", labels.shape)
        print("Rentang piksel     :", float(tf.reduce_min(images)), "-", float(tf.reduce_max(images)))