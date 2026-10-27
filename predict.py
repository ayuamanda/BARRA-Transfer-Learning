import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from load_data import class_names, IMG_SIZE

RESULTS_DIR = "results"

if len(sys.argv) < 2:
    print("Cara pakai: python predict.py <path_gambar>")
    sys.exit(1)

image_path = sys.argv[1]

model = tf.keras.models.load_model(f"{RESULTS_DIR}/best_model.keras")

img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
img_batch = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)

probs = model.predict(img_batch, verbose=0)[0]
idx = int(np.argmax(probs))

print(f"\nGambar     : {image_path}")
print(f"Prediksi   : {class_names[idx]}")
print(f"Confidence : {probs[idx] * 100:.2f}%")
print("\nProbabilitas semua kelas:")
for name, p in zip(class_names, probs):
    print(f"  {name:10s} {p * 100:6.2f}%")