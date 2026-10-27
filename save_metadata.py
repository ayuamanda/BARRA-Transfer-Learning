import os
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from load_data import test_ds, class_names, DATA_DIR, IMG_SIZE

RESULTS_DIR = "results"

model = tf.keras.models.load_model(f"{RESULTS_DIR}/best_model.keras")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)

# 1. Urutan kelas
with open(f"{RESULTS_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

# 2. Ringkasan evaluasi
metrics = {
    "dataset_dir": str(DATA_DIR),
    "model": "MobileNetV2",
    "pretrained_weights": "ImageNet",
    "image_size": list(IMG_SIZE),
    "classes": class_names,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
}
with open(f"{RESULTS_DIR}/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Tersimpan: class_names.json dan evaluation_metrics.json")
print(json.dumps(metrics, indent=4))