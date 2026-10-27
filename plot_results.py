import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")  # simpan ke file, tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from load_data import test_ds, class_names

RESULTS_DIR = "results"

# 1. Grafik training (dari training_history.csv)
hist = pd.read_csv(f"{RESULTS_DIR}/training_history.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(hist["accuracy"], label="Training")
axes[0].plot(hist["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(hist["loss"], label="Training")
axes[1].plot(hist["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan:", f"{RESULTS_DIR}/training_history.png")

# 2. Confusion matrix (pada data test)
model = tf.keras.models.load_model(f"{RESULTS_DIR}/best_model.keras")

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", values_format="d", colorbar=False
)
ax.set_title("BARRA Object Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan:", f"{RESULTS_DIR}/confusion_matrix.png")