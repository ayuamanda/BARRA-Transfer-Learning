import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from load_data import train_ds, val_ds
from build_model import build_model

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 20

model = build_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        f"{RESULTS_DIR}/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.CSVLogger(f"{RESULTS_DIR}/training_history.csv"),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("\nTraining selesai. File tersimpan di folder:", RESULTS_DIR)