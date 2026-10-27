import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from load_data import test_ds, class_names

RESULTS_DIR = "results"

model = tf.keras.models.load_model(f"{RESULTS_DIR}/best_model.keras")

# 1. Akurasi dan loss pada data test
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_acc * 100:.2f}%")

# 2. Prediksi untuk seluruh data test
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
y_prob = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# 3. Classification report (precision, recall, F1 per kelas)
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)
report_df = pd.DataFrame(report).transpose()
print("\nClassification report:")
print(report_df.round(3))

report_df.to_csv(f"{RESULTS_DIR}/classification_report.csv")
print(f"\nTersimpan: {RESULTS_DIR}/classification_report.csv")