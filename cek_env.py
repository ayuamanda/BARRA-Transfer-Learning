import sys
import tensorflow as tf

print("Python      :", sys.version.split()[0])
print("Interpreter :", sys.executable)
print("TensorFlow  :", tf.__version__)
print("GPU         :", tf.config.list_physical_devices("GPU"))