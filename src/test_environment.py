import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib
import nltk

print("===== CyberShield Environment =====")
print("NumPy       :", np.__version__)
print("Pandas      :", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("TensorFlow  :", tf.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("NLTK        :", nltk.__version__)

print("\nTensorFlow devices:")
print(tf.config.list_physical_devices())