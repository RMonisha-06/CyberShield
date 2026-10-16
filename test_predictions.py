import sys
sys.path.insert(0, "src")

import json
import re
import numpy as np
import tensorflow as tf

from normalize_text import normalize_text

MODEL_PATH = "models/baseline_bilstm_fixed.keras"
TOKENIZER_PATH = "models/tokenizer.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

vocab = tokenizer_data["token_to_index"]


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return re.findall(r"\w+|[^\w\s]", text)


def predict(text):
    normalized = normalize_text(text)
    tokens = tokenize(normalized)

    sequence = [vocab.get(token, 1) for token in tokens]
    sequence = (sequence + [0] * 80)[:80]

    probability = float(
        model.predict(np.array([sequence]), verbose=0)[0][0]
    )

    prediction = (
        "Cyberbullying"
        if probability >= 0.5
        else "Non-Cyberbullying"
    )

    confidence = (
        probability if probability >= 0.5
        else 1 - probability
    ) * 100

    print(f"Input:       {text}")
    print(f"Normalized:  {normalized}")
    print(f"Probability: {probability:.4f}")
    print(f"Prediction:  {prediction}")
    print(f"Confidence:  {confidence:.2f}%")
    print("-" * 60)


tests = [
    "hey you stupid",
    "h3y y00 s1u7",
    "you are stupid",
    "you are an idiot",
]

for text in tests:
    predict(text)
