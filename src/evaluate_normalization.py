import os
import json
import re
import sys

import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Allow importing normalize_text.py
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from normalize_text import (
    normalize_text,
    build_deletion_index,
)


# ============================================================
# Configuration
# ============================================================

MODEL_FILE = "models/baseline_bilstm.keras"
TOKENIZER_FILE = "models/tokenizer.json"

DATA_DIR = "data/obfuscated"
OUTPUT_DIR = "results/metrics"

MAX_SEQUENCE_LENGTH = 80
BATCH_SIZE = 64
THRESHOLD = 0.5

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load model
# ============================================================

print("Loading baseline model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded.")


# ============================================================
# Load tokenizer
# ============================================================

print("Loading tokenizer...")

with open(
    TOKENIZER_FILE,
    "r",
    encoding="utf-8"
) as f:

    tokenizer_data = json.load(f)

word_index = tokenizer_data[
    "token_to_index"
]

print(
    f"Vocabulary entries: {len(word_index)}"
)


# ============================================================
# Build vocabulary
# ============================================================

vocabulary = set(
    word.lower()
    for word in word_index.keys()
    if word.isalpha()
)

print(
    f"Alphabetic vocabulary words: "
    f"{len(vocabulary)}"
)


# ============================================================
# Build efficient deletion index
# ============================================================

print("Building deletion recovery index...")

deletion_index = build_deletion_index(
    vocabulary
)

print(
    f"Deletion index entries: "
    f"{len(deletion_index)}"
)


# ============================================================
# Tokenization
# ============================================================

def tokenize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text
    )

    return [
        word_index.get(
            token,
            1
        )
        for token in tokens
    ]


# ============================================================
# Evaluate one condition
# ============================================================

def evaluate_condition(
    filename
):

    path = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(path)

    original_texts = (
        df["text"]
        .astype(str)
        .tolist()
    )

    labels = (
        df["final_label"]
        .astype(int)
        .to_numpy()
    )

    # Normalize
    normalized_texts = [
        normalize_text(
            text,
            deletion_index
        )
        for text in original_texts
    ]

    # Tokenize
    sequences = [
        tokenize_text(text)
        for text in normalized_texts
    ]

    # Pad
    X = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        value=0
    )

    # Predict
    probabilities = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # Metrics
    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    condition = (
        filename
        .replace("test_", "")
        .replace(".csv", "")
    )

    return {
        "condition": condition,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Conditions
# ============================================================

files = [
    "test_leetspeak.csv",
    "test_punctuation.csv",
    "test_spacing.csv",
    "test_repetition.csv",
    "test_deletion.csv",
    "test_mixed.csv",
]


# ============================================================
# Run experiment
# ============================================================

results = []

print("\nStarting normalization experiment...\n")

for filename in files:

    print(
        f"Normalizing and evaluating: "
        f"{filename}"
    )

    result = evaluate_condition(
        filename
    )

    results.append(result)

    print(
        f"  Accuracy : "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{result['precision']:.4f}"
    )

    print(
        f"  Recall   : "
        f"{result['recall']:.4f}"
    )

    print(
        f"  F1-score : "
        f"{result['f1']:.4f}"
    )

    print()


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    results
)

output_file = os.path.join(
    OUTPUT_DIR,
    "normalization_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# Display
# ============================================================

print("=" * 70)
print("NORMALIZATION RESULTS")
print("=" * 70)

display_df = results_df.copy()

for column in [
    "accuracy",
    "precision",
    "recall",
    "f1"
]:

    display_df[column] *= 100


print(
    display_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print(
    "\nResults saved to:"
)

print(
    output_file
)