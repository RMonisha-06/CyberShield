import os
import json
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load baseline model
# ============================================================

print("Loading baseline model...")

model = tf.keras.models.load_model(MODEL_FILE)

print("Model loaded.")


# ============================================================
# Load custom tokenizer
# ============================================================

print("Loading tokenizer...")

with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

word_index = tokenizer_data["token_to_index"]

print("Tokenizer loaded.")
print(f"Vocabulary entries: {len(word_index)}")


# ============================================================
# Tokenization
# Must match the tokenizer used during baseline preparation
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

    # 1 = OOV token
    return [
        word_index.get(token, 1)
        for token in tokens
    ]


# ============================================================
# Evaluate one dataset
# ============================================================

def evaluate_dataset(filename):

    path = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(path)

    texts = (
        df["text"]
        .astype(str)
        .tolist()
    )

    labels = (
        df["final_label"]
        .astype(int)
        .to_numpy()
    )

    # Tokenize
    sequences = [
        tokenize_text(text)
        for text in texts
    ]

    # Padding/truncation must match baseline
    X = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        value=0
    )

    # Model prediction
    probabilities = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    # Binary prediction
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
        "f1": f1
    }


# ============================================================
# Test conditions
# ============================================================

files = [
    "test_clean.csv",
    "test_leetspeak.csv",
    "test_punctuation.csv",
    "test_spacing.csv",
    "test_repetition.csv",
    "test_deletion.csv",
    "test_mixed.csv",
]


# ============================================================
# Run evaluation
# ============================================================

results = []

print("\nStarting evaluation...\n")

for filename in files:

    print(f"Evaluating: {filename}")

    result = evaluate_dataset(filename)

    results.append(result)

    print(
        f"  Accuracy : {result['accuracy']:.4f}"
    )

    print(
        f"  Precision: {result['precision']:.4f}"
    )

    print(
        f"  Recall   : {result['recall']:.4f}"
    )

    print(
        f"  F1-score : {result['f1']:.4f}"
    )

    print()


# ============================================================
# Create results table
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# Calculate performance drop from clean baseline
# ============================================================

clean_accuracy = results_df.loc[
    results_df["condition"] == "clean",
    "accuracy"
].iloc[0]

clean_f1 = results_df.loc[
    results_df["condition"] == "clean",
    "f1"
].iloc[0]


results_df["accuracy_drop"] = (
    clean_accuracy -
    results_df["accuracy"]
)

results_df["f1_drop"] = (
    clean_f1 -
    results_df["f1"]
)


# Convert to percentage points
results_df["accuracy_drop_pp"] = (
    results_df["accuracy_drop"] * 100
)

results_df["f1_drop_pp"] = (
    results_df["f1_drop"] * 100
)


# ============================================================
# Save results
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "obfuscation_baseline_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# Display final results
# ============================================================

print("=" * 70)
print("FINAL OBFUSCATION ROBUSTNESS RESULTS")
print("=" * 70)

display_df = results_df.copy()

display_df["accuracy"] = (
    display_df["accuracy"] * 100
)

display_df["precision"] = (
    display_df["precision"] * 100
)

display_df["recall"] = (
    display_df["recall"] * 100
)

display_df["f1"] = (
    display_df["f1"] * 100
)

print(
    display_df[
        [
            "condition",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "accuracy_drop_pp",
            "f1_drop_pp"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nResults saved to:")
print(output_file)