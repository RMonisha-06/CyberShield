import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OBFUSCATED_DIR = PROJECT_ROOT / "data" / "obfuscated"
CONTEXT_DIR = PROJECT_ROOT / "data" / "processed" / "context_features"
MODEL_PATH = PROJECT_ROOT / "models" / "context_bilstm.keras"

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from normalize_text import normalize_text
from context_features import extract_context_features


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEQUENCE_LENGTH = 80
OOV_INDEX = 1

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

print("=" * 70)
print("CyberShield - Context Model Obfuscation Evaluation")
print("=" * 70)

print("\nLoading tokenizer...")

import json

with open(
    PROJECT_ROOT / "models" / "tokenizer.json",
    "r",
    encoding="utf-8"
) as f:
    tokenizer_data = json.load(f)

token_to_index = tokenizer_data["token_to_index"]

print(
    "Vocabulary size:",
    len(token_to_index)
)


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

import re


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return re.findall(
        r"\w+|[^\w\s]",
        text
    )


def text_to_sequence(text):
    tokens = tokenize(text)

    sequence = []

    for token in tokens:
        sequence.append(
            token_to_index.get(
                token,
                OOV_INDEX
            )
        )

    return sequence


def pad_sequence(sequence):
    sequence = sequence[:SEQUENCE_LENGTH]

    if len(sequence) < SEQUENCE_LENGTH:
        sequence += [0] * (
            SEQUENCE_LENGTH - len(sequence)
        )

    return sequence


# ---------------------------------------------------------
# Context feature conversion
# ---------------------------------------------------------

def features_to_vector(features):

    return [
        features["word_count"],
        features["character_count"],
        int(features["emoji_detected"]),
        features["emoji_count"],
        len(features["negative_cues"]),
        len(features["positive_cues"]),
        int(features["repeated_punctuation"]),
        features["exclamation_count"],
        features["question_count"],
        features["uppercase_ratio"],
        int(features["aggressive_cues"])
    ]


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\nLoading context-enhanced model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

def evaluate_dataset(
    csv_path,
    normalized=False
):

    df = pd.read_csv(csv_path)

    texts = df["text"].fillna("").astype(str).tolist()
    y_true = df["final_label"].astype(int).to_numpy()

    processed_texts = []

    context_vectors = []
    sequences = []

    for text in texts:

        if normalized:
            processed = normalize_text(text)
        else:
            processed = text

        processed_texts.append(processed)

        # Context features from the processed text
        features = extract_context_features(
            processed
        )

        context_vectors.append(
            features_to_vector(features)
        )

        sequence = text_to_sequence(
            processed
        )

        sequences.append(
            pad_sequence(sequence)
        )

    X_text = np.array(
        sequences,
        dtype=np.int32
    )

    X_context = np.array(
        context_vectors,
        dtype=np.float32
    )

    probabilities = model.predict(
        {
            "text_input": X_text,
            "context_input": X_context
        },
        verbose=0
    ).ravel()

    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1]
    }


# ---------------------------------------------------------
# Evaluate all datasets
# ---------------------------------------------------------

obfuscation_types = [
    "clean",
    "leetspeak",
    "punctuation",
    "spacing",
    "repetition",
    "deletion",
    "mixed"
]


results = []


for obfuscation_type in obfuscation_types:

    filename = (
        f"test_{obfuscation_type}.csv"
    )

    csv_path = OBFUSCATED_DIR / filename

    print(
        f"\nEvaluating: {obfuscation_type}"
    )

    # Without normalization
    raw_result = evaluate_dataset(
        csv_path,
        normalized=False
    )

    raw_result["obfuscation"] = obfuscation_type
    raw_result["normalization"] = "No"

    results.append(raw_result)

    # With normalization
    normalized_result = evaluate_dataset(
        csv_path,
        normalized=True
    )

    normalized_result["obfuscation"] = obfuscation_type
    normalized_result["normalization"] = "Yes"

    results.append(normalized_result)


# ---------------------------------------------------------
# Create results table
# ---------------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = results_df[
    [
        "obfuscation",
        "normalization",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp"
    ]
]


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_path = (
    RESULTS_DIR /
    "context_obfuscation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("CONTEXT-ENHANCED OBFUSCATION RESULTS")
print("=" * 70)

display_df = results_df.copy()

for column in [
    "accuracy",
    "precision",
    "recall",
    "f1"
]:
    display_df[column] = (
        display_df[column] * 100
    ).round(2)

print(
    display_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(output_path)

print("\n" + "=" * 70)
print("Obfuscation evaluation complete")
print("=" * 70)