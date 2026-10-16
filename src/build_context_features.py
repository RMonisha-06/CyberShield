import pandas as pd
import numpy as np
from pathlib import Path
import sys

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "context_features"

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from context_features import extract_context_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_SEED = 42


# --------------------------------------------------
# Convert context dictionary to numeric vector
# --------------------------------------------------

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
        int(features["aggressive_cues"]),
    ]


# --------------------------------------------------
# Process a CSV file
# --------------------------------------------------

def process_file(input_path, output_path):

    df = pd.read_csv(input_path)

    print(f"\nProcessing: {input_path.name}")
    print(f"Rows: {len(df)}")

    feature_vectors = []

    for text in df["text"].fillna(""):

        features = extract_context_features(text)

        vector = features_to_vector(features)

        feature_vectors.append(vector)

    X_context = np.array(
        feature_vectors,
        dtype=np.float32
    )

    np.save(
        output_path,
        X_context
    )

    print("Saved:", output_path)
    print("Shape:", X_context.shape)


# --------------------------------------------------
# Main
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


process_file(
    DATA_DIR / "train.csv",
    OUTPUT_DIR / "train_context.npy"
)

process_file(
    DATA_DIR / "validation.csv",
    OUTPUT_DIR / "validation_context.npy"
)

process_file(
    DATA_DIR / "test.csv",
    OUTPUT_DIR / "test_context.npy"
)


print("\n" + "=" * 60)
print("Context feature generation complete")
print("=" * 60)