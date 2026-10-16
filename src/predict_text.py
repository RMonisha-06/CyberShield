import json
import os
import re
import sys

import numpy as np
import tensorflow as tf
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"


# Allow importing normalize_text.py
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from normalize_text import normalize_text


MODEL_FILE = "models/baseline_bilstm.keras"
TOKENIZER_FILE = "models/tokenizer.json"

MAX_SEQUENCE_LENGTH = 80
OOV_INDEX = 1


def load_tokenizer():
    """
    Load the custom tokenizer created during research.
    """

    with open(
        TOKENIZER_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        tokenizer_data = json.load(f)

    return tokenizer_data["token_to_index"]


def tokenize_text(text, word_index):
    """
    Apply the exact same tokenization strategy
    used during the research pipeline.
    """

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

    sequence = [
        word_index.get(
            token,
            OOV_INDEX
        )
        for token in tokens
    ]

    # Truncate
    sequence = sequence[
        :MAX_SEQUENCE_LENGTH
    ]

    # Post-padding
    sequence += [
        0
    ] * (
        MAX_SEQUENCE_LENGTH
        - len(sequence)
    )

    return sequence


def predict(text):

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    print("Model loaded.")

    print("Loading tokenizer...")

    word_index = load_tokenizer()

    print("Tokenizer loaded.")

    # Build vocabulary for normalization
    vocabulary = set(
        word.lower()
        for word in word_index.keys()
        if word.isalpha()
    )

    # Normalize user input
    normalized_text = normalize_text(
        text,
        vocabulary=vocabulary
    )

    # Tokenize normalized text
    sequence = tokenize_text(
        normalized_text,
        word_index
    )

    # Convert to NumPy array
    input_data = np.array(
        [sequence],
        dtype=np.int32
    )

    # Predict
    probability = float(
        model.predict(
            input_data,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        label = "Cyberbullying"
    else:
        label = "Non-Cyberbullying"

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    print("\n" + "=" * 60)
    print("CYBERSHIELD PREDICTION")
    print("=" * 60)

    print(f"\nOriginal text:")
    print(text)

    print(f"\nNormalized text:")
    print(normalized_text)

    print(f"\nPrediction:")
    print(label)

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print(
        f"Raw model probability: "
        f"{probability:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":

    user_text = input(
        "\nEnter text to analyze: "
    )

    if not user_text.strip():

        print(
            "Please enter some text."
        )

    else:

        predict(user_text)