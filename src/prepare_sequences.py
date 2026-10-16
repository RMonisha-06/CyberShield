import pandas as pd
import json
import re
import numpy as np
from pathlib import Path


print("=" * 70)
print("CyberShield - Preparing Sequences")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_FILE = PROJECT_ROOT / "models" / "tokenizer.json"

INPUT_FILES = {
    "train": PROJECT_ROOT / "data" / "processed" / "train.csv",
    "validation": PROJECT_ROOT / "data" / "processed" / "validation.csv",
    "test": PROJECT_ROOT / "data" / "processed" / "test.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"

MAX_SEQUENCE_LENGTH = 80
OOV_INDEX = 1


with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

token_to_index = tokenizer_data["token_to_index"]


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return re.findall(r"\w+|[^\w\s]", text)


def text_to_sequence(text):
    tokens = tokenize(text)

    sequence = [
        token_to_index.get(token, OOV_INDEX)
        for token in tokens
    ]

    # Truncate
    sequence = sequence[:MAX_SEQUENCE_LENGTH]

    # Post-padding
    sequence += [0] * (MAX_SEQUENCE_LENGTH - len(sequence))

    return sequence


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


for split_name, input_file in INPUT_FILES.items():

    print(f"\nProcessing {split_name}...")

    df = pd.read_csv(input_file)

    sequences = np.array(
        [text_to_sequence(text) for text in df["text"]],
        dtype=np.int32
    )

    labels = df["final_label"].to_numpy(dtype=np.int32)

    sequence_file = OUTPUT_DIR / f"{split_name}_X.npy"
    label_file = OUTPUT_DIR / f"{split_name}_y.npy"

    np.save(sequence_file, sequences)
    np.save(label_file, labels)

    print("Samples:", len(df))
    print("X shape:", sequences.shape)
    print("y shape:", labels.shape)

    print("First sequence:")
    print(sequences[0])

    print("First label:")
    print(labels[0])


print("\n" + "=" * 70)
print("Sequence preparation complete")
print("=" * 70)

print("\nSaved files:")
print(OUTPUT_DIR)