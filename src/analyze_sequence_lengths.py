import pandas as pd
import json
import re
import numpy as np
from pathlib import Path


print("=" * 70)
print("CyberShield - Sequence Length Analysis")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
TOKENIZER_FILE = PROJECT_ROOT / "models" / "tokenizer.json"


with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

token_to_index = tokenizer_data["token_to_index"]


train = pd.read_csv(TRAIN_FILE)


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return re.findall(r"\w+|[^\w\s]", text)


def convert_to_sequence(text):
    tokens = tokenize(text)

    sequence = []

    for token in tokens:
        sequence.append(token_to_index.get(token, 1))

    return sequence


lengths = []

for text in train["text"]:
    sequence = convert_to_sequence(text)
    lengths.append(len(sequence))


lengths = np.array(lengths)


print("\nNumber of training samples:", len(lengths))

print("\nSequence length statistics:")
print("Minimum:", lengths.min())
print("Maximum:", lengths.max())
print("Mean:", round(lengths.mean(), 2))
print("Median:", int(np.median(lengths)))

print("\nPercentiles:")

for percentile in [50, 75, 90, 95, 97, 98, 99, 99.5, 100]:
    value = np.percentile(lengths, percentile)
    print(f"{percentile:5}%: {value:.0f}")


print("\nSamples above possible sequence lengths:")

for limit in [40, 50, 60, 70, 80, 100]:
    count = np.sum(lengths > limit)
    percentage = count / len(lengths) * 100

    print(
        f">{limit:3} tokens: "
        f"{count:5} samples "
        f"({percentage:.2f}%)"
    )


print("\n" + "=" * 70)
print("Sequence length analysis complete")
print("=" * 70)