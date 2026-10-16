import pandas as pd
import re
from collections import Counter
from pathlib import Path


print("=" * 70)
print("CyberShield - Vocabulary Analysis")
print("=" * 70)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"


# --------------------------------------------------
# Load training data ONLY
# --------------------------------------------------

train = pd.read_csv(TRAIN_FILE)

print("\nTraining samples:", len(train))


# --------------------------------------------------
# Simple tokenization for analysis
# --------------------------------------------------

def tokenize(text):
    text = str(text).lower()

    # Keep words, numbers and individual punctuation marks
    tokens = re.findall(r"\w+|[^\w\s]", text)

    return tokens


# --------------------------------------------------
# Count vocabulary
# --------------------------------------------------

counter = Counter()

total_tokens = 0

for text in train["text"]:
    tokens = tokenize(text)

    counter.update(tokens)

    total_tokens += len(tokens)


# --------------------------------------------------
# Results
# --------------------------------------------------

vocabulary_size = len(counter)

print("\nTotal tokens:", total_tokens)

print("Unique tokens:", vocabulary_size)


print("\nTop 30 most frequent tokens:")

for token, count in counter.most_common(30):
    print(f"{token!r:20} {count}")


# --------------------------------------------------
# Rare token statistics
# --------------------------------------------------

print("\nTokens appearing only once:")
print(sum(1 for count in counter.values() if count == 1))

print("\nTokens appearing <= 2 times:")
print(sum(1 for count in counter.values() if count <= 2))

print("\nTokens appearing <= 5 times:")
print(sum(1 for count in counter.values() if count <= 5))


print("\n" + "=" * 70)
print("Vocabulary analysis complete")
print("=" * 70)