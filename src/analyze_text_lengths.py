import pandas as pd
import numpy as np
from pathlib import Path


print("=" * 70)
print("CyberShield - Text Length Analysis")
print("=" * 70)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


# --------------------------------------------------
# Combine only for analysis
# --------------------------------------------------

df = pd.concat(
    [train, validation, test],
    ignore_index=True
)


# --------------------------------------------------
# Calculate word counts
# --------------------------------------------------

df["word_count"] = (
    df["text"]
    .astype(str)
    .str.split()
    .str.len()
)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

print("\nWord-count statistics:")
print(df["word_count"].describe())


# --------------------------------------------------
# Important percentiles
# --------------------------------------------------

percentiles = [50, 75, 90, 95, 99, 99.5, 100]

print("\nImportant percentiles:")

for p in percentiles:
    value = np.percentile(df["word_count"], p)
    print(f"{p:>5}% : {value:.0f} words")


# --------------------------------------------------
# Very long texts
# --------------------------------------------------

print("\nTexts longer than 100 words:")
print((df["word_count"] > 100).sum())

print("\nTexts longer than 150 words:")
print((df["word_count"] > 150).sum())

print("\nTexts longer than 200 words:")
print((df["word_count"] > 200).sum())


print("\n" + "=" * 70)
print("Text length analysis complete")
print("=" * 70)