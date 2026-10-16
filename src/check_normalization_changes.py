import pandas as pd
import sys
import os

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from normalize_text import (
    normalize_text,
    build_deletion_index,
)

TOKENIZER_FILE = "models/tokenizer.json"
DATA_FILE = "data/obfuscated/test_mixed.csv"

import json

with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

word_index = tokenizer_data["token_to_index"]

vocabulary = set(
    word.lower()
    for word in word_index.keys()
    if word.isalpha()
)

deletion_index = build_deletion_index(vocabulary)

df = pd.read_csv(DATA_FILE)

changed = 0

print("\nExamples of normalization changes:\n")

for _, row in df.iterrows():

    original = str(row["text"])

    normalized = normalize_text(
        original,
        deletion_index
    )

    if original != normalized:

        print("ORIGINAL : ", original)
        print("NORMALIZED:", normalized)
        print("-" * 70)

        changed += 1

        if changed >= 30:
            break

print(f"\nDisplayed {changed} examples.")