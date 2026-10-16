import json
import os
import sys

import pandas as pd

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from normalize_text import normalize_text


TOKENIZER_FILE = "models/tokenizer.json"

DATA_FILES = {
    "leetspeak": "data/obfuscated/test_leetspeak.csv",
    "punctuation": "data/obfuscated/test_punctuation.csv",
    "spacing": "data/obfuscated/test_spacing.csv",
    "repetition": "data/obfuscated/test_repetition.csv",
    "deletion": "data/obfuscated/test_deletion.csv",
    "mixed": "data/obfuscated/test_mixed.csv",
}


# --------------------------------------------------
# Load training vocabulary
# --------------------------------------------------

with open(
    TOKENIZER_FILE,
    "r",
    encoding="utf-8"
) as f:

    tokenizer_data = json.load(f)


word_index = tokenizer_data["token_to_index"]

vocabulary = set(
    word.lower()
    for word in word_index.keys()
    if word.isalpha()
)


print("=" * 70)
print("NORMALIZATION VERIFICATION")
print("=" * 70)

print(f"Vocabulary words: {len(vocabulary)}")


# --------------------------------------------------
# Verify each condition
# --------------------------------------------------

for condition, file_path in DATA_FILES.items():

    print("\n" + "=" * 70)
    print(f"CONDITION: {condition.upper()}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    total = len(df)
    changed = 0

    examples = []

    for _, row in df.iterrows():

        original = str(row["text"])

        normalized = normalize_text(
            original,
            vocabulary=vocabulary
        )

        if original.lower() != normalized:

            changed += 1

            if len(examples) < 5:
                examples.append(
                    (
                        original,
                        normalized
                    )
                )

    percentage = (
        changed / total
    ) * 100

    print(f"Total samples   : {total}")
    print(f"Changed samples : {changed}")
    print(f"Change rate     : {percentage:.2f}%")

    print("\nExamples:")

    if not examples:
        print("No changes detected.")

    else:

        for original, normalized in examples:

            print("\nORIGINAL:")
            print(original)

            print("\nNORMALIZED:")
            print(normalized)

            print("-" * 60)