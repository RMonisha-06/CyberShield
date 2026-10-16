import os
import re
import random
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

INPUT_FILE = "data/processed/test.csv"
OUTPUT_DIR = "data/obfuscated"

os.makedirs(OUTPUT_DIR, exist_ok=True)


LEET_MAP = {
    "a": "4",
    "e": "3",
    "i": "1",
    "o": "0",
    "s": "5",
    "t": "7",
}


def get_words(text):
    return re.findall(r"\b[a-zA-Z]{4,}\b", text)


def replace_word(text, old_word, new_word):
    pattern = rf"\b{re.escape(old_word)}\b"
    return re.sub(pattern, new_word, text, count=1)


def leetspeak(word):
    return "".join(LEET_MAP.get(c.lower(), c) for c in word)


def punctuation_insert(word):
    punctuation = random.choice([".", "_", "-", "*"])
    return punctuation.join(word)


def spacing_insert(word):
    return " ".join(word)


def character_repetition(word):
    if len(word) < 4:
        return word

    index = random.randint(1, len(word) - 2)
    repeat_count = random.choice([2, 3])

    return (
        word[:index]
        + word[index] * repeat_count
        + word[index + 1:]
    )


def character_deletion(word):
    if len(word) < 4:
        return word

    index = random.randint(1, len(word) - 2)

    return word[:index] + word[index + 1:]


def choose_word(text):
    words = get_words(text)

    if not words:
        return None

    return random.choice(words)


def apply_single_transformation(text, transformation):
    word = choose_word(text)

    if word is None:
        return text

    new_word = transformation(word)

    return replace_word(text, word, new_word)


def apply_mixed(text):
    word = choose_word(text)

    if word is None:
        return text

    transformations = random.sample(
        [
            leetspeak,
            punctuation_insert,
            spacing_insert,
            character_repetition,
            character_deletion,
        ],
        2
    )

    new_word = word

    for transformation in transformations:
        new_word = transformation(new_word)

    return replace_word(text, word, new_word)


def main():

    print("Loading test dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Test samples: {len(df)}")

    # Save clean reference
    clean_df = df[["text", "final_label"]].copy()
    clean_df.to_csv(
        f"{OUTPUT_DIR}/test_clean.csv",
        index=False
    )

    transformations = {
        "leetspeak": leetspeak,
        "punctuation": punctuation_insert,
        "spacing": spacing_insert,
        "repetition": character_repetition,
        "deletion": character_deletion,
    }

    for name, transformation in transformations.items():

        print(f"Generating {name}...")

        result = df.copy()

        result["text"] = result["text"].apply(
            lambda x: apply_single_transformation(
                str(x),
                transformation
            )
        )

        result[["text", "final_label"]].to_csv(
            f"{OUTPUT_DIR}/test_{name}.csv",
            index=False
        )

    print("Generating mixed obfuscation...")

    mixed_df = df.copy()

    mixed_df["text"] = mixed_df["text"].apply(
        lambda x: apply_mixed(str(x))
    )

    mixed_df[["text", "final_label"]].to_csv(
        f"{OUTPUT_DIR}/test_mixed.csv",
        index=False
    )

    print("\nObfuscated test sets created successfully.")

    print("\nFiles:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(" -", filename)


if __name__ == "__main__":
    main()