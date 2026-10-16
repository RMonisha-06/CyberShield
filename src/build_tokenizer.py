import pandas as pd
import re
import json
from collections import Counter
from pathlib import Path


print("=" * 70)
print("CyberShield - Building Tokenizer")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
TOKENIZER_FILE = PROJECT_ROOT / "models" / "tokenizer.json"


MAX_VOCAB_SIZE = 20000
OOV_TOKEN = "<OOV>"


train = pd.read_csv(TRAIN_FILE)

print("\nTraining samples:", len(train))


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return re.findall(r"\w+|[^\w\s]", text)


counter = Counter()

for text in train["text"]:
    tokens = tokenize(text)
    counter.update(tokens)


print("Total unique tokens:", len(counter))


# Reserve index 1 for OOV
# Index 0 will be used for padding
vocabulary = [OOV_TOKEN]

for token, count in counter.most_common(MAX_VOCAB_SIZE - 1):
    vocabulary.append(token)


token_to_index = {
    token: index
    for index, token in enumerate(vocabulary, start=1)
}


# Save tokenizer information
tokenizer_data = {
    "max_vocab_size": MAX_VOCAB_SIZE,
    "oov_token": OOV_TOKEN,
    "token_to_index": token_to_index
}


TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(TOKENIZER_FILE, "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)


print("\nTokenizer vocabulary size:", len(token_to_index))
print("OOV token index:", token_to_index[OOV_TOKEN])

print("\nFirst 30 vocabulary entries:")

for token, index in list(token_to_index.items())[:30]:
    print(f"{index:5} -> {token!r}")


print("\nTokenizer saved to:")
print(TOKENIZER_FILE)

print("\n" + "=" * 70)
print("Tokenizer creation complete")
print("=" * 70)