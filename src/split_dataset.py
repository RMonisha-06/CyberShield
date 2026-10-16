import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


print("=" * 70)
print("CyberShield - Train / Validation / Test Split")
print("=" * 70)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cyberbullyx_english.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Load processed dataset
# --------------------------------------------------

print("\nLoading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total samples:", len(df))


# --------------------------------------------------
# First split: 80% train, 20% temporary
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["final_label"]
)


# --------------------------------------------------
# Second split: temporary → validation + test
# --------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["final_label"]
)


# --------------------------------------------------
# Reset indices
# --------------------------------------------------

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False,
    encoding="utf-8"
)

validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# Function to display statistics
# --------------------------------------------------

def show_statistics(name, data):
    print(f"\n{name}")
    print("-" * 50)

    print("Samples:", len(data))

    print("\nLabel counts:")
    print(data["final_label"].value_counts())

    print("\nLabel percentages:")
    print(
        (data["final_label"].value_counts(normalize=True) * 100)
        .round(2)
    )


# --------------------------------------------------
# Display statistics
# --------------------------------------------------

show_statistics("TRAINING SET", train_df)
show_statistics("VALIDATION SET", validation_df)
show_statistics("TEST SET", test_df)


# --------------------------------------------------
# Final verification
# --------------------------------------------------

print("\n" + "=" * 70)
print("Split verification")
print("=" * 70)

print("\nTotal samples across splits:")
print(
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print("\nExpected total:")
print(len(df))

print("\nFiles created:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "validation.csv")
print(OUTPUT_DIR / "test.csv")

print("\n" + "=" * 70)
print("Dataset split complete")
print("=" * 70)