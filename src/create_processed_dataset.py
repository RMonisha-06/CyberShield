import pandas as pd
from pathlib import Path


print("=" * 70)
print("CyberShield - Creating Processed Dataset")
print("=" * 70)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CyberbullyX-63K A Large-Scale TwitterX Multilingua"
    / "CyberbullyX-63K.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "cyberbullyx_english.csv"


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load original dataset
# --------------------------------------------------

print("\nLoading original dataset...")

df = pd.read_excel(INPUT_FILE)

print("Original rows:", len(df))


# --------------------------------------------------
# Select English tweets
# --------------------------------------------------

df = df[df["lang"] == "en"].copy()

print("English rows:", len(df))


# --------------------------------------------------
# Keep only required columns
# --------------------------------------------------

df = df[["text", "final_label"]].copy()


# --------------------------------------------------
# Clean text formatting
# --------------------------------------------------

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
)


# --------------------------------------------------
# Remove empty texts
# --------------------------------------------------

before = len(df)

df = df[df["text"].str.len() > 0].copy()

removed = before - len(df)

print("Empty texts removed:", removed)


# --------------------------------------------------
# Remove duplicate text entries
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates(subset=["text"]).copy()

duplicates_removed = before - len(df)

print("Duplicate texts removed:", duplicates_removed)


# --------------------------------------------------
# Reset index
# --------------------------------------------------

df = df.reset_index(drop=True)


# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# Final statistics
# --------------------------------------------------

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal label distribution:")
print(df["final_label"].value_counts())

print("\nFinal label percentages:")
print(
    (df["final_label"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("Dataset creation complete")
print("=" * 70)