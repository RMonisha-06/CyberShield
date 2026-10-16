import pandas as pd
from pathlib import Path

print("=" * 70)
print("CyberShield - Dataset Statistics")
print("=" * 70)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CyberbullyX-63K A Large-Scale TwitterX Multilingua"
    / "CyberbullyX-63K.xlsx"
)

df = pd.read_excel(FILE_PATH)

# --------------------------------------------------
# Overall label distribution
# --------------------------------------------------

print("\nOverall label distribution:")
print(df["final_label"].value_counts())

print("\nOverall label percentage:")
print(
    (df["final_label"].value_counts(normalize=True) * 100)
    .round(2)
)

# --------------------------------------------------
# Language distribution
# --------------------------------------------------

print("\nLanguage distribution:")
print(df["lang"].value_counts())

# --------------------------------------------------
# English subset
# --------------------------------------------------

english_df = df[df["lang"] == "en"].copy()

print("\nEnglish dataset size:")
print(len(english_df))

print("\nEnglish label distribution:")
print(english_df["final_label"].value_counts())

print("\nEnglish label percentage:")
print(
    (english_df["final_label"].value_counts(normalize=True) * 100)
    .round(2)
)

# --------------------------------------------------
# Text statistics
# --------------------------------------------------

english_df["text_length"] = english_df["text"].astype(str).str.len()

print("\nEnglish text length statistics:")
print(english_df["text_length"].describe())

print("\n" + "=" * 70)
print("Statistics complete")
print("=" * 70)