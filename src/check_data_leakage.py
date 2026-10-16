import pandas as pd
from pathlib import Path


print("=" * 70)
print("CyberShield - Data Leakage Check")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


# --------------------------------------------------
# Convert text columns to sets
# --------------------------------------------------

train_texts = set(train["text"])
validation_texts = set(validation["text"])
test_texts = set(test["text"])


# --------------------------------------------------
# Check overlaps
# --------------------------------------------------

train_validation = train_texts & validation_texts
train_test = train_texts & test_texts
validation_test = validation_texts & test_texts


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nTrain ∩ Validation:", len(train_validation))
print("Train ∩ Test:", len(train_test))
print("Validation ∩ Test:", len(validation_test))


# --------------------------------------------------
# Final result
# --------------------------------------------------

if (
    len(train_validation) == 0
    and len(train_test) == 0
    and len(validation_test) == 0
):
    print("\n✅ No text leakage detected.")
else:
    print("\n⚠️ Text overlap detected.")

print("\n" + "=" * 70)
print("Leakage check complete")
print("=" * 70)