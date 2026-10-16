import pandas as pd
from pathlib import Path

print("=" * 70)
print("CyberShield - Dataset Inspection")
print("=" * 70)

# Build the path safely for Windows
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CyberbullyX-63K A Large-Scale TwitterX Multilingua"
    / "CyberbullyX-63K.xlsx"
)

print("\nDataset file:")
print(FILE_PATH)

# Check that the file exists
if not FILE_PATH.exists():
    print("\nERROR: Dataset file was not found.")
    raise SystemExit(1)

# Load Excel file
excel_file = pd.ExcelFile(FILE_PATH)

print("\nSheets:")
print(excel_file.sheet_names)

# Read the first sheet
df = pd.read_excel(
    FILE_PATH,
    sheet_name=excel_file.sheet_names[0]
)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique values per column:")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].nunique())

print("\n" + "=" * 70)
print("Inspection complete")
print("=" * 70)