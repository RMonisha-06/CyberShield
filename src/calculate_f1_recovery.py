import pandas as pd


INPUT_FILE = "results/metrics/final_comparison_results.csv"
OUTPUT_FILE = "results/metrics/f1_recovery_results.csv"


df = pd.read_csv(INPUT_FILE)

# Clean-text baseline F1
clean_f1 = df.loc[
    df["condition"] == "clean",
    "baseline_f1"
].iloc[0]


# Keep only obfuscated conditions
df = df[
    df["condition"] != "clean"
].copy()


# F1 lost because of obfuscation
df["f1_loss"] = (
    clean_f1 - df["baseline_f1"]
)


# F1 recovered by normalization
df["f1_recovered"] = (
    df["normalized_f1"] - df["baseline_f1"]
)


# Percentage of lost F1 recovered
df["recovery_percentage"] = (
    df["f1_recovered"]
    / df["f1_loss"]
    * 100
)


# Round values
df["f1_loss"] = df["f1_loss"].round(2)
df["f1_recovered"] = df["f1_recovered"].round(2)
df["recovery_percentage"] = (
    df["recovery_percentage"].round(2)
)


print("=" * 70)
print("F1 RECOVERY ANALYSIS")
print("=" * 70)

print(f"\nClean-text F1: {clean_f1:.2f}%\n")

print(
    df[
        [
            "condition",
            "baseline_f1",
            "normalized_f1",
            "f1_loss",
            "f1_recovered",
            "recovery_percentage",
        ]
    ].to_string(index=False)
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nResults saved to:")
print(OUTPUT_FILE)