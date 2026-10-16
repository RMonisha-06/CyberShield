import pandas as pd


COMPARISON_FILE = (
    "results/metrics/final_comparison_results.csv"
)

RECOVERY_FILE = (
    "results/metrics/f1_recovery_results.csv"
)

OUTPUT_FILE = (
    "results/metrics/final_results_summary.csv"
)


# Load existing results
comparison = pd.read_csv(
    COMPARISON_FILE
)

recovery = pd.read_csv(
    RECOVERY_FILE
)


# Merge the two result files
summary = comparison.merge(
    recovery[
        [
            "condition",
            "f1_loss",
            "f1_recovered",
            "recovery_percentage",
        ]
    ],
    on="condition",
    how="left"
)


# Add clean baseline information
clean_f1 = summary.loc[
    summary["condition"] == "clean",
    "baseline_f1"
].iloc[0]


summary.insert(
    1,
    "clean_f1",
    clean_f1
)


# Rename columns for clarity
summary = summary.rename(
    columns={
        "baseline_f1": "f1_without_normalization",
        "normalized_f1": "f1_with_normalization",
        "f1_change": "f1_change_pp",
        "f1_loss": "f1_loss_pp",
        "f1_recovered": "f1_recovered_pp",
        "recovery_percentage": "f1_recovery_percentage",
    }
)


# Fill clean-condition recovery fields
summary.loc[
    summary["condition"] == "clean",
    [
        "f1_loss_pp",
        "f1_recovered_pp",
        "f1_recovery_percentage",
    ]
] = 0.0


# Round numerical values
numeric_columns = [
    "clean_f1",
    "f1_without_normalization",
    "f1_with_normalization",
    "f1_change_pp",
    "f1_loss_pp",
    "f1_recovered_pp",
    "f1_recovery_percentage",
]


summary[numeric_columns] = summary[
    numeric_columns
].round(2)


# Save
summary.to_csv(
    OUTPUT_FILE,
    index=False
)


print("=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(OUTPUT_FILE)