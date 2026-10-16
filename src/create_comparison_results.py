import pandas as pd


BASELINE_RESULTS = {
    "clean": {
        "accuracy": 93.74,
        "precision": 91.03,
        "recall": 82.12,
        "f1": 86.35,
    },
    "leetspeak": {
        "accuracy": 90.66,
        "precision": 90.79,
        "recall": 68.13,
        "f1": 77.84,
    },
    "punctuation": {
        "accuracy": 91.63,
        "precision": 90.91,
        "recall": 72.51,
        "f1": 80.68,
    },
    "spacing": {
        "accuracy": 91.35,
        "precision": 92.99,
        "recall": 69.31,
        "f1": 79.42,
    },
    "repetition": {
        "accuracy": 91.15,
        "precision": 90.67,
        "recall": 70.49,
        "f1": 79.32,
    },
    "deletion": {
        "accuracy": 91.63,
        "precision": 91.43,
        "recall": 72.01,
        "f1": 80.57,
    },
    "mixed": {
        "accuracy": 91.35,
        "precision": 90.95,
        "recall": 71.16,
        "f1": 79.85,
    },
}


NORMALIZATION_RESULTS = {
    "leetspeak": {
        "accuracy": 91.15,
        "precision": 90.15,
        "recall": 70.99,
        "f1": 79.43,
    },
    "punctuation": {
        "accuracy": 91.92,
        "precision": 90.20,
        "recall": 74.54,
        "f1": 81.63,
    },
    "spacing": {
        "accuracy": 91.59,
        "precision": 91.06,
        "recall": 72.18,
        "f1": 80.53,
    },
    "repetition": {
        "accuracy": 91.31,
        "precision": 90.41,
        "recall": 71.50,
        "f1": 79.85,
    },
    "deletion": {
        "accuracy": 91.63,
        "precision": 91.43,
        "recall": 72.01,
        "f1": 80.57,
    },
    "mixed": {
        "accuracy": 91.27,
        "precision": 90.56,
        "recall": 71.16,
        "f1": 79.70,
    },
}


rows = []

for condition, baseline in BASELINE_RESULTS.items():

    if condition == "clean":

        rows.append({
            "condition": "clean",
            "baseline_f1": baseline["f1"],
            "normalized_f1": baseline["f1"],
            "f1_change": 0.00,
        })

        continue

    normalized = NORMALIZATION_RESULTS[condition]

    rows.append({
        "condition": condition,
        "baseline_f1": baseline["f1"],
        "normalized_f1": normalized["f1"],
        "f1_change": round(
            normalized["f1"] - baseline["f1"],
            2
        ),
    })


df = pd.DataFrame(rows)


print("=" * 70)
print("FINAL OBFUSCATION COMPARISON")
print("=" * 70)

print(df.to_string(index=False))


output_file = "results/metrics/final_comparison_results.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)