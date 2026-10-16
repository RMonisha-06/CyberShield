import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = "results/metrics/final_comparison_results.csv"
OUTPUT_FILE = "results/figures/f1_comparison.png"


df = pd.read_csv(INPUT_FILE)

# Exclude clean condition from the obfuscation comparison
plot_df = df[df["condition"] != "clean"].copy()

conditions = plot_df["condition"].tolist()

baseline_f1 = plot_df["baseline_f1"].tolist()
normalized_f1 = plot_df["normalized_f1"].tolist()

x = np.arange(len(conditions))
width = 0.35


plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    baseline_f1,
    width,
    label="Without Normalization"
)

plt.bar(
    x + width / 2,
    normalized_f1,
    width,
    label="With Normalization"
)


plt.xlabel("Obfuscation Type")
plt.ylabel("F1 Score (%)")

plt.title(
    "Cyberbullying Detection Performance Under Obfuscation"
)

plt.xticks(
    x,
    [
        "Leetspeak",
        "Punctuation",
        "Spacing",
        "Repetition",
        "Deletion",
        "Mixed",
    ]
)

plt.ylim(75, 85)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 70)
print("F1 COMPARISON FIGURE")
print("=" * 70)

print(f"Saved to:")
print(OUTPUT_FILE)