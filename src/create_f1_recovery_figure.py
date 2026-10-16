import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "results/metrics/f1_recovery_results.csv"
OUTPUT_FILE = "results/figures/f1_recovery.png"


df = pd.read_csv(INPUT_FILE)


plt.figure(figsize=(9, 6))

bars = plt.bar(
    df["condition"],
    df["recovery_percentage"]
)


plt.axhline(
    y=0,
    linewidth=1
)


plt.xlabel("Obfuscation Type")
plt.ylabel("F1 Recovery (%)")

plt.title(
    "F1 Performance Recovery Through Obfuscation Normalization"
)

plt.xticks(
    rotation=20
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


# Display values above bars
for bar, value in zip(
    bars,
    df["recovery_percentage"]
):

    if value >= 0:
        y_position = value + 0.5
        vertical_alignment = "bottom"

    else:
        y_position = value - 0.5
        vertical_alignment = "top"

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        y_position,
        f"{value:.2f}%",
        ha="center",
        va=vertical_alignment
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("=" * 70)
print("F1 RECOVERY FIGURE")
print("=" * 70)

print("Saved to:")
print(OUTPUT_FILE)