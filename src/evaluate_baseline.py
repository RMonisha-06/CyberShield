import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


print("=" * 70)
print("CyberShield - Baseline Bi-LSTM Evaluation")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODEL_FILE = PROJECT_ROOT / "models" / "baseline_bilstm.keras"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


# --------------------------------------------------
# Load test data
# --------------------------------------------------

X_test = np.load(DATA_DIR / "test_X.npy")
y_test = np.load(DATA_DIR / "test_y.npy")


print("\nTest samples:", len(y_test))


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_FILE)


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=1
).ravel()


predictions = (
    probabilities >= 0.5
).astype(int)


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 70)
print("Overall Metrics")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n" + "=" * 70)
print("Classification Report")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Non-Cyberbullying",
            "Cyberbullying"
        ],
        digits=4,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)


print("\n" + "=" * 70)
print("Confusion Matrix")
print("=" * 70)

print(cm)


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "macro_f1": float(macro_f1),
    "confusion_matrix": cm.tolist()
}


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


metrics_file = RESULTS_DIR / "baseline_metrics.npy"

np.save(
    metrics_file,
    metrics,
    allow_pickle=True
)


# Save predictions for reproducibility
predictions_file = RESULTS_DIR / "baseline_test_predictions.npy"

np.save(
    predictions_file,
    predictions
)


probabilities_file = RESULTS_DIR / "baseline_test_probabilities.npy"

np.save(
    probabilities_file,
    probabilities
)


print("\nMetrics saved to:")
print(metrics_file)

print("\nPredictions saved to:")
print(predictions_file)

print("\nProbabilities saved to:")
print(probabilities_file)


print("\n" + "=" * 70)
print("Baseline evaluation complete")
print("=" * 70)