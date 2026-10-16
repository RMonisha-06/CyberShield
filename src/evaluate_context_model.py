import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("CyberShield - Context-Enhanced Model Evaluation")
print("=" * 70)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
CONTEXT_DIR = PROJECT_ROOT / "data" / "processed" / "context_features"
MODEL_PATH = PROJECT_ROOT / "models" / "context_bilstm.keras"

# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

X_test_text = np.load(
    SEQUENCE_DIR / "test_X.npy"
)

y_test = np.load(
    SEQUENCE_DIR / "test_y.npy"
)

X_test_context = np.load(
    CONTEXT_DIR / "test_context.npy"
)

print("\nTest data:")
print("Text shape:", X_test_text.shape)
print("Context shape:", X_test_context.shape)
print("Labels shape:", y_test.shape)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\nLoading context-enhanced model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

# ---------------------------------------------------------
# Generate probabilities
# ---------------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict(
    {
        "text_input": X_test_text,
        "context_input": X_test_context
    },
    verbose=1
).ravel()

# ---------------------------------------------------------
# Convert probabilities to labels
# ---------------------------------------------------------

y_pred = (
    probabilities >= 0.5
).astype(int)

# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONTEXT-ENHANCED MODEL RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"Macro F1 : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Cyberbullying",
            "Cyberbullying"
        ],
        digits=4,
        zero_division=0
    )
)

# ---------------------------------------------------------
# Compare with baseline
# ---------------------------------------------------------

baseline_f1 = 0.8635

improvement = (
    f1 - baseline_f1
) * 100

print("=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print(f"\nBaseline Bi-LSTM F1       : 86.35%")
print(f"Context-Enhanced F1       : {f1 * 100:.2f}%")
print(f"F1 Difference             : {improvement:+.2f} percentage points")

if improvement > 0:
    print("\nResult: Context-enhanced model improved over baseline.")
elif improvement < 0:
    print("\nResult: Context-enhanced model is lower than baseline.")
else:
    print("\nResult: Context-enhanced model matches baseline.")

print("\n" + "=" * 70)
print("Evaluation complete")
print("=" * 70)