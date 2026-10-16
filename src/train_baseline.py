import numpy as np
import tensorflow as tf
from pathlib import Path


print("=" * 70)
print("CyberShield - Baseline Bi-LSTM Training")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCAB_SIZE = 20001
SEQUENCE_LENGTH = 80

EMBEDDING_DIM = 128
LSTM_UNITS = 64

BATCH_SIZE = 64
EPOCHS = 10

RANDOM_SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

tf.keras.utils.set_random_seed(RANDOM_SEED)


# --------------------------------------------------
# Load data
# --------------------------------------------------

X_train = np.load(DATA_DIR / "train_X.npy")
y_train = np.load(DATA_DIR / "train_y.npy")

X_val = np.load(DATA_DIR / "validation_X.npy")
y_val = np.load(DATA_DIR / "validation_y.npy")

X_test = np.load(DATA_DIR / "test_X.npy")
y_test = np.load(DATA_DIR / "test_y.npy")


print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# Build model
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,)),

    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS)
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


print("\nModel architecture:\n")

model.summary()


# --------------------------------------------------
# Create output directories
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

checkpoint_path = MODEL_DIR / "baseline_bilstm.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )
]


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nStarting training...\n")


history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1
)


# --------------------------------------------------
# Save training history
# --------------------------------------------------

history_path = RESULTS_DIR / "baseline_training_history.npy"

np.save(
    history_path,
    history.history,
    allow_pickle=True
)


# --------------------------------------------------
# Final test evaluation
# --------------------------------------------------

print("\nEvaluating on test set...\n")


test_results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


print("\nTest results:")

for metric_name, value in zip(model.metrics_names, test_results):
    print(f"{metric_name}: {value:.4f}")


print("\nBest model saved to:")
print(checkpoint_path)

print("\nTraining history saved to:")
print(history_path)


print("\n" + "=" * 70)
print("Baseline training complete")
print("=" * 70)