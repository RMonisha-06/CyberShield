import numpy as np
import tensorflow as tf
from pathlib import Path


print("=" * 70)
print("CyberShield - Context-Enhanced Bi-LSTM Training")
print("=" * 70)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
CONTEXT_DIR = PROJECT_ROOT / "data" / "processed" / "context_features"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VOCAB_SIZE = 20001
SEQUENCE_LENGTH = 80

EMBEDDING_DIM = 128
LSTM_UNITS = 64

CONTEXT_FEATURES = 11

BATCH_SIZE = 64
EPOCHS = 10

RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ---------------------------------------------------------
# Load text sequences
# ---------------------------------------------------------

print("\nLoading text sequences...")

X_train_text = np.load(SEQUENCE_DIR / "train_X.npy")
y_train = np.load(SEQUENCE_DIR / "train_y.npy")

X_val_text = np.load(SEQUENCE_DIR / "validation_X.npy")
y_val = np.load(SEQUENCE_DIR / "validation_y.npy")

X_test_text = np.load(SEQUENCE_DIR / "test_X.npy")
y_test = np.load(SEQUENCE_DIR / "test_y.npy")


# ---------------------------------------------------------
# Load context features
# ---------------------------------------------------------

print("Loading context features...")

X_train_context = np.load(
    CONTEXT_DIR / "train_context.npy"
)

X_val_context = np.load(
    CONTEXT_DIR / "validation_context.npy"
)

X_test_context = np.load(
    CONTEXT_DIR / "test_context.npy"
)


# ---------------------------------------------------------
# Display dataset shapes
# ---------------------------------------------------------

print("\nDataset shapes:")

print("Text:")
print("X_train_text:", X_train_text.shape)
print("X_val_text:  ", X_val_text.shape)
print("X_test_text: ", X_test_text.shape)

print("\nContext:")
print("X_train_context:", X_train_context.shape)
print("X_val_context:  ", X_val_context.shape)
print("X_test_context: ", X_test_context.shape)

print("\nLabels:")
print("y_train:", y_train.shape)
print("y_val:  ", y_val.shape)
print("y_test: ", y_test.shape)


# ---------------------------------------------------------
# Build context normalization layer
# ---------------------------------------------------------

print("\nPreparing context feature normalization...")

context_normalizer = tf.keras.layers.Normalization(
    axis=-1,
    name="context_normalization"
)

context_normalizer.adapt(X_train_context)


# ---------------------------------------------------------
# Text branch
# ---------------------------------------------------------

text_input = tf.keras.layers.Input(
    shape=(SEQUENCE_LENGTH,),
    name="text_input"
)

embedding = tf.keras.layers.Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="word_embedding"
)(text_input)

bilstm_output = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        LSTM_UNITS
    ),
    name="bidirectional_lstm"
)(embedding)

bilstm_output = tf.keras.layers.Dropout(
    0.5,
    name="text_dropout"
)(bilstm_output)


# ---------------------------------------------------------
# Context branch
# ---------------------------------------------------------

context_input = tf.keras.layers.Input(
    shape=(CONTEXT_FEATURES,),
    name="context_input"
)

normalized_context = context_normalizer(
    context_input
)


# ---------------------------------------------------------
# Feature fusion
# ---------------------------------------------------------

combined_features = tf.keras.layers.Concatenate(
    name="feature_fusion"
)([
    bilstm_output,
    normalized_context
])


# ---------------------------------------------------------
# Classification layers
# ---------------------------------------------------------

dense_features = tf.keras.layers.Dense(
    32,
    activation="relu",
    name="fusion_dense"
)(combined_features)

dense_features = tf.keras.layers.Dropout(
    0.3,
    name="fusion_dropout"
)(dense_features)

output = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="cyberbullying_output"
)(dense_features)


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------

model = tf.keras.Model(
    inputs=[
        text_input,
        context_input
    ],
    outputs=output,
    name="CyberShield_Context_BiLSTM"
)


# ---------------------------------------------------------
# Compile model
# ---------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ---------------------------------------------------------
# Display architecture
# ---------------------------------------------------------

print("\nModel architecture:\n")

model.summary()


# ---------------------------------------------------------
# Create output directories
# ---------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Model checkpoint
# ---------------------------------------------------------

checkpoint_path = MODEL_DIR / "context_bilstm.keras"

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


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\nStarting context-enhanced training...\n")

history = model.fit(
    x={
        "text_input": X_train_text,
        "context_input": X_train_context
    },
    y=y_train,

    validation_data=(
        {
            "text_input": X_val_text,
            "context_input": X_val_context
        },
        y_val
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1
)


# ---------------------------------------------------------
# Save training history
# ---------------------------------------------------------

history_path = (
    RESULTS_DIR /
    "context_bilstm_training_history.npy"
)

np.save(
    history_path,
    history.history,
    allow_pickle=True
)


# ---------------------------------------------------------
# Evaluate test set
# ---------------------------------------------------------

print("\nEvaluating context-enhanced model on test set...\n")

test_results = model.evaluate(
    {
        "text_input": X_test_text,
        "context_input": X_test_context
    },
    y_test,
    verbose=1
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\nTest results:")

for metric_name, value in zip(
    model.metrics_names,
    test_results
):
    print(
        f"{metric_name}: {value:.4f}"
    )


# ---------------------------------------------------------
# Final information
# ---------------------------------------------------------

print("\nContext-enhanced model saved to:")
print(checkpoint_path)

print("\nTraining history saved to:")
print(history_path)

print("\n" + "=" * 70)
print("Context-enhanced Bi-LSTM training complete")
print("=" * 70)