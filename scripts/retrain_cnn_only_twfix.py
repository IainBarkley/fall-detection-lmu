"""
Re-run of the CNN-only baseline (fall_detection_data/models/cnn_only_model)
with batch_size=512, learning_rate=1e-3 (Jain & Semwal's Table II settings),
instead of the original batch_size=2, learning_rate=1e-5.

Architecture is identical to the documented CNN-only model in
fall_detection_data/models/cnn_only_model/cnn_only_summary.txt:
Conv1D(128, k=3) -> MaxPool -> Flatten -> Dense(1024) -> Dense(512) -> Dense(6, softmax)
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path("fall_detection_data/processed")
OUT_DIR = Path("fall_detection_data/models/cnn_only_twfix_bs512_lr1e3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_data = np.load(DATA_DIR / "X_data_6class_twfix.npy")
y_labels = np.load(DATA_DIR / "y_labels_6class_twfix.npy")
y_categorical = keras.utils.to_categorical(y_labels, num_classes=6)
label_map = json.load(open(DATA_DIR / "label_map_6class_twfix.json"))
reverse_label_map = {v: k for k, v in label_map.items()}

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 50
K_FOLDS = 5

print(f"Data: X={X_data.shape} y={y_labels.shape}")
print(f"Config: batch_size={BATCH_SIZE} lr={LEARNING_RATE} epochs={EPOCHS} k_folds={K_FOLDS}")


def build_cnn_only(input_shape=(200, 6), n_classes=6):
    inputs = layers.Input(shape=input_shape, name="input")
    x = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d_layer")(inputs)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_layer")(x)
    x = layers.Flatten(name="flatten_layer")(x)
    x = layers.Dense(1024, activation="relu", name="cnn_dense1")(x)
    x = layers.Dropout(0.2, name="cnn_dropout1")(x)
    x = layers.Dense(512, activation="relu", name="cnn_dense2")(x)
    x = layers.Dropout(0.2, name="cnn_dropout2")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="cnn_output")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="FallNet_CNN_Only")


skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_results = []

class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_labels), y=y_labels
)
class_weights = dict(enumerate(np.clip(class_weights_array, None, 3.0)))
counts = Counter(y_labels)
print("\nClass distribution / weights:")
for cls_idx in range(6):
    print(f"  {reverse_label_map[cls_idx]:<25s}: {counts[cls_idx]:>5d} -> weight {class_weights[cls_idx]:.2f}x")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels), 1):
    print(f"\n{'='*80}\nFOLD {fold}/{K_FOLDS}\n{'='*80}")
    keras.backend.clear_session()

    X_train, X_val = X_data[train_idx], X_data[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
    print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,}")

    model = build_cnn_only()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
    n_epochs_run = len(history.history["loss"])

    print(f"\nFold {fold} -> acc={val_acc:.4f} f1={val_f1:.4f} (ran {n_epochs_run} epochs)")

    fold_results.append({
        "fold": fold, "epochs_run": n_epochs_run,
        "val_loss": val_loss, "val_accuracy": val_acc,
        "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1,
    })
    model.save(OUT_DIR / f"cnn_only_bs512_fold_{fold}.keras")

results_df = pd.DataFrame(fold_results)
mean_r = results_df.mean(numeric_only=True)
std_r = results_df.std(numeric_only=True)

summary = f"""
================================================================================
CNN-ONLY, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE} (vs original batch_size=2, lr=1e-5)
================================================================================

Per-fold:
{results_df.to_string(index=False)}

Average Performance (5-fold CV):
  Accuracy:  {mean_r['val_accuracy']:.4f} +/- {std_r['val_accuracy']:.4f}
  Precision: {mean_r['val_precision']:.4f} +/- {std_r['val_precision']:.4f}
  Recall:    {mean_r['val_recall']:.4f} +/- {std_r['val_recall']:.4f}
  F1-Score:  {mean_r['val_f1']:.4f} +/- {std_r['val_f1']:.4f}

Comparison:
  Original (batch=2, lr=1e-5):   88.82% +/- 0.66%
  This run (batch=512, lr=1e-3): {mean_r['val_accuracy']*100:.2f}% +/- {std_r['val_accuracy']*100:.2f}%
"""
print(summary)
with open(OUT_DIR / "retrain_summary.txt", "w") as f:
    f.write(summary)
results_df.to_json(OUT_DIR / "fold_results.json", orient="records", indent=2)
print(f"\nSaved to {OUT_DIR}/")
