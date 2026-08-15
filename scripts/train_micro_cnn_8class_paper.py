"""
Micro-CNN (41K params, notebooks/ReducingCNNWeights.ipynb architecture) trained
on the paper-exact 8-class dataset, using the identical protocol as
scripts/train_fallnet_paper_replication.py so the two are directly comparable:

  - 8 classes, batch_size=512, lr=1e-3, epochs=200, early stopping patience 20
  - Stratified 5-fold CV (random_state=42), NO class weights

Question this answers: can the deployable 41K-parameter model match the paper's
13.95M-parameter CNN-LSTM ensemble on the paper's own task?

Reference points:
  CNN-LSTM replication (same data/protocol): 88.73% acc, 97.85% Fall_Init recall
  Paper's reported numbers:                  97.52% acc, 99.24% Fall_Init recall
  Micro-CNN on 6-class (buggy) data:         94.71% acc, 97.82% Fall_Init recall
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

DATA_DIR = Path("fall_detection_data/processed")
OUT_DIR = Path("fall_detection_data/models/micro_cnn_8class_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_data = np.load(DATA_DIR / "X_data_8class_paper.npy")
y_labels = np.load(DATA_DIR / "y_labels_8class_paper.npy")
N_CLASSES = 8
y_categorical = keras.utils.to_categorical(y_labels, num_classes=N_CLASSES)
label_map = json.load(open(DATA_DIR / "label_map_8class_paper.json"))
reverse_label_map = {v: k for k, v in label_map.items()}

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 200
K_FOLDS = 5

print(f"Data: X={X_data.shape} y={y_labels.shape}")
print(f"Config: batch_size={BATCH_SIZE} lr={LEARNING_RATE} epochs={EPOCHS} k_folds={K_FOLDS} classes={N_CLASSES}")


def build_micro_cnn(input_shape=(200, 6), n_classes=N_CLASSES):
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv1D(32, 3, padding="same", name="micro_conv1")(inputs)
    x = layers.BatchNormalization(name="micro_bn1")(x)
    x = layers.ReLU(name="micro_relu1")(x)
    x = layers.MaxPooling1D(2, name="micro_pool1")(x)

    x = layers.Conv1D(64, 3, padding="same", name="micro_conv2")(x)
    x = layers.BatchNormalization(name="micro_bn2")(x)
    x = layers.ReLU(name="micro_relu2")(x)
    x = layers.MaxPooling1D(2, name="micro_pool2")(x)

    x = layers.Conv1D(128, 3, padding="same", name="micro_conv3")(x)
    x = layers.BatchNormalization(name="micro_bn3")(x)
    x = layers.ReLU(name="micro_relu3")(x)
    x = layers.MaxPooling1D(2, name="micro_pool3")(x)

    x = layers.GlobalAveragePooling1D(name="micro_global_pool")(x)
    x = layers.Dense(64, activation="relu", name="micro_fc1")(x)
    x = layers.Dropout(0.3, name="micro_dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="micro_output")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="FallNet_MicroCNN_8class")


skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_results = []
per_class_reports = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels), 1):
    print(f"\n{'='*80}\nFOLD {fold}/{K_FOLDS} - MICRO-CNN 8-CLASS (paper-exact data)\n{'='*80}")
    keras.backend.clear_session()

    X_train, X_val = X_data[train_idx], X_data[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
    y_val_int = y_labels[val_idx]
    print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,}")

    model = build_micro_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    if fold == 1:
        print(f"Params: {model.count_params():,}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,   # no class weights, matching the replication protocol
        verbose=2,
    )

    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(
        y_val_int, y_pred,
        labels=list(range(N_CLASSES)),
        target_names=[reverse_label_map[i] for i in range(N_CLASSES)],
        output_dict=True, zero_division=0,
    )
    per_class_reports.append(report)
    fi = report["Fall_Initiation"]
    n_epochs_run = len(history.history["loss"])

    print(f"\nFold {fold} -> acc={val_acc:.4f} Fall_Init recall={fi['recall']:.4f} f1={fi['f1-score']:.4f} ({n_epochs_run} epochs)")

    fold_results.append({
        "fold": fold, "epochs_run": n_epochs_run,
        "val_loss": val_loss, "val_accuracy": val_acc,
        "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1,
        "fall_init_recall": fi["recall"], "fall_init_f1": fi["f1-score"],
    })
    model.save(OUT_DIR / f"micro_cnn_8class_fold_{fold}.keras")

results_df = pd.DataFrame(fold_results)
mean_r = results_df.mean(numeric_only=True)
std_r = results_df.std(numeric_only=True)

class_rows = []
for cls in range(N_CLASSES):
    name = reverse_label_map[cls]
    class_rows.append({
        "class": name,
        "precision": np.mean([r[name]["precision"] for r in per_class_reports]),
        "recall": np.mean([r[name]["recall"] for r in per_class_reports]),
        "f1": np.mean([r[name]["f1-score"] for r in per_class_reports]),
        "mean_support": int(np.mean([r[name]["support"] for r in per_class_reports])),
    })
class_df = pd.DataFrame(class_rows)

summary = f"""
================================================================================
MICRO-CNN (41K params) on paper-exact 8-class dataset
batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, epochs={EPOCHS} (early stopping), no class weights
================================================================================

Per-fold:
{results_df.to_string(index=False)}

Average Performance (5-fold CV):
  Accuracy:           {mean_r['val_accuracy']:.4f} +/- {std_r['val_accuracy']:.4f}
  F1-Score:           {mean_r['val_f1']:.4f} +/- {std_r['val_f1']:.4f}
  Fall_Init Recall:   {mean_r['fall_init_recall']:.4f} +/- {std_r['fall_init_recall']:.4f}
  Fall_Init F1:       {mean_r['fall_init_f1']:.4f} +/- {std_r['fall_init_f1']:.4f}

Per-class (mean over 5 folds):
{class_df.to_string(index=False)}

Comparison (same data, same protocol):
  CNN-LSTM ensemble (13.95M params): 88.73% acc, 97.85% Fall_Init recall
  Micro-CNN (41K params, this run):  {mean_r['val_accuracy']*100:.2f}% acc, {mean_r['fall_init_recall']*100:.2f}% Fall_Init recall
  Paper's reported:                  97.52% acc, 99.24% Fall_Init recall
"""
print(summary)
with open(OUT_DIR / "summary.txt", "w") as f:
    f.write(summary)
results_df.to_json(OUT_DIR / "fold_results.json", orient="records", indent=2)
class_df.to_json(OUT_DIR / "per_class_results.json", orient="records", indent=2)
print(f"\nSaved to {OUT_DIR}/")
