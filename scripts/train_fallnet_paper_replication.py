"""
Paper-replication training run: CNN-LSTM ensemble (Jain & Semwal 2022,
Table II) on the paper-exact 8-class dataset built by
scripts/preprocess_paper_exact.py.

Training config per the paper:
  - 8 classes, batch_size=512, epochs=200, Adam with Keras-default LR (1e-3)
  - Dropout 0.2 + L1L2 regularization + early stopping
  - Stratified 5-fold CV, NO class weights (paper handles imbalance via
    stratified sampling only)

Paper targets (Table III): accuracy 97.52%, Fall_Initiation recall 99.24%,
Fall_Initiation F1 98.79%.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight  # noqa: F401 (unused, paper uses none)

DATA_DIR = Path("fall_detection_data/processed")
OUT_DIR = Path("fall_detection_data/models/fallnet_paper_replication")
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_data = np.load(DATA_DIR / "X_data_8class_paper.npy")
y_labels = np.load(DATA_DIR / "y_labels_8class_paper.npy")
N_CLASSES = 8
y_categorical = keras.utils.to_categorical(y_labels, num_classes=N_CLASSES)
label_map = json.load(open(DATA_DIR / "label_map_8class_paper.json"))
reverse_label_map = {v: k for k, v in label_map.items()}

BATCH_SIZE = 512
LEARNING_RATE = 1e-3  # Keras default, per paper
EPOCHS = 200          # per paper
K_FOLDS = 5

print(f"Data: X={X_data.shape} y={y_labels.shape}")
print(f"Config: batch_size={BATCH_SIZE} lr={LEARNING_RATE} epochs={EPOCHS} k_folds={K_FOLDS} classes={N_CLASSES}")

REG = regularizers.l1_l2(l1=1e-5, l2=1e-4)


def build_cnn_lstm_ensemble(input_shape=(200, 6), n_classes=N_CLASSES):
    inputs = layers.Input(shape=input_shape, name="input")

    lx = layers.LSTM(256, activation="tanh", return_sequences=False, name="lstm_layer")(inputs)
    lx = layers.Dense(128, activation="relu", kernel_regularizer=REG, name="lstm_dense1")(lx)
    lx = layers.Dropout(0.2, name="lstm_dropout1")(lx)
    lx = layers.Dense(64, activation="relu", kernel_regularizer=REG, name="lstm_dense2")(lx)
    lx = layers.Dropout(0.2, name="lstm_dropout2")(lx)
    lx = layers.Dense(32, activation="relu", kernel_regularizer=REG, name="lstm_dense3")(lx)
    lx = layers.Dropout(0.2, name="lstm_dropout3")(lx)
    lstm_output = layers.Dense(n_classes, activation="softmax", name="lstm_output")(lx)

    cx = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d_layer")(inputs)
    cx = layers.MaxPooling1D(pool_size=2, name="maxpool_layer")(cx)
    cx = layers.Flatten(name="flatten_layer")(cx)
    cx = layers.Dense(1024, activation="relu", kernel_regularizer=REG, name="cnn_dense1")(cx)
    cx = layers.Dropout(0.2, name="cnn_dropout1")(cx)
    cx = layers.Dense(512, activation="relu", kernel_regularizer=REG, name="cnn_dense2")(cx)
    cx = layers.Dropout(0.2, name="cnn_dropout2")(cx)
    cnn_output = layers.Dense(n_classes, activation="softmax", name="cnn_output")(cx)

    outputs = layers.Average(name="ensemble_average")([lstm_output, cnn_output])
    return models.Model(inputs=inputs, outputs=outputs, name="FallNet_CNN_LSTM_paper")


skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_results = []
per_class_reports = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels), 1):
    print(f"\n{'='*80}\nFOLD {fold}/{K_FOLDS} - PAPER REPLICATION\n{'='*80}")
    keras.backend.clear_session()

    X_train, X_val = X_data[train_idx], X_data[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
    y_val_int = y_labels[val_idx]
    print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,}")

    model = build_cnn_lstm_ensemble()
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
        callbacks=callbacks,   # NO class_weight (paper uses stratified sampling only)
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
    model.save(OUT_DIR / f"fallnet_paper_fold_{fold}.keras")

results_df = pd.DataFrame(fold_results)
mean_r = results_df.mean(numeric_only=True)
std_r = results_df.std(numeric_only=True)

# average per-class metrics across folds
class_rows = []
for cls in range(N_CLASSES):
    name = reverse_label_map[cls]
    prec = np.mean([r[name]["precision"] for r in per_class_reports])
    rec = np.mean([r[name]["recall"] for r in per_class_reports])
    f1 = np.mean([r[name]["f1-score"] for r in per_class_reports])
    sup = int(np.mean([r[name]["support"] for r in per_class_reports]))
    class_rows.append({"class": name, "precision": prec, "recall": rec, "f1": f1, "mean_support": sup})
class_df = pd.DataFrame(class_rows)

paper_table3 = """
Paper (Table III, single test fold):
  Walking               P=0.9868 R=0.9803 F1=0.9835  (sup 304)
  Jogging               P=0.9799 R=0.9899 F1=0.9849  (sup 296)
  Walking_stairs_updown P=0.9620 R=0.9870 F1=0.9744  (sup 308)
  Stumble_while_walking P=0.9455 R=0.9286 F1=0.9369  (sup 56)
  Fall_Recovery         P=0.9792 R=0.8545 F1=0.9126  (sup 55)
  Fall_Initiation       P=0.9834 R=0.9924 F1=0.9879  (sup 658)
  Impact                P=0.9583 R=0.9787 F1=0.9684  (sup 329)
  Aftermath             P=0.9778 R=0.9362 F1=0.9565  (sup 329)
  Accuracy: 0.9752
"""

summary = f"""
================================================================================
FALLNET PAPER REPLICATION - CNN-LSTM ensemble, 8-class paper-exact dataset
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
{paper_table3}
Paper targets: accuracy 97.52%, Fall_Initiation recall 99.24%, F1 98.79%
This run:      accuracy {mean_r['val_accuracy']*100:.2f}%, Fall_Initiation recall {mean_r['fall_init_recall']*100:.2f}%, F1 {mean_r['fall_init_f1']*100:.2f}%
"""
print(summary)
with open(OUT_DIR / "replication_summary.txt", "w") as f:
    f.write(summary)
results_df.to_json(OUT_DIR / "fold_results.json", orient="records", indent=2)
class_df.to_json(OUT_DIR / "per_class_results.json", orient="records", indent=2)
print(f"\nSaved to {OUT_DIR}/")
