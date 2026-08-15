"""
CNN-LSTM ensemble (paper-exact FallNet, Jain & Semwal Table II) trained on the
transitional-window-fixed dataset (X_data_6class_twfix.npy), with
batch_size=512 / lr=1e-3 to match the retrained CNN-only baselines.

Architecture copied from notebooks/02_DataExplorationBothDataSets.ipynb
("EXACT replication of paper" FallNet class): LSTM(256) branch and
Conv1D(128,k=3) branch, each with their own dense stacks and softmax heads,
averaged at the output.

Comparison targets:
  CNN-only  (orig data,  bs512/lr1e-3): 88.88% acc, 95.63% Fall_Init recall
  CNN-only  (twfix data, bs512/lr1e-3): 88.83% acc, 96.60% Fall_Init recall
  CNN-LSTM  (orig data,  bs64, old run): 87.56% +/- 3.74% acc
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
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path("fall_detection_data/processed")
OUT_DIR = Path("fall_detection_data/models/cnn_lstm_twfix_bs512_lr1e3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_data = np.load(DATA_DIR / "X_data_6class_twfix.npy")
y_labels = np.load(DATA_DIR / "y_labels_6class_twfix.npy")
y_categorical = keras.utils.to_categorical(y_labels, num_classes=6)
label_map = json.load(open(DATA_DIR / "label_map_6class_twfix.json"))
reverse_label_map = {v: k for k, v in label_map.items()}

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 100  # LSTM converges slower than CNN; early stopping still governs
K_FOLDS = 5

print(f"Data: X={X_data.shape} y={y_labels.shape}")
print(f"Config: batch_size={BATCH_SIZE} lr={LEARNING_RATE} epochs={EPOCHS} k_folds={K_FOLDS}")

REG = regularizers.l1_l2(l1=1e-5, l2=1e-4)


def build_cnn_lstm_ensemble(input_shape=(200, 6), n_classes=6):
    inputs = layers.Input(shape=input_shape, name="input")

    # LSTM branch (paper Table II)
    lx = layers.LSTM(256, activation="tanh", return_sequences=False, name="lstm_layer")(inputs)
    lx = layers.Dense(128, activation="relu", kernel_regularizer=REG, name="lstm_dense1")(lx)
    lx = layers.Dropout(0.2, name="lstm_dropout1")(lx)
    lx = layers.Dense(64, activation="relu", kernel_regularizer=REG, name="lstm_dense2")(lx)
    lx = layers.Dropout(0.2, name="lstm_dropout2")(lx)
    lx = layers.Dense(32, activation="relu", kernel_regularizer=REG, name="lstm_dense3")(lx)
    lx = layers.Dropout(0.2, name="lstm_dropout3")(lx)
    lstm_output = layers.Dense(n_classes, activation="softmax", name="lstm_output")(lx)

    # CNN branch (paper Table II)
    cx = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d_layer")(inputs)
    cx = layers.MaxPooling1D(pool_size=2, name="maxpool_layer")(cx)
    cx = layers.Flatten(name="flatten_layer")(cx)
    cx = layers.Dense(1024, activation="relu", kernel_regularizer=REG, name="cnn_dense1")(cx)
    cx = layers.Dropout(0.2, name="cnn_dropout1")(cx)
    cx = layers.Dense(512, activation="relu", kernel_regularizer=REG, name="cnn_dense2")(cx)
    cx = layers.Dropout(0.2, name="cnn_dropout2")(cx)
    cnn_output = layers.Dense(n_classes, activation="softmax", name="cnn_output")(cx)

    outputs = layers.Average(name="ensemble_average")([lstm_output, cnn_output])
    return models.Model(inputs=inputs, outputs=outputs, name="FallNet_CNN_LSTM")


skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_results = []
fall_init_idx = label_map["Fall_Initiation"]

class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_labels), y=y_labels
)
class_weights = dict(enumerate(np.clip(class_weights_array, None, 3.0)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels), 1):
    print(f"\n{'='*80}\nFOLD {fold}/{K_FOLDS} - CNN-LSTM ENSEMBLE (twfix data)\n{'='*80}")
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
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(
        y_val_int, y_pred,
        target_names=[reverse_label_map[i] for i in range(6)],
        output_dict=True, zero_division=0,
    )
    fi_recall = report["Fall_Initiation"]["recall"]
    fi_f1 = report["Fall_Initiation"]["f1-score"]
    n_epochs_run = len(history.history["loss"])

    print(f"\nFold {fold} -> acc={val_acc:.4f} f1={val_f1:.4f} Fall_Init recall={fi_recall:.4f} ({n_epochs_run} epochs)")

    fold_results.append({
        "fold": fold, "epochs_run": n_epochs_run,
        "val_loss": val_loss, "val_accuracy": val_acc,
        "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1,
        "fall_init_recall": fi_recall, "fall_init_f1": fi_f1,
    })
    model.save(OUT_DIR / f"cnn_lstm_twfix_fold_{fold}.keras")

results_df = pd.DataFrame(fold_results)
mean_r = results_df.mean(numeric_only=True)
std_r = results_df.std(numeric_only=True)

summary = f"""
================================================================================
CNN-LSTM ENSEMBLE on twfix data, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}
================================================================================

Per-fold:
{results_df.to_string(index=False)}

Average Performance (5-fold CV):
  Accuracy:           {mean_r['val_accuracy']:.4f} +/- {std_r['val_accuracy']:.4f}
  Precision:          {mean_r['val_precision']:.4f} +/- {std_r['val_precision']:.4f}
  Recall:             {mean_r['val_recall']:.4f} +/- {std_r['val_recall']:.4f}
  F1-Score:           {mean_r['val_f1']:.4f} +/- {std_r['val_f1']:.4f}
  Fall_Init Recall:   {mean_r['fall_init_recall']:.4f} +/- {std_r['fall_init_recall']:.4f}

Comparison:
  CNN-LSTM (orig data, bs64, old run):     87.56% +/- 3.74% acc
  CNN-only (orig data, bs512/lr1e-3):      88.88% acc, 95.63% Fall_Init recall
  CNN-only (twfix data, bs512/lr1e-3):     88.83% acc, 96.60% Fall_Init recall
  CNN-LSTM (twfix data, this run):         {mean_r['val_accuracy']*100:.2f}% acc, {mean_r['fall_init_recall']*100:.2f}% Fall_Init recall
"""
print(summary)
with open(OUT_DIR / "retrain_summary.txt", "w") as f:
    f.write(summary)
results_df.to_json(OUT_DIR / "fold_results.json", orient="records", indent=2)
print(f"\nSaved to {OUT_DIR}/")
