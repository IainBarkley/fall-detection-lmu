"""Compute per-class metrics (esp. Fall_Initiation recall) for a set of saved
5-fold CNN-only models, using the same StratifiedKFold split (random_state=42)
used at training time, so each fold's validation set matches what it was
trained against."""
import sys
import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

model_dir = Path(sys.argv[1])
data_suffix = sys.argv[2] if len(sys.argv) > 2 else ""  # "" or "_twfix"

DATA_DIR = Path("fall_detection_data/processed")
X_data = np.load(DATA_DIR / f"X_data_6class{data_suffix}.npy")
y_labels = np.load(DATA_DIR / f"y_labels_6class{data_suffix}.npy")
label_map = json.load(open(DATA_DIR / f"label_map_6class{data_suffix}.json"))
reverse_label_map = {v: k for k, v in label_map.items()}
fall_init_idx = label_map["Fall_Initiation"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fall_init_recalls = []
overall_accs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels), 1):
    model_path = model_dir / f"cnn_only_bs512_fold_{fold}.keras"
    if not model_path.exists():
        print(f"missing {model_path}")
        continue
    model = keras.models.load_model(model_path)
    X_val, y_val = X_data[val_idx], y_labels[val_idx]
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    report = classification_report(
        y_val, y_pred,
        target_names=[reverse_label_map[i] for i in range(6)],
        output_dict=True, zero_division=0,
    )
    fi_recall = report["Fall_Initiation"]["recall"]
    fi_f1 = report["Fall_Initiation"]["f1-score"]
    acc = report["accuracy"]
    fall_init_recalls.append(fi_recall)
    overall_accs.append(acc)
    print(f"Fold {fold}: acc={acc:.4f}  Fall_Initiation recall={fi_recall:.4f}  f1={fi_f1:.4f}")
    keras.backend.clear_session()

print(f"\n{model_dir.name}")
print(f"  Mean accuracy:            {np.mean(overall_accs):.4f} +/- {np.std(overall_accs):.4f}")
print(f"  Mean Fall_Initiation recall: {np.mean(fall_init_recalls):.4f} +/- {np.std(fall_init_recalls):.4f}")
