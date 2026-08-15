"""
Quantize the 8-class paper-dataset Micro-CNN folds
(fall_detection_data/models/micro_cnn_8class_paper/) to Float16 and full INT8
TFLite, and evaluate each variant on the fold's own validation split (same
StratifiedKFold random_state=42 used in training), so results are directly
comparable to the FP32 numbers (91.44% acc, 98.54% Fall_Init recall).

INT8 uses full integer quantization with int8 input/output (what TFLite Micro
on the Arduino needs), calibrated on 300 training samples per fold.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

DATA_DIR = Path("fall_detection_data/processed")
MODEL_DIR = Path("fall_detection_data/models/micro_cnn_8class_paper")
QUANT_DIR = MODEL_DIR / "quantized"
QUANT_DIR.mkdir(exist_ok=True)

X_data = np.load(DATA_DIR / "X_data_8class_paper.npy").astype(np.float32)
y_labels = np.load(DATA_DIR / "y_labels_8class_paper.npy")
label_map = json.load(open(DATA_DIR / "label_map_8class_paper.json"))
reverse_label_map = {v: k for k, v in label_map.items()}
N_CLASSES = 8

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def convert_float16(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_int8(model, X_repr):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(len(X_repr)):
            yield [X_repr[i:i + 1]]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def eval_tflite(tflite_bytes, X_val, y_val_int):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    preds = np.empty(len(X_val), dtype=np.int64)
    for i in range(len(X_val)):
        x = X_val[i:i + 1]
        if inp["dtype"] == np.int8:
            scale, zero_point = inp["quantization"]
            x = np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        preds[i] = int(np.argmax(interpreter.get_tensor(out["index"])[0]))

    report = classification_report(
        y_val_int, preds,
        labels=list(range(N_CLASSES)),
        target_names=[reverse_label_map[i] for i in range(N_CLASSES)],
        output_dict=True, zero_division=0,
    )
    return report["accuracy"], report["Fall_Initiation"]["recall"], report["Fall_Initiation"]["f1-score"]


rows = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels), 1):
    print(f"\n{'='*70}\nFOLD {fold}/5\n{'='*70}")
    keras.backend.clear_session()
    model = keras.models.load_model(MODEL_DIR / f"micro_cnn_8class_fold_{fold}.keras")
    X_val, y_val_int = X_data[val_idx], y_labels[val_idx]

    # FP32 reference on the same split
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(
        y_val_int, y_pred, labels=list(range(N_CLASSES)),
        target_names=[reverse_label_map[i] for i in range(N_CLASSES)],
        output_dict=True, zero_division=0,
    )
    fp32_acc = report["accuracy"]
    fp32_fi = report["Fall_Initiation"]["recall"]

    # Float16
    f16_bytes = convert_float16(model)
    f16_path = QUANT_DIR / f"micro_cnn_8class_fold_{fold}_float16.tflite"
    f16_path.write_bytes(f16_bytes)
    f16_acc, f16_fi, f16_fi_f1 = eval_tflite(f16_bytes, X_val, y_val_int)
    print(f"  Float16: acc={f16_acc:.4f} FI_recall={f16_fi:.4f} size={len(f16_bytes)/1024:.1f} KB")

    # INT8 (calibrate on 300 training samples)
    rng = np.random.default_rng(42)
    repr_idx = rng.choice(train_idx, size=300, replace=False)
    i8_bytes = convert_int8(model, X_data[repr_idx])
    i8_path = QUANT_DIR / f"micro_cnn_8class_fold_{fold}_int8.tflite"
    i8_path.write_bytes(i8_bytes)
    i8_acc, i8_fi, i8_fi_f1 = eval_tflite(i8_bytes, X_val, y_val_int)
    print(f"  INT8:    acc={i8_acc:.4f} FI_recall={i8_fi:.4f} size={len(i8_bytes)/1024:.1f} KB")

    rows.append({
        "fold": fold,
        "fp32_acc": fp32_acc, "fp32_fi_recall": fp32_fi,
        "f16_acc": f16_acc, "f16_fi_recall": f16_fi, "f16_kb": len(f16_bytes) / 1024,
        "int8_acc": i8_acc, "int8_fi_recall": i8_fi, "int8_kb": len(i8_bytes) / 1024,
    })

df = pd.DataFrame(rows)
m, s = df.mean(numeric_only=True), df.std(numeric_only=True)

summary = f"""
================================================================================
MICRO-CNN 8-CLASS (paper-exact data) - QUANTIZATION RESULTS
================================================================================

Per-fold:
{df.to_string(index=False)}

5-fold averages:
              Accuracy              Fall_Init Recall      Size
  FP32:       {m['fp32_acc']:.4f} +/- {s['fp32_acc']:.4f}   {m['fp32_fi_recall']:.4f} +/- {s['fp32_fi_recall']:.4f}   (keras ~553 KB)
  Float16:    {m['f16_acc']:.4f} +/- {s['f16_acc']:.4f}   {m['f16_fi_recall']:.4f} +/- {s['f16_fi_recall']:.4f}   {m['f16_kb']:.1f} KB
  INT8:       {m['int8_acc']:.4f} +/- {s['int8_acc']:.4f}   {m['int8_fi_recall']:.4f} +/- {s['int8_fi_recall']:.4f}   {m['int8_kb']:.1f} KB

Drop from FP32:
  Float16: {(m['fp32_acc']-m['f16_acc'])*100:+.2f} pts accuracy, {(m['fp32_fi_recall']-m['f16_fi_recall'])*100:+.2f} pts Fall_Init recall
  INT8:    {(m['fp32_acc']-m['int8_acc'])*100:+.2f} pts accuracy, {(m['fp32_fi_recall']-m['int8_fi_recall'])*100:+.2f} pts Fall_Init recall

Reference (6-class, buggy data): FP32 94.71% / F16 94.71% (~85 KB) / INT8 93.96% (~56 KB)
"""
print(summary)
with open(QUANT_DIR / "quantization_summary.txt", "w") as f:
    f.write(summary)
df.to_json(QUANT_DIR / "quantization_results.json", orient="records", indent=2)
print(f"Saved to {QUANT_DIR}/")
