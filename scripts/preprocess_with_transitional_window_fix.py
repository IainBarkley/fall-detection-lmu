"""
Re-run of the preprocessing pipeline (notebooks/02_DataExplorationBothDataSets.ipynb,
cells "Algorithm 1: Temporal Feature Extraction" + "Data Loading and Preprocessing")
with ONE fix applied.

Bug in the original: process_fall_activity() extracted EITHER the transitional
window (0.5*Sw) OR the full Fall_Initiation window (Sw), chosen by a coin flip
per fall event:

    if np.random.random() < 0.5:
        ... use transitional window ...
    else:
        ... use full window ...

But Jain & Semwal (2022), Section III.B, extract BOTH as separate Fall_Initiation
training samples: the full window captures the fall itself, and the transitional
window (Sp to the midpoint of Sw) is extracted *in addition*, specifically to
train the ADL-to-fall-initiation transition and reduce reaction time. The coin
flip was silently halving the amount of Fall_Initiation training data instead of
doubling it as the paper does.

Fix: emit both samples unconditionally for every fall event.

Outputs land in fall_detection_data/processed/ with a "_twfix" suffix so the
original _6class files are left untouched for comparison.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
import json
from collections import Counter
from tensorflow import keras

base_dir = Path("fall_detection_data")
kfall_sensor_dir = base_dir / "KFall" / "sensor_data"
sisfall_dir = base_dir / "SisFall"
processed_dir = base_dir / "processed"

kfall_fall_activities = ['T28', 'T30', 'T31', 'T32', 'T33', 'T34']
kfall_stumble = ['T10']

sisfall_adl_map = {
    'D01': 'Walking', 'D02': 'Walking',
    'D03': 'Jogging', 'D04': 'Jogging',
    'D05': 'Walking_stairs_updown', 'D06': 'Walking_stairs_updown',
    'D18': 'Stumble_while_walking',
}
sisfall_falls = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06']

label_map = {
    'Walking': 0, 'Jogging': 1, 'Walking_stairs_updown': 2,
    'Stumble_while_walking': 3, 'Fall_Recovery': 4,
    'Fall_Initiation': 5, 'Impact': 6, 'Aftermath': 7,
}


def load_sisfall_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip().replace(';', '').replace(',', ' ')
        values = line.split()
        if len(values) == 9:
            data.append([float(v) for v in values])
    if len(data) == 0:
        return None
    data = np.array(data)
    converted = np.zeros((data.shape[0], 6))
    adxl_factor = (2 * 16) / (2 ** 13)
    converted[:, 0:3] = data[:, 0:3] * adxl_factor
    itg_factor = (2 * 2000) / (2 ** 16)
    converted[:, 3:6] = data[:, 3:6] * itg_factor
    return converted


def load_kfall_file(filepath):
    try:
        df = pd.read_csv(filepath)
        return df[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']].values
    except Exception:
        return None


def upsample_to_200hz(data, original_freq=100):
    n_samples, n_features = data.shape
    original_time = np.arange(n_samples) / original_freq
    target_time = np.arange(0, n_samples / original_freq, 1 / 200)
    upsampled = np.zeros((len(target_time), n_features))
    for i in range(n_features):
        cs = CubicSpline(original_time, data[:, i])
        upsampled[:, i] = cs(target_time)
    return upsampled


def extract_temporal_features(data, sampling_freq=200):
    acc_y = data[:, 1]
    W_s = sampling_freq // 4
    std_devs = []
    for i in range(0, len(acc_y) - W_s, W_s):
        std_devs.append(np.std(acc_y[i:i + W_s]))
    if len(std_devs) == 0:
        return None
    Sp = int(np.argmax(std_devs))
    return {
        'W_s': W_s,
        'fall_init_start': Sp * W_s,
        'fall_init_end': min((Sp + 4) * W_s, len(data)),
        'transitional_end': min((Sp + 2) * W_s, len(data)),
        'impact_start': min((Sp + 4) * W_s, len(data)),
        'impact_end': min((Sp + 8) * W_s, len(data)),
        'aftermath_start': min((Sp + 8) * W_s, len(data)),
    }


def interp_to_200(segment):
    if len(segment) == 200:
        return segment
    time_orig = np.linspace(0, 1, len(segment))
    time_new = np.linspace(0, 1, 200)
    out = np.zeros((200, 6))
    for i in range(6):
        out[:, i] = np.interp(time_new, time_orig, segment[:, i])
    return out


def process_fall_activity(data):
    """FIXED: emits BOTH the transitional window and the full window as
    separate Fall_Initiation samples (paper Section III.B), instead of the
    original coin-flip that emitted only one of the two."""
    segments = extract_temporal_features(data)
    if segments is None:
        return []
    results = []

    adl_start = max(0, segments['fall_init_start'] - 200)
    adl_end = segments['fall_init_start']
    if adl_end - adl_start >= 200 and adl_start >= 0:
        adl_segment = data[adl_start:adl_end]
        acc_std = np.std(adl_segment[:, 1])
        adl_label = label_map['Jogging'] if acc_std > 0.5 else label_map['Walking']
        results.append((adl_segment[:200], adl_label))

    fi_start = segments['fall_init_start']

    # (a) transitional window (0.5s) -- early-detection signal
    tw_end = segments['transitional_end']
    if tw_end <= len(data) and (tw_end - fi_start) >= 100:
        tw_segment = data[fi_start:tw_end]
        results.append((interp_to_200(tw_segment), label_map['Fall_Initiation']))

    # (b) full Fall_Initiation window (1s) -- ALWAYS ALSO extracted (the fix)
    fi_end = segments['fall_init_end']
    if fi_end <= len(data) and (fi_end - fi_start) >= 200:
        fi_segment = data[fi_start:fi_end]
        results.append((fi_segment[:200], label_map['Fall_Initiation']))

    impact_start = segments['impact_start']
    impact_end = segments['impact_end']
    if impact_end <= len(data) and (impact_end - impact_start) >= 200:
        results.append((data[impact_start:impact_end][:200], label_map['Impact']))

    aftermath_start = segments['aftermath_start']
    if len(data) - aftermath_start >= 200:
        results.append((data[aftermath_start:aftermath_start + 200], label_map['Aftermath']))

    return results


def process_stumble_activity(data):
    segments = extract_temporal_features(data)
    if segments is None:
        return []
    results = []
    stumble_start = segments['fall_init_start']
    stumble_end = segments['transitional_end']
    if stumble_end <= len(data) and (stumble_end - stumble_start) >= 100:
        results.append((interp_to_200(data[stumble_start:stumble_end]), label_map['Stumble_while_walking']))
    recovery_start = segments['transitional_end']
    recovery_end = segments['impact_end']
    if recovery_end <= len(data) and (recovery_end - recovery_start) >= 200:
        results.append((data[recovery_start:recovery_end][:200], label_map['Fall_Recovery']))
    return results


def process_adl_activity(data, label_name):
    results = []
    label = label_map[label_name]
    max_samples = min(len(data), 4000)
    for i in range(0, max_samples - 200, 200):
        segment = data[i:i + 200]
        if len(segment) == 200:
            results.append((segment, label))
    return results


def process_kfall_dataset():
    print("=" * 80)
    print("PROCESSING KFALL DATASET")
    print("=" * 80)
    X_data, y_labels = [], []
    subjects = sorted([d for d in kfall_sensor_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subjects")
    for si, subject_dir in enumerate(subjects, 1):
        files = list(subject_dir.glob("*.csv"))
        for file in files:
            filename = file.stem
            if len(filename) < 6:
                continue
            activity_code = filename[3:6]
            data = load_kfall_file(file)
            if data is None or len(data) < 100:
                continue
            try:
                data_upsampled = upsample_to_200hz(data)
                if activity_code in kfall_fall_activities:
                    features = process_fall_activity(data_upsampled)
                elif activity_code in kfall_stumble:
                    features = process_stumble_activity(data_upsampled)
                else:
                    continue
                for segment, label in features:
                    if segment.shape == (200, 6):
                        X_data.append(segment)
                        y_labels.append(label)
            except Exception as e:
                print(f"  Error processing {file.name}: {e}")
                continue
        if si % 8 == 0:
            print(f"  ...{si}/{len(subjects)} subjects done")
    return np.array(X_data), np.array(y_labels)


def process_sisfall_dataset():
    print("=" * 80)
    print("PROCESSING SISFALL DATASET")
    print("=" * 80)
    X_data, y_labels = [], []
    subjects = sorted([d for d in sisfall_dir.iterdir()
                        if d.is_dir() and (d.name.startswith('SA') or d.name.startswith('SE'))])
    print(f"Found {len(subjects)} subjects")
    for si, subject_dir in enumerate(subjects, 1):
        files = list(subject_dir.glob("*.txt"))
        for file in files:
            filename = file.stem
            parts = filename.split('_')
            if len(parts) < 2:
                continue
            activity_code = parts[0]
            data = load_sisfall_file(file)
            if data is None or len(data) < 200:
                continue
            try:
                if activity_code in sisfall_adl_map:
                    features = process_adl_activity(data, sisfall_adl_map[activity_code])
                elif activity_code in sisfall_falls:
                    features = process_fall_activity(data)
                else:
                    continue
                for segment, label in features:
                    if segment.shape == (200, 6):
                        X_data.append(segment)
                        y_labels.append(label)
            except Exception:
                continue
        if si % 8 == 0:
            print(f"  ...{si}/{len(subjects)} subjects done")
    return np.array(X_data), np.array(y_labels)


def normalize_and_fuse(X_kfall, y_kfall, X_sisfall, y_sisfall):
    n_kfall, ts, feat = X_kfall.shape
    n_sisfall = X_sisfall.shape[0]
    scaler_kfall = StandardScaler()
    X_kfall_norm = scaler_kfall.fit_transform(X_kfall.reshape(-1, feat)).reshape(n_kfall, ts, feat)
    scaler_sisfall = StandardScaler()
    X_sisfall_norm = scaler_sisfall.fit_transform(X_sisfall.reshape(-1, feat)).reshape(n_sisfall, ts, feat)
    X_fused = np.concatenate([X_kfall_norm, X_sisfall_norm], axis=0)
    y_fused = np.concatenate([y_kfall, y_sisfall], axis=0)
    scaler_final = StandardScaler()
    X_fused_norm = scaler_final.fit_transform(X_fused.reshape(-1, feat)).reshape(-1, ts, feat)
    return X_fused_norm, y_fused


if __name__ == "__main__":
    X_kfall, y_kfall = process_kfall_dataset()
    print(f"KFall: X={X_kfall.shape} y={y_kfall.shape}")

    X_sisfall, y_sisfall = process_sisfall_dataset()
    print(f"SisFall: X={X_sisfall.shape} y={y_sisfall.shape}")

    X_final, y_final = normalize_and_fuse(X_kfall, y_kfall, X_sisfall, y_sisfall)
    print(f"Fused (8-class): X={X_final.shape} y={y_final.shape}")
    print("8-class distribution:", dict(sorted(Counter(y_final).items())))

    # --- merge to 6-class (identical to notebook cells 32/33) ---
    y_labels = y_final.copy()
    y_labels[y_labels == 7] = 6  # Aftermath -> Impact
    mask = y_labels != 4  # drop Fall_Recovery
    X_data = X_final[mask]
    y_labels = y_labels[mask]
    y_labels[y_labels > 4] -= 1  # shift 5,6 -> 4,5

    label_map_6class = {
        'Walking': 0, 'Jogging': 1, 'Walking_stairs_updown': 2,
        'Stumble_while_walking': 3, 'Fall_Initiation': 4, 'Impact_Aftermath': 5,
    }
    print("6-class distribution:", dict(sorted(Counter(y_labels).items())))

    np.save(processed_dir / "X_data_6class_twfix.npy", X_data)
    np.save(processed_dir / "y_labels_6class_twfix.npy", y_labels)
    with open(processed_dir / "label_map_6class_twfix.json", "w") as f:
        json.dump(label_map_6class, f, indent=2)

    print(f"\nSaved: X_data_6class_twfix.npy {X_data.shape}, y_labels_6class_twfix.npy {y_labels.shape}")
    print(f"For comparison, original _6class.npy had 16,732 samples total.")
