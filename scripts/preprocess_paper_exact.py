"""
Paper-exact preprocessing for replicating Jain & Semwal (2022) FallNet.

Differences from the project's existing pipeline
(notebooks/02_DataExplorationBothDataSets.ipynb):

1. 8 classes kept as in the paper (Fall_Recovery retained, Impact/Aftermath
   NOT merged).
2. Stumble (SisFall D18, KFall T10) is segmented with Algorithm 1: the
   high-variance moment (Sp..transitional_end) becomes Stumble_while_walking
   and the following recovery period becomes Fall_Recovery. The old pipeline
   windowed D18 like an ADL, mislabeling ~19 ordinary-walking windows per
   trial as "stumble".
3. ADLs capped at 20 seconds per (subject, task) TOTAL across trials, per the
   paper ("Twenty seconds' worth of ADL data is directly extracted"). This
   reproduces the paper's test supports (Walking ~304/fold = 38 subj x 2
   tasks x 20 windows / 5). The old pipeline capped 20 s per FILE, giving
   5x over-extraction for multi-trial tasks (stairs).
4. Fall events emit BOTH the transitional window and the full window as
   Fall_Initiation (paper's Tw scheme; support 658 = 2 x Impact's 329).
5. No ADL-before-fall segments (not described in the paper).

Deliberately does not import TensorFlow so it can run on CPU while a training
job holds the GPU.

Outputs: fall_detection_data/processed/{X_data,y_labels}_8class_paper.npy
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler

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
}
sisfall_stumble = ['D18']
sisfall_falls = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06']

label_map = {
    'Walking': 0, 'Jogging': 1, 'Walking_stairs_updown': 2,
    'Stumble_while_walking': 3, 'Fall_Recovery': 4,
    'Fall_Initiation': 5, 'Impact': 6, 'Aftermath': 7,
}

MAX_ADL_WINDOWS_PER_TASK = 20  # 20 s per (subject, task) across all trials


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
    converted[:, 0:3] = data[:, 0:3] * ((2 * 16) / (2 ** 13))    # ADXL345
    converted[:, 3:6] = data[:, 3:6] * ((2 * 2000) / (2 ** 16))  # ITG3200
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
    """Fall_Initiation (Tw AND full window), Impact, Aftermath. No ADL-before."""
    segments = extract_temporal_features(data)
    if segments is None:
        return []
    results = []
    fi_start = segments['fall_init_start']

    tw_end = segments['transitional_end']
    if tw_end <= len(data) and (tw_end - fi_start) >= 100:
        results.append((interp_to_200(data[fi_start:tw_end]), label_map['Fall_Initiation']))

    fi_end = segments['fall_init_end']
    if fi_end <= len(data) and (fi_end - fi_start) >= 200:
        results.append((data[fi_start:fi_end][:200], label_map['Fall_Initiation']))

    impact_start, impact_end = segments['impact_start'], segments['impact_end']
    if impact_end <= len(data) and (impact_end - impact_start) >= 200:
        results.append((data[impact_start:impact_end][:200], label_map['Impact']))

    aftermath_start = segments['aftermath_start']
    if len(data) - aftermath_start >= 200:
        results.append((data[aftermath_start:aftermath_start + 200], label_map['Aftermath']))
    return results


def process_stumble_activity(data):
    """Algorithm 1 on stumble trials: stumble moment + recovery period."""
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


def process_sisfall():
    X, y = [], []
    subjects = sorted([d for d in sisfall_dir.iterdir()
                        if d.is_dir() and (d.name.startswith('SA') or d.name.startswith('SE'))])
    print(f"SisFall: {len(subjects)} subjects")
    adl_windows_used = defaultdict(int)  # (subject, task) -> windows emitted

    for si, subject_dir in enumerate(subjects, 1):
        for file in sorted(subject_dir.glob("*.txt")):
            parts = file.stem.split('_')
            if len(parts) < 2:
                continue
            task = parts[0]
            if task in sisfall_adl_map:
                key = (subject_dir.name, task)
                if adl_windows_used[key] >= MAX_ADL_WINDOWS_PER_TASK:
                    continue
                data = load_sisfall_file(file)
                if data is None or len(data) < 200:
                    continue
                label = label_map[sisfall_adl_map[task]]
                for i in range(0, len(data) - 200, 200):
                    if adl_windows_used[key] >= MAX_ADL_WINDOWS_PER_TASK:
                        break
                    X.append(data[i:i + 200])
                    y.append(label)
                    adl_windows_used[key] += 1
            elif task in sisfall_stumble:
                data = load_sisfall_file(file)
                if data is None or len(data) < 200:
                    continue
                for seg, label in process_stumble_activity(data):
                    if seg.shape == (200, 6):
                        X.append(seg)
                        y.append(label)
            elif task in sisfall_falls:
                data = load_sisfall_file(file)
                if data is None or len(data) < 200:
                    continue
                for seg, label in process_fall_activity(data):
                    if seg.shape == (200, 6):
                        X.append(seg)
                        y.append(label)
        if si % 8 == 0:
            print(f"  ...{si}/{len(subjects)}")
    return np.array(X), np.array(y)


def process_kfall():
    X, y = [], []
    subjects = sorted([d for d in kfall_sensor_dir.iterdir() if d.is_dir()])
    print(f"KFall: {len(subjects)} subjects")
    for si, subject_dir in enumerate(subjects, 1):
        for file in sorted(subject_dir.glob("*.csv")):
            filename = file.stem
            if len(filename) < 6:
                continue
            activity_code = filename[3:6]
            if activity_code not in kfall_fall_activities and activity_code not in kfall_stumble:
                continue
            data = load_kfall_file(file)
            if data is None or len(data) < 100:
                continue
            try:
                data = upsample_to_200hz(data)
                if activity_code in kfall_fall_activities:
                    feats = process_fall_activity(data)
                else:
                    feats = process_stumble_activity(data)
                for seg, label in feats:
                    if seg.shape == (200, 6):
                        X.append(seg)
                        y.append(label)
            except Exception as e:
                print(f"  Error {file.name}: {e}")
        if si % 8 == 0:
            print(f"  ...{si}/{len(subjects)}")
    return np.array(X), np.array(y)


def normalize_and_fuse(X_a, y_a, X_b, y_b):
    feat = X_a.shape[2]
    X_a_n = StandardScaler().fit_transform(X_a.reshape(-1, feat)).reshape(X_a.shape)
    X_b_n = StandardScaler().fit_transform(X_b.reshape(-1, feat)).reshape(X_b.shape)
    X = np.concatenate([X_a_n, X_b_n], axis=0)
    y = np.concatenate([y_a, y_b], axis=0)
    X = StandardScaler().fit_transform(X.reshape(-1, feat)).reshape(X.shape)
    return X, y


if __name__ == "__main__":
    X_kfall, y_kfall = process_kfall()
    print(f"KFall: {X_kfall.shape}")
    X_sisfall, y_sisfall = process_sisfall()
    print(f"SisFall: {X_sisfall.shape}")

    X, y = normalize_and_fuse(X_kfall, y_kfall, X_sisfall, y_sisfall)
    reverse = {v: k for k, v in label_map.items()}
    counts = Counter(y)
    print(f"\nFused 8-class dataset: {X.shape}")
    print("Class distribution (paper test supports x5 in parentheses):")
    paper_totals = {0: 1520, 1: 1480, 2: 1540, 3: 280, 4: 275, 5: 3290, 6: 1645, 7: 1645}
    for cls in sorted(counts):
        print(f"  {cls} {reverse[cls]:<25s}: {counts[cls]:>5d}  (paper ~{paper_totals[cls]})")

    np.save(processed_dir / "X_data_8class_paper.npy", X)
    np.save(processed_dir / "y_labels_8class_paper.npy", y)
    with open(processed_dir / "label_map_8class_paper.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nSaved X_data_8class_paper.npy {X.shape}, y_labels_8class_paper.npy {y.shape}")
