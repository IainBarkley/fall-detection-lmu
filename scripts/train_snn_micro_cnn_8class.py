"""
Spiking Micro-CNN (snnTorch) on the paper-exact 8-class dataset.

Architecture and training methodology ported unchanged from the project's
existing 6-class SNN (notebooks/ReducingCNNWeights.ipynb, MicroCNN_SNN):
  - Same conv topology as the Micro-CNN (32/64/128 conv1d + BN), with
    Leaky-Integrate-and-Fire neurons in place of ReLU
  - 25 timesteps, constant current injection (input re-presented each step)
  - beta=0.95, threshold=1.0, fast_sigmoid(slope=25) surrogate gradient
  - Loss: CrossEntropy on the mean output membrane potential over time
  - Prediction: argmax of summed output spikes
  - Adam lr=5e-4, batch 64, 30 epochs, best-val-accuracy checkpoint per fold

Protocol matches the other 8-class runs: stratified 5-fold (random_state=42),
no class weights.

Reference points (same data/protocol unless noted):
  Micro-CNN ANN (41K):        91.44% acc, 98.54% Fall_Init recall
  CNN-LSTM ensemble (13.95M): 88.73% acc, 97.85% Fall_Init recall
  Prior SNN (6-class data):   88.78% acc, 93.3%  Fall_Init recall
"""
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

DATA_DIR = Path("fall_detection_data/processed")
OUT_DIR = Path("fall_detection_data/models/snn_micro_cnn_8class")
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_data = np.load(DATA_DIR / "X_data_8class_paper.npy").astype(np.float32)
y_labels = np.load(DATA_DIR / "y_labels_8class_paper.npy").astype(np.int64)
label_map = json.load(open(DATA_DIR / "label_map_8class_paper.json"))
reverse_label_map = {v: k for k, v in label_map.items()}
N_CLASSES = 8

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 5e-4
NUM_STEPS = 25
BETA = 0.95
THRESHOLD = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Data: X={X_data.shape} y={y_labels.shape} | device={device}")
print(f"Config: batch={BATCH_SIZE} epochs={EPOCHS} lr={LEARNING_RATE} steps={NUM_STEPS} beta={BETA}")


class MicroCNN_SNN(nn.Module):
    def __init__(self, num_classes=N_CLASSES, num_steps=NUM_STEPS, beta=BETA, threshold=THRESHOLD):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)
        self.pool3 = nn.MaxPool1d(2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=threshold)

        self.fc2 = nn.Linear(64, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False,
                                 threshold=threshold, output=True)

    def forward(self, x):
        # fresh membrane states per forward pass: passing None would make
        # snnTorch 0.9.x reuse internally-stored state from the previous
        # batch, dragging its freed autograd graph into this backward
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem_out = self.lif_out.init_leaky()
        spk_out_rec, mem_out_rec = [], []

        for _ in range(self.num_steps):
            cur1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.pool1(spk1)

            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.pool2(spk2)

            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.pool3(spk3)

            spk3 = self.global_pool(spk3).squeeze(-1)

            cur4 = self.fc1(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            cur_out = self.fc2(spk4)
            spk_out_step, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_rec.append(spk_out_step)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            spk_out, _ = model(data)
            preds = spk_out.sum(dim=0).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(targets)
    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
per_class_reports = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels), 1):
    print(f"\n{'='*80}\nFOLD {fold}/5 - SPIKING MICRO-CNN 8-CLASS\n{'='*80}")

    # torch conv1d wants (N, channels, time)
    X_train = torch.from_numpy(X_data[train_idx].transpose(0, 2, 1))
    X_val = torch.from_numpy(X_data[val_idx].transpose(0, 2, 1))
    y_train = torch.from_numpy(y_labels[train_idx])
    y_val = torch.from_numpy(y_labels[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

    model = MicroCNN_SNN().to(device)
    if fold == 1:
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            _, mem_out = model(data)
            loss = criterion(mem_out.mean(dim=0), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        preds, targs = evaluate(model, val_loader)
        val_acc = (preds == targs).mean()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"  epoch {epoch:2d}: loss={epoch_loss/len(train_loader):.4f} val_acc={val_acc:.4f}"
              f"{'  *best*' if val_acc == best_acc and best_state is not None else ''}")

    model.load_state_dict(best_state)
    preds, targs = evaluate(model, val_loader)
    report = classification_report(
        targs, preds, labels=list(range(N_CLASSES)),
        target_names=[reverse_label_map[i] for i in range(N_CLASSES)],
        output_dict=True, zero_division=0,
    )
    per_class_reports.append(report)
    fi = report["Fall_Initiation"]
    acc = report["accuracy"]
    print(f"\nFold {fold} (best ckpt) -> acc={acc:.4f} Fall_Init recall={fi['recall']:.4f} f1={fi['f1-score']:.4f}")

    fold_results.append({
        "fold": fold, "val_accuracy": acc,
        "fall_init_recall": fi["recall"], "fall_init_f1": fi["f1-score"],
    })
    torch.save(model.state_dict(), OUT_DIR / f"snn_micro_cnn_8class_fold_{fold}.pth")

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
SPIKING MICRO-CNN (snnTorch) on paper-exact 8-class dataset
{NUM_STEPS} timesteps, beta={BETA}, threshold={THRESHOLD}, batch={BATCH_SIZE}, lr={LEARNING_RATE}, {EPOCHS} epochs
Loss: CE on mean output membrane | Prediction: output spike count argmax
================================================================================

Per-fold (best-val checkpoint):
{results_df.to_string(index=False)}

Average Performance (5-fold CV):
  Accuracy:           {mean_r['val_accuracy']:.4f} +/- {std_r['val_accuracy']:.4f}
  Fall_Init Recall:   {mean_r['fall_init_recall']:.4f} +/- {std_r['fall_init_recall']:.4f}
  Fall_Init F1:       {mean_r['fall_init_f1']:.4f} +/- {std_r['fall_init_f1']:.4f}

Per-class (mean over 5 folds):
{class_df.to_string(index=False)}

Comparison (same data/protocol unless noted):
  Micro-CNN ANN (41K params):        91.44% acc, 98.54% Fall_Init recall
  CNN-LSTM ensemble (13.95M params): 88.73% acc, 97.85% Fall_Init recall
  Prior SNN (6-class, buggy data):   88.78% acc, 93.3%  Fall_Init recall
  Spiking Micro-CNN (this run):      {mean_r['val_accuracy']*100:.2f}% acc, {mean_r['fall_init_recall']*100:.2f}% Fall_Init recall
"""
print(summary)
with open(OUT_DIR / "summary.txt", "w") as f:
    f.write(summary)
results_df.to_json(OUT_DIR / "fold_results.json", orient="records", indent=2)
class_df.to_json(OUT_DIR / "per_class_results.json", orient="records", indent=2)
print(f"\nSaved to {OUT_DIR}/")
