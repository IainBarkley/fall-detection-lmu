# %% [markdown]
# ## FallNet Phase 2: LMU-SNN Hybrid
#
# Architecture:
#   IMU [batch, 6, 200]
#     → LMUEncoder (one LMUCell per IMU channel, processes 200 timesteps)
#     → [batch, lmu_hidden × 6]  compact Legendre temporal encoding
#     → Spiking FC classifier (3 × Linear → BN → LIF, integrated over num_steps)
#     → Spike accumulation → [batch, 6] class logits
#
# Theory: Fourier → Wavelets → LMU (Padé/Legendre basis) → SNN (LIF integration)

# %%
# --- Cell 1: Imports ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
from scipy.signal import cont2discrete

print("✅ Imports OK")

# %%
# --- Cell 2: LMU Cell ---

class LMUCell(nn.Module):
    """
    Single-step Legendre Memory Unit.

    Per timestep:
        e_t  = tanh(W_x @ x_t + W_h @ h_{t-1} + W_m @ m_{t-1})   # encoder
        m_t  = A_bar @ m_{t-1} + B_bar @ e_t                       # memory update
        h_t  = tanh(H_x @ x_t + H_m @ m_t)                        # hidden state

    A_bar, B_bar are analytically derived (Voelker et al. 2019) via
    zero-order-hold discretisation of the continuous Legendre delay ODE.
    They are frozen by default — the network learns only the coupling weights.
    """

    def __init__(self, input_size, hidden_size, order, theta, learn_ab=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.order = order

        # --- Derive A_bar, B_bar analytically ---
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A_cont = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R / theta
        B_cont = (-1.0) ** Q[:, None] * R / theta

        C = np.zeros((1, order))
        D = np.zeros((1,))
        A_bar, B_bar, _, _, _ = cont2discrete(
            (A_cont, B_cont, C, D), dt=1.0, method='zoh'
        )

        if learn_ab:
            self.A_bar = nn.Parameter(torch.FloatTensor(A_bar))
            self.B_bar = nn.Parameter(torch.FloatTensor(B_bar))
        else:
            self.register_buffer('A_bar', torch.FloatTensor(A_bar))
            self.register_buffer('B_bar', torch.FloatTensor(B_bar))

        # Encoder weights
        self.e_x = nn.Linear(input_size,  1, bias=False)
        self.e_h = nn.Linear(hidden_size, 1, bias=False)
        self.e_m = nn.Linear(order,       1, bias=False)

        # Hidden weights
        self.h_x = nn.Linear(input_size, hidden_size, bias=False)
        self.h_m = nn.Linear(order,      hidden_size, bias=False)

    def forward(self, x, state):
        h, m = state
        u     = torch.tanh(self.e_x(x) + self.e_h(h) + self.e_m(m))   # [B, 1]
        m_new = F.linear(m, self.A_bar) + F.linear(u, self.B_bar.T)    # [B, order]
        h_new = torch.tanh(self.h_x(x) + self.h_m(m_new))              # [B, hidden]
        return h_new, (h_new, m_new)

    def init_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.order,       device=device),
        )


print("✅ LMUCell defined")

# %%
# --- Cell 3: LMU Encoder ---

class LMUEncoder(nn.Module):
    """
    Runs one LMUCell per IMU channel over the full 200-timestep window.
    Each channel independently learns its Legendre memory representation.
    Outputs are concatenated: [batch, hidden_size * in_channels].
    """

    def __init__(self, in_channels=6, hidden_size=32, order=8,
                 theta=200, learn_ab=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.output_size = hidden_size * in_channels

        self.lmu_cells = nn.ModuleList([
            LMUCell(1, hidden_size, order, theta, learn_ab)
            for _ in range(in_channels)
        ])

    def forward(self, x):
        # x: [batch, in_channels, seq_len]
        batch_size = x.size(0)
        device     = x.device

        states = [cell.init_state(batch_size, device) for cell in self.lmu_cells]

        for t in range(x.size(2)):                          # step over 200 timesteps
            for c, cell in enumerate(self.lmu_cells):
                _, states[c] = cell(x[:, c:c+1, t], states[c])

        h_finals = [states[c][0] for c in range(self.in_channels)]
        return torch.cat(h_finals, dim=-1)                  # [batch, hidden * channels]


print("✅ LMUEncoder defined")

# %%
# --- Cell 4: FallNet LMU-SNN Model ---

class FallNet_LMU_SNN(nn.Module):
    """
    Stage 1 (LMU):  Encodes 200-timestep IMU window into Legendre state
    Stage 2 (SNN):  Integrates that state over num_steps via spiking LIF layers
    """

    def __init__(
        self,
        num_classes = 6,
        num_steps   = 25,
        lmu_hidden  = 32,    # hidden size per channel → 192 total (32×6)
        lmu_order   = 8,     # Legendre polynomial order
        lmu_theta   = 200,   # match your sequence length
        beta        = 0.95,
        threshold   = 1.0,
        learn_ab    = False,
    ):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Stage 1: LMU encoder
        self.lmu_encoder = LMUEncoder(
            in_channels=6,
            hidden_size=lmu_hidden,
            order=lmu_order,
            theta=lmu_theta,
            learn_ab=learn_ab,
        )
        lmu_out = lmu_hidden * 6  # 192 with defaults

        # Stage 2: Spiking classifier
        self.fc1  = nn.Linear(lmu_out, 128)
        self.bn1  = nn.BatchNorm1d(128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               threshold=threshold, learn_beta=True)

        self.fc2  = nn.Linear(128, 64)
        self.bn2  = nn.BatchNorm1d(64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               threshold=threshold, learn_beta=True)

        self.fc3  = nn.Linear(64, num_classes)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               threshold=threshold, learn_beta=True)

    def forward(self, x):
        # LMU encoding runs once (not per SNN step)
        lmu_out = self.lmu_encoder(x)           # [batch, 192]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec, mem_rec = [], []

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.bn1(self.fc1(lmu_out)), mem1)
            spk2, mem2 = self.lif2(self.bn2(self.fc2(spk1)),    mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2),              mem3)
            spk_rec.append(spk3)
            mem_rec.append(mem3)

        return torch.stack(spk_rec), torch.stack(mem_rec)


# Sanity check
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model     = FallNet_LMU_SNN().to(device)
_x         = torch.randn(4, 6, 200).to(device)
with torch.no_grad():
    _spk, _mem = _model(_x)

total_p = sum(p.numel() for p in _model.parameters())
lmu_p   = sum(p.numel() for p in _model.lmu_encoder.parameters())
print(f"✅ FallNet_LMU_SNN defined")
print(f"   Input:  {list(_x.shape)}")
print(f"   Output: {list(_spk.shape)}  [steps, batch, classes]")
print(f"   Params: {total_p:,} total  ({lmu_p:,} LMU + {total_p-lmu_p:,} SNN)")
del _model, _x, _spk, _mem

# %%
# --- Cell 5: Dataset + Class Weights ---

class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Assumes X_data_torch, y_labels_torch already defined from your preprocessing cell
cw = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(6),
    y=y_labels_torch.numpy()
)
cw = np.clip(cw, None, 3.0)
cw_tensor = torch.FloatTensor(cw).to(device)

print("Class weights (capped at 3×):")
for i, w in enumerate(cw):
    print(f"  {reverse_label_map[i]:<30s}: {w:.3f}×")

dataset = FallDataset(X_data_torch, y_labels_torch)
print(f"\n✅ Dataset ready: {len(dataset):,} samples")

# %%
# --- Cell 6: Training Config ---

BATCH_SIZE = 32     # smaller than pure-SNN due to LMU sequential cost
EPOCHS     = 30
LR         = 5e-4
NUM_STEPS  = 25
LMU_HIDDEN = 32     # 32 × 6 channels = 192 features into SNN
LMU_ORDER  = 8      # Legendre polynomial order; try 4 or 16 to tune
N_FOLDS    = 5

lmu_snn_dir = models_dir / 'lmu_snn'
lmu_snn_dir.mkdir(exist_ok=True)

print("=" * 80)
print("FallNet LMU-SNN — 5-Fold Cross-Validation")
print("=" * 80)
print(f"Device:     {device}")
print(f"Batch:      {BATCH_SIZE} | Epochs: {EPOCHS} | SNN steps: {NUM_STEPS}")
print(f"LMU hidden: {LMU_HIDDEN}/channel | LMU order: {LMU_ORDER}")
print(f"LMU output: {LMU_HIDDEN * 6} features → SNN")

# %%
# --- Cell 7: 5-Fold Training Loop ---

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = {
    'val_acc':          [],
    'fall_init_recall': [],
    'predictions':      [],
    'targets':          [],
}

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_data_torch, y_labels_torch), 1
):
    print(f"\n{'='*80}")
    print(f"FOLD {fold}/{N_FOLDS} — train: {len(train_idx):,}  val: {len(val_idx):,}")
    print(f"{'='*80}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model     = FallNet_LMU_SNN(
        num_steps=NUM_STEPS, lmu_hidden=LMU_HIDDEN, lmu_order=LMU_ORDER
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, mem_out = model(data)

            # Combined loss: membrane average + 0.5 × spike count
            loss = (criterion(mem_out.mean(dim=0), targets)
                    + 0.5 * criterion(spk_out.sum(dim=0), targets))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # ---- Validate ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                predicted = model(data)[0].sum(dim=0).argmax(dim=1)
                total    += targets.size(0)
                correct  += (predicted == targets).sum().item()

        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       lmu_snn_dir / f'lmu_snn_fold_{fold}.pth')

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | "
                  f"Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Val: {val_acc:.4f} | Best: {best_acc:.4f}")

    # ---- Final fold eval ----
    model.load_state_dict(torch.load(
        lmu_snn_dir / f'lmu_snn_fold_{fold}.pth', weights_only=True
    ))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, targets in val_loader:
            predicted = model(data.to(device))[0].sum(dim=0).argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    fi_idx    = label_map['Fall_Initiation']
    fi_mask   = all_targets == fi_idx
    fi_recall = (all_preds[fi_mask] == fi_idx).mean() if fi_mask.any() else 0.0

    fold_results['val_acc'].append(best_acc)
    fold_results['fall_init_recall'].append(fi_recall)
    fold_results['predictions'].append(all_preds)
    fold_results['targets'].append(all_targets)

    print(f"\n  Fold {fold} → Acc: {best_acc*100:.2f}%  "
          f"Fall_Init Recall: {fi_recall*100:.2f}%")

# %%
# --- Cell 8: Results ---

mean_acc    = np.mean(fold_results['val_acc'])
std_acc     = np.std(fold_results['val_acc'])
mean_recall = np.mean(fold_results['fall_init_recall'])
std_recall  = np.std(fold_results['fall_init_recall'])

print("=" * 80)
print("5-FOLD RESULTS — FallNet LMU-SNN")
print("=" * 80)
for i, (a, r) in enumerate(zip(
    fold_results['val_acc'], fold_results['fall_init_recall']
), 1):
    print(f"  Fold {i}: Acc {a*100:.2f}%  Fall_Init Recall {r*100:.2f}%")

print(f"\nMean Accuracy:        {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"Mean Fall_Init Recall:{mean_recall*100:.2f}% ± {std_recall*100:.2f}%")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"{'Model':<25} {'Accuracy':<20} {'Fall_Init Recall'}")
print("-" * 65)
print(f"{'CNN (FP32)':<25} {'94.71%':<20} {'97.82%'}")
print(f"{'SNN (trained)':<25} {'88.83%':<20} {'see cv_results.json'}")
print(f"{'LMU-SNN':<25} "
      f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%    "
      f"{mean_recall*100:.2f}% ± {std_recall*100:.2f}%")

print("\n" + "=" * 80)
print("AGGREGATE CLASSIFICATION REPORT")
print("=" * 80)
all_p = np.concatenate(fold_results['predictions'])
all_t = np.concatenate(fold_results['targets'])
names = [reverse_label_map[i] for i in range(6)]
print(classification_report(all_t, all_p, target_names=names, digits=4))
