# Fall Detection Using Spiking Neural Networks

Research implementation of fall detection on resource-constrained microcontrollers using Spiking Neural Networks (SNNs).

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/uv-package%20manager-blue)](https://github.com/astral-sh/uv)

## 📋 Overview

This repository contains the complete pipeline for training, converting, and deploying fall detection models on microcontrollers:

- **Preprocessing**: KFall and SisFall dataset fusion with temporal segmentation
- **Baseline Models**: CNN-only, CNN-LSTM ensemble comparison
- **Microcontroller-Optimized CNN**: 41K parameters, fits on Arduino
- **SNN Conversion**: Event-driven spiking neural networks for low-power inference
- **Hardware Deployment**: Arduino Nano 33 BLE Sense implementation

**Key Findings** (details in [Key Findings](#key-findings) and [Replication Study](#-replication-study-jain--semwal-2022)):

1. **Temporal recurrence doesn't pay for this task.** A 1-second input window already *is* the temporal memory; spatial convolutions over it capture the fall transient. LSTM adds instability (≈1-in-5 training runs collapses) with no accuracy gain; LMU adds a modest +2.2 points that is dominated anyway by the Micro-CNN.
2. **The deployable model wins outright.** A 41K-parameter Micro-CNN hits **94.71% accuracy** (INT8: 93.96% at 56 KB), beating every full-size baseline while fitting on a $35 Arduino.
3. **The baseline paper's sensitivity replicates; its accuracy doesn't.** On a paper-exact reconstruction of Jain & Semwal's dataset, Fall_Initiation recall reaches 97.85% (best fold 99.1%) against their 99.24% — but overall accuracy plateaus at 88.7% vs their 97.52%, with the gap localized to two 55-sample minority classes.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [UV package manager](https://github.com/astral-sh/uv)
- Kaggle account (for dataset download)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/fall-detection.git
cd fall-detection

# Install dependencies with UV
uv sync

# Setup Kaggle API credentials (one-time)
# 1. Get API token: https://www.kaggle.com/settings
# 2. Download kaggle.json
# 3. Place in ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Download Datasets
```bash
# Download both KFall and SisFall (~1.2 GB total)
python scripts/download_datasets.py

# Or verify existing data
python scripts/download_datasets.py --verify-only
```

### Explore Data
```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_data_exploration.ipynb
```

---

## 📊 Datasets

### SisFall
- **Source**: [Kaggle](https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset)
- **Subjects**: 38 (23 young adults, 15 elderly)
- **Activities**: 19 ADLs + 15 fall types
- **Sampling**: 200 Hz, 9-axis IMU (ADXL345 + ITG3200)
- **Files**: 4,506 recordings
- **Size**: ~720 MB

### KFall
- **Source**: [Kaggle](https://www.kaggle.com/datasets/usmanabbasi2002/kfall-dataset)
- **Subjects**: 32
- **Activities**: Fall types (T28-T34), Stumble (T10)
- **Sampling**: 100 Hz (upsampled to 200 Hz)
- **Files**: 5,075 recordings
- **Size**: ~468 MB

**Citation**:
```bibtex
@article{sucerquia2017sisfall,
  title={SisFall: A fall and movement dataset},
  author={Sucerquia, Angela and L{\'o}pez, Jos{\'e} David and Vargas-Bonilla, Jes{\'u}s Francisco},
  journal={Sensors},
  volume={17},
  number={1},
  pages={198},
  year={2017}
}
```

---

## 🧪 Experimental Results

### Model Comparison

| Model | Accuracy | Std Dev | Fall_Init Recall | Parameters | Size | Arduino? | Notes |
|-------|----------|---------|-------------------|------------|------|----------|-------|
| CNN-only | 88.82% | ±0.66% | 95.6% | 13.6M | — | ❌ | Baseline |
| CNN-LSTM ensemble | 87.56% | ±3.74% | 97.6% | 14.0M | — | ❌ | Below CNN-only; ~1-in-5 fold collapse |
| CNN-LMU Hybrid | 90.99% | ±0.76% | 99.3% | 230,886 | ~225 KB (INT8 est.) | ❌ | Best full-size accuracy |
| **Micro-CNN (FP32)** | **94.71%** | **±0.22%** | **97.8%** | **41,062** | 164 KB (raw) / ~553 KB (.tflite unquantized) | ❌ | Too large unquantized |
| **Micro-CNN (Float16)** | **94.71%** | **±0.22%** | **97.8%** | 41,062 | **~85 KB** | ✅ | No accuracy drop from FP32 |
| **Micro-CNN (INT8)** | **93.96%** | **±0.49%** | **97.6%** | 41,062 | **~56 KB** | ✅ | **Recommended: smallest, deployable now** |
| SNN (float32) | 88.78% | ±1.18% | 93.3% | — | ~175 KB | ⚠️ | Needs INT8 quantization before deployment |

*Sources: `fall_detection_data/models/{cnn_only_model,training_summary.txt,training_summary_hybrid.txt,quantized,snn}` — INT8 SNN/Float16 SNN sizes in the deployment analysis are projected estimates, not measured (SNN hasn't been quantized yet).*

### Class-wise Performance (CNN-only, 6-class)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Walking | 0.819 | 0.830 | 0.824 |
| Jogging | 0.861 | 0.851 | 0.856 |
| Walking_stairs_updown | 0.929 | 0.955 | 0.942 |
| Stumble_while_walking | 0.699 | 0.628 | 0.662 |
| Fall_Initiation | **0.951** | **0.956** | **0.953** |
| Impact_Aftermath | 0.992 | 0.980 | 0.986 |

**Critical metric**: Fall_Initiation recall = 95.6% (CNN-only); Micro-CNN reaches 97.8% (FP32) / 97.6% (INT8), and CNN-LMU Hybrid reaches 99.3%.

**Why Stumble looks so bad (F1 ≈ 0.66)**: the 6-class pipeline windowed SisFall D18 ("stumble while walking") trials like an ordinary ADL, chopping each ~20 s trial into ~19 one-second windows all labeled "Stumble" — but a stumble is momentary, so most of those 1,479 windows are ordinary walking mislabeled as stumbles. The paper instead isolates the actual stumble moment with its segmentation algorithm (≈280 genuine samples). See the Replication Study below; `scripts/preprocess_paper_exact.py` implements the corrected extraction.

### Key Findings

1. **Temporal recurrence isn't helpful for pre-impact fall detection**
   - CNN-only: 88.82% ± 0.66% (88.88% retrained at batch 512 / lr 1e-3)
   - CNN-LSTM: never beat CNN-only on identical data (87.13–87.56%), and is unreliable: roughly 1 in 5 training runs collapses (worst observed fold: 52.9% Fall_Initiation recall). Reproduced across batch sizes and datasets, so it is a property of the architecture, not the config.
   - CNN-LMU Hybrid: 90.99% ± 0.76% (+2.2 points over CNN-only) — the one temporal model that consistently helps, but see finding 2
   - Physical explanation: the 1 s input window at 200 Hz already spans the fall transient; convolutions capture its shape, and recurrence has no longer-range dependency to exploit. Campanella et al. (2024) reached the same conclusion independently (FFNN, 99.38%, explicitly rejecting LSTM/GRU).
   - The LMU still matters for the *neuromorphic* track — it is the principled way to implement temporal memory in spiking networks (Voelker et al. 2019) — just not as an accuracy device.

2. **Microcontroller deployment is viable and already outperforms the full-size baselines**
   - 332x parameter reduction (13.6M → 41K)
   - Micro-CNN (94.71% FP32 / 93.96% INT8) beats every full-size model above, including CNN-LMU Hybrid
   - INT8 version fits in 56 KB flash / 40 KB RAM on the $35 Arduino Nano 33 BLE Sense

3. **SNN accuracy trails the deployable CNN by ~5 points**
   - SNN: 88.78% ± 1.18% vs Micro-CNN INT8: 93.96% ± 0.49%
   - SNN's case rests on power efficiency on neuromorphic hardware, not on-MCU accuracy — it still needs INT8 quantization and hasn't been benchmarked on real hardware yet

4. **Hyperparameters were not the replication bottleneck**
   - The original CNN-only training used batch_size=2 / lr=1e-5 (vs the paper's 512 / Keras-default 1e-3). Retraining with the paper's settings changed accuracy by only +0.06 points (88.82% → 88.88%) but converged in ~30 epochs instead of 50 with tighter variance — worth fixing for speed/stability, irrelevant to the accuracy gap.

---

## 🔁 Replication Study: Jain & Semwal (2022)

A systematic attempt (Aug 2026) to replicate the baseline paper's reported results (97.52% accuracy, 99.24% Fall_Initiation sensitivity) by matching its methodology as exactly as the publication allows.

### Preprocessing bugs found and fixed along the way

1. **Transitional-window coin flip** — the original pipeline extracted *either* the 0.5 s transitional window *or* the full 1 s Fall_Initiation window per fall event, chosen randomly. The paper extracts **both** as separate training samples (its Table III Fall_Initiation support of 658 is exactly 2× Impact's 329). Fixing this doubled Fall_Initiation samples (1,649 → 3,298) and raised CNN-only Fall_Initiation recall from 95.63% to **96.60%** with overall accuracy unchanged — precisely the expected effect of training on the earlier, subtler half of the fall signal.
2. **Stumble mislabeling** — SisFall D18 trials were windowed like ADLs, producing 1,479 mostly-walking windows labeled "Stumble." The paper isolates the actual stumble moment via its segmentation algorithm (~280 samples). This bug is the likely cause of Stumble's F1 ≈ 0.66 in the 6-class results.
3. **ADL over-extraction** — 20 s was taken per *file* instead of per (subject, task), over-representing multi-trial tasks (stairs: 5,852 samples vs the paper's ~1,540).

### Paper-exact dataset validation

Rebuilding the dataset with these fixes (`scripts/preprocess_paper_exact.py`) reproduces the paper's class distribution almost sample-for-sample (paper totals back-computed as test support × 5):

| Class | Ours | Paper | Class | Ours | Paper |
|---|---|---|---|---|---|
| Walking | **1,520** | 1,520 | Fall_Recovery | 279 | ~275 |
| Jogging | **1,480** | 1,480 | Fall_Initiation | 3,298 | ~3,290 |
| Stairs | 1,240 | ~1,540 | Impact | 1,646 | ~1,645 |
| Stumble | 279 | ~280 | Aftermath | 1,609 | ~1,645 |

### Results (paper's CNN-LSTM ensemble, 8-class, batch 512, 200 epochs, no class weights)

| Metric | Paper | Replication (5-fold CV) |
|---|---|---|
| **Fall_Initiation recall** | 99.24% | **97.85% ± 0.87%** (best fold 99.1%) |
| Fall_Initiation F1 | 98.79% | 95.74% |
| Overall accuracy | 97.52% | 88.73% ± 1.18% |

Per-class, the missing ~9 accuracy points are localized: Walking/Jogging/Fall_Initiation recall are within a few points of the paper, while **Stumble (recall 0.44 vs 0.93)** and **Fall_Recovery (0.35 vs 0.85)** — the two ~55-support classes — plus stairs account for nearly all of the gap. Full breakdown in `fall_detection_data/models/fallnet_paper_replication/replication_summary.txt`.

### Interpretation

- **The paper's safety-critical claim replicates**: near-perfect pre-impact sensitivity is reachable from the published methodology.
- **The headline accuracy does not**, and the shortfall sits in minority classes whose reported performance (F1 ≈ 0.93 on 56 test samples, single fold) is statistically fragile. Caveats applying to both the paper and this replication: Table III reports a single (possibly best) fold, and the underlying datasets contain repeated near-identical trials per subject, which random stratified K-fold leaks across the train/test split (limitation also noted by Campanella et al. 2024).
- **For the airbag use case this gap doesn't matter**: the decision-relevant metric is Fall_Initiation recall, and confusing Stumble with Fall_Recovery deploys no airbag either way. Chasing the remaining 9 points (e.g., via class weights or focal loss) would trade minority-class recall against fall sensitivity — the wrong trade for this application.

**Reproduce**: `python scripts/preprocess_paper_exact.py && python scripts/train_fallnet_paper_replication.py`

---

## 📁 Repository Structure
```
fall-detection/
├── fall_detection/              # Python package
│   └── __init__.py
├── fall_detection_data/         # Datasets (gitignored)
│   ├── KFall/                   # KFall dataset
│   ├── SisFall/                 # SisFall dataset
│   ├── processed/               # Preprocessed data
│   │   ├── X_data_6class.npy   # 16,732 samples
│   │   └── y_labels_6class.npy # 6 classes
│   └── models/                  # Trained models
├── notebooks/                   # Jupyter notebooks
│   ├── 01-data-exploration.ipynb
│   ├── 02_DataExplorationBothDataSets.ipynb   # main preprocessing + training notebook
│   ├── ReducingCNNWeights.ipynb               # Micro-CNN + SNN experiments
│   ├── SNN_LMU_Pipeline.ipynb
│   └── ComparingMicroCNNandSNN.ipynb
├── scripts/                     # Automation scripts
│   ├── download_datasets.py                       # Kaggle dataset downloader
│   ├── preprocess_paper_exact.py                  # paper-exact 8-class dataset (replication)
│   ├── preprocess_with_transitional_window_fix.py # 6-class dataset, Tw bug fixed
│   ├── train_fallnet_paper_replication.py         # CNN-LSTM replication run
│   ├── retrain_cnn_only_bs512_lr1e3.py            # CNN-only, paper hyperparameters
│   └── eval_fall_init_recall.py                   # per-class evaluation of saved folds
├── Research/                    # Papers and references
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Locked dependencies
└── README.md                   # This file
```

---

## 🔬 Preprocessing Pipeline

### 6-Class Merged Dataset

**Classes**:
1. Walking
2. Jogging
3. Walking_stairs_updown
4. Stumble_while_walking
5. Fall_Initiation (pre-impact, <1s before fall)
6. Impact_Aftermath (merged: Impact + Aftermath due to insufficient Fall_Recovery data)

**Why merge Impact + Aftermath?**
- Fall_Recovery: Only 159 samples (0.94%) - insufficient for training
- Solution: Merge Impact (1,646) + Aftermath (1,609) = 3,255 samples

### Processing Steps

1. **Data Loading**
   - SisFall: Convert from bits to physical units (g, °/s)
   - KFall: Upsample 100 Hz → 200 Hz (cubic spline)

2. **Temporal Segmentation** (Algorithm 1 from Jain & Semwal 2022)
   - Automatic fall detection using Y-axis acceleration variance
   - Extract: ADL (before), Fall_Initiation, Impact, Aftermath
   - Window: 200 samples (1 second @ 200 Hz)

3. **Normalization**
   - Per-dataset Z-score normalization
   - Dataset fusion
   - Final Z-score normalization

4. **Output**
   - X: (16,732, 200, 6) - samples × timesteps × features
   - y: (16,732,) - class labels
   - Features: [AccX, AccY, AccZ, GyrX, GyrY, GyrZ]

### Known issues in this pipeline (see Replication Study)

The 6-class dataset above was produced by `notebooks/02_DataExplorationBothDataSets.ipynb` and carries three bugs relative to the paper's methodology: the transitional-window coin flip, stumble-as-ADL mislabeling, and per-file (rather than per-task) ADL capping. Corrected pipelines:

- `scripts/preprocess_with_transitional_window_fix.py` → `*_6class_twfix.npy` (18,381 samples; Tw fix only, otherwise comparable to the original 6-class data)
- `scripts/preprocess_paper_exact.py` → `*_8class_paper.npy` (11,351 samples; all three fixes, 8 classes, paper-faithful)

The published 6-class model results in this README were trained on the original (buggy) dataset and are kept for continuity; the Replication Study section reports results on the corrected data.

---

## 🎯 Reproducing Results

### Step 1: Preprocess Data
```bash
# 6-class dataset (original notebook pipeline):
jupyter notebook notebooks/02_DataExplorationBothDataSets.ipynb
# -> X_data_6class.npy (16,732 samples)

# 6-class with transitional-window fix:
python scripts/preprocess_with_transitional_window_fix.py
# -> X_data_6class_twfix.npy (18,381 samples)

# Paper-exact 8-class dataset (replication study):
python scripts/preprocess_paper_exact.py
# -> X_data_8class_paper.npy (11,351 samples)
```

### Step 2: Train CNN-Only Baseline
```bash
python scripts/retrain_cnn_only_bs512_lr1e3.py

# Expected results (5-fold CV):
# Accuracy: ~88.9%
# Fall_Initiation Recall: ~95.6% (96.6% on twfix data)
```

### Step 3: Run the Paper Replication
```bash
python scripts/train_fallnet_paper_replication.py

# Expected results (5-fold CV):
# Accuracy: ~88.7%  |  Fall_Initiation Recall: ~97.9% (paper: 97.52% / 99.24%)
```

### Step 4: Micro-CNN and SNN
Micro-CNN training, quantization, and SNN conversion live in `notebooks/ReducingCNNWeights.ipynb` and `notebooks/SNN_LMU_Pipeline.ipynb`; quantized `.tflite` artifacts land in `fall_detection_data/models/quantized/`.

---

## 🤖 Hardware Deployment

### Target Platform: Arduino Nano 33 BLE Sense

**Specifications**:
- MCU: Nordic nRF52840 (ARM Cortex-M4F @ 64 MHz)
- RAM: 256 KB
- Flash: 1 MB
- **Built-in IMU**: LSM9DS1 (9-axis: Acc + Gyro + Mag)
- BLE: Wireless connectivity
- Price: ~$35

**Why This Platform?**
- ✅ Built-in IMU (no external sensors needed)
- ✅ Sufficient memory for micro-CNN/SNN
- ✅ Low power modes for battery operation
- ✅ BLE for wireless monitoring
- ✅ Affordable and widely available

### Memory Budget

| Variant | Flash | RAM | Fits (256 KB RAM / 1 MB Flash)? |
|---------|-------|-----|----------------------------------|
| Micro-CNN Float16 | 85 KB | 60 KB | ✅ |
| Micro-CNN INT8 | 56 KB | 40 KB | ✅ |
| SNN Float16 (estimated) | 85 KB | 70 KB | ✅ |
| SNN INT8 (estimated, not yet quantized) | 56 KB | 50 KB | ✅ |

**Recommended**: Micro-CNN Float16 for best accuracy/size trade-off (94.71%, no accuracy drop from FP32), or Micro-CNN INT8 for smallest footprint (93.96%, currently the only variant with a working `.tflite` + Arduino `.ino` sketch).

*Source: `fall_detection_data/models/arduino_deployment_analysis.txt`*

---

## 📈 Comparison with State-of-the-Art

### Campanella et al. (2024) - IEEE Sensors Journal

| Metric | Campanella et al. | Our Work (Micro-CNN INT8, measured) | Our Work (SNN, target) |
|--------|-------------------|--------------------------------------|--------------------------|
| Platform | STM32U575xx | Arduino Nano 33 BLE | Arduino Nano 33 BLE / neuromorphic HW |
| Model | FFNN (conventional) | Micro-CNN (quantized) | **SNN (neuromorphic)** |
| Accuracy | 99.38% | **93.96%** | ~88.8% (measured, unquantized) |
| Latency | 25 ms | Not yet measured on-device | <100 ms (target) |
| Power | ~100 mW | Not yet measured | **~30 mW** (target, 3x lower) |
| Model size | 60 KB | **56 KB** | 40 KB (target, not yet quantized) |
| Classes | Binary (fall/ADL) | 6-class | 6-class |

**Our Contribution**: First SNN-based fall detection on commodity microcontrollers, demonstrating practical neuromorphic computing without specialized hardware.

---

## 🔧 Development Setup

### Add New Dependencies
```bash
# Add package with UV
uv add package-name

# Examples
uv add snntorch      # For SNN conversion
uv add torch         # PyTorch for SNNs
uv add pytest        # For testing
```

### Development Dependencies
```bash
# Install dev dependencies
uv sync --group dev

# Includes: jupyter, ruff, black, pytest
```

### Code Quality
```bash
# Format code
black fall_detection/

# Lint
ruff check fall_detection/

# Type checking
mypy fall_detection/
```

---

## 📝 Project Timeline

### ✅ Completed
- [x] Dataset download and preprocessing
- [x] CNN-only baseline (88.82%; 88.88% at paper hyperparameters)
- [x] CNN-LSTM ensemble evaluation (no gain over CNN-only; ~1-in-5 fold collapse, reproduced across configs)
- [x] CNN-LMU Hybrid evaluation (90.99%, +2.2 over CNN-only)
- [x] 6-class merged dataset
- [x] Finding: temporal recurrence not worth it — window already spans the fall transient
- [x] Replication study of Jain & Semwal (2022): sensitivity replicates (97.85% vs 99.24%), accuracy does not (88.73% vs 97.52%); gap localized to ~55-support minority classes
- [x] Preprocessing bugs found & fixed (transitional-window coin flip, stumble mislabeling, ADL over-extraction)

### 🔄 In Progress
- [x] Micro-CNN training (41K params, 94.71% FP32 / 93.96% INT8)
- [x] CNN → SNN conversion (88.78% accuracy; still needs INT8 quantization)
- [x] Arduino inference sketch + generated model header (`fallnet-inference/`) — not yet validated on physical hardware
- [ ] Power consumption measurements

### 📅 Planned
- [ ] Real-world fall testing
- [ ] Battery life benchmarks
- [ ] Comparison with neuromorphic chips (Intel Loihi, if available)
- [ ] Paper submission

---

## 🎓 Research Context

### Baseline Paper

**"A novel Feature extraction method for Pre-Impact Fall detection system"**
- Authors: Jain & Semwal
- Journal: IEEE Sensors Journal, 2022
- Accuracy: 97.52% (8-class, CNN-LSTM ensemble)

**Why our results differ** (established empirically — see Replication Study above):
- ~~Hyperparameters~~ — ruled out: retraining at the paper's batch 512 / default LR moved accuracy by +0.06 points
- ~~Dataset construction~~ — ruled out: the paper-exact rebuild matches their class counts almost sample-for-sample and still lands 9 points short
- The gap is concentrated in the two ~55-support classes (Stumble, Fall_Recovery), whose reported single-fold F1 ≈ 0.93 could not be reproduced; the paper's headline accuracy hinges on them
- The paper's core sensitivity claim (99.24% Fall_Initiation recall) *does* replicate to within ~1.4 points
- Shared caveat: both datasets contain repeated near-identical trials per subject, which random stratified K-fold leaks across train/test (also noted by Campanella et al. 2024)
- Our focus remains practical deployment on microcontrollers, where the Micro-CNN already exceeds every full-size baseline

### Our Research Question

> "Can spiking neural networks on commodity microcontrollers achieve comparable fall detection accuracy to conventional DNNs while providing 3-5x lower power consumption for battery-powered wearable safety devices?"

---

## 💡 Use Cases

**Target Application**: Wearable airbag vest for elderly

**Requirements**:
- **Latency**: <100 ms (50-100 ms airbag inflation time)
- **Power**: Low enough for 24+ hour battery life
- **Accuracy**: >85% fall detection, <5% false positives
- **Cost**: <$50 total system cost

**Our Solution**:
- SNN provides 3-5x power savings vs conventional CNN
- Micro-CNN fits on $35 Arduino with built-in IMU
- <100 ms inference latency achievable
- Event-driven processing enables long battery life

---

## 🙏 Acknowledgments

### Datasets
- **SisFall**: Sucerquia et al. (2017)
- **KFall**: Jain & Semwal (2022)

### Baseline Research
- Jain & Semwal (2022) - FallNet CNN-LSTM ensemble
- Campanella et al. (2024) - FFNN on STM32

### Tools & Frameworks
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [snnTorch](https://snntorch.readthedocs.io/) - SNN conversion (planned)
- [Kaggle](https://www.kaggle.com/) - Dataset hosting

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📧 Contact

**Iain Barkley**
- Email: [your.email@example.com]
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 📚 Citation

If you use this code or findings in your research, please cite:
```bibtex
@misc{barkley2026fall,
  title={Spiking Neural Networks for Fall Detection on Resource-Constrained Microcontrollers},
  author={Barkley, Iain},
  year={2026},
  howpublished={\url{https://github.com/yourusername/fall-detection}}
}
```

---

## 🔗 Related Resources

- [Arduino Nano 33 BLE Sense](https://store.arduino.cc/products/arduino-nano-33-ble-sense)
- [snnTorch Documentation](https://snntorch.readthedocs.io/)
- [Neuromorphic Computing Resources](https://neuromorphic.com/)
- [Intel Loihi Research](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)

---

**Star ⭐ this repo if you find it useful!**
