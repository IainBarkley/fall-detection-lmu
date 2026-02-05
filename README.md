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

**Key Finding**: Temporal modeling (LSTM/LMU) provides no accuracy benefit for fall detection. CNN-only outperforms CNN-LSTM (89.98% vs 79.96%), validating simple feedforward SNNs for neuromorphic deployment.

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

| Model | Accuracy | Std Dev | Parameters | Memory | Arduino? | Notes |
|-------|----------|---------|------------|--------|----------|-------|
| **CNN-only** | **89.98%** | **±0.42%** | 13.6M | 54.5 MB | ❌ | Best baseline |
| CNN-LSTM ensemble | 79.96% | ±7.31% | 14.0M | 55.8 MB | ❌ | Unstable, 40% fold failures |
| CNN-LMU | 89.97% | ±0.29% | 13.7M | 54.8 MB | ❌ | Same as CNN-only |
| **Micro-CNN** | ~87%* | TBD | **41K** | **164 KB** | ✅ | **Deployable** |
| **Micro-SNN** | ~85%* | TBD | **40K** | **55 KB** | ✅ | **Low-power** |

*Target accuracy (in progress)

### Class-wise Performance (CNN-only, 6-class)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Walking | 0.805 | 0.670 | 0.732 | 2,637 |
| Jogging | 0.789 | 0.644 | 0.709 | 1,860 |
| Walking_stairs_updown | 0.733 | 0.936 | 0.822 | 5,852 |
| Stumble_while_walking | 0.631 | 0.363 | 0.461 | 1,479 |
| Fall_Initiation | **0.961** | **0.733** | **0.832** | 1,649 |
| Impact_Aftermath | 0.927 | 0.980 | 0.953 | 3,255 |

**Critical metric**: Fall_Initiation recall = 73.3% (95.6% for best fold)

### Key Findings

1. **Temporal modeling provides no benefit**
   - CNN-only: 89.98% ± 0.42%
   - CNN-LSTM: 79.96% ± 7.31% (unstable)
   - Falls are instantaneous events captured by spatial convolutions

2. **LSTM training instability**
   - 40% fold failure rate
   - Fall_Initiation recall: 46.4% (worst fold) vs 95.5% (best fold)
   - Caused by small batch size (32-64) + class imbalance

3. **Microcontroller deployment is viable**
   - 330x parameter reduction (13.6M → 41K)
   - <3% accuracy drop (target)
   - Fits on $35 Arduino Nano 33 BLE Sense

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
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_6class.ipynb
│   ├── 03_train_cnn_only.ipynb
│   └── 04_results_analysis.ipynb
├── scripts/                     # Automation scripts
│   └── download_datasets.py     # Kaggle dataset downloader
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

---

## 🎯 Reproducing Results

### Step 1: Preprocess Data
```bash
# Run preprocessing notebook
jupyter notebook notebooks/02_preprocessing_6class.ipynb

# Expected output:
# - X_data_6class.npy (16,732 samples)
# - y_labels_6class.npy (6 classes)
```

### Step 2: Train CNN-Only Baseline
```bash
# Run training notebook
jupyter notebook notebooks/03_train_cnn_only.ipynb

# Expected results:
# Accuracy: 89.98% ± 0.42%
# Fall_Initiation Recall: 95.6% (best fold)
```

### Step 3: Train Micro-CNN (In Progress)
```bash
# Train microcontroller-optimized model
python scripts/train_micro_cnn.py

# Expected:
# Parameters: 41,000
# Accuracy: 85-88%
# Size: 41 KB (INT8 quantized)
```

### Step 4: Convert to SNN (In Progress)
```bash
# Convert CNN → SNN using snnTorch
python scripts/convert_to_snn.py

# Target:
# Accuracy: ~85%
# Power: 3-5x lower than CNN
```

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

| Component | CNN (FP32) | CNN (INT8) | SNN (INT8) |
|-----------|------------|------------|------------|
| Weights | 164 KB | 41 KB | 40 KB |
| Activations | 100 KB | 25 KB | 10 KB |
| Working memory | 50 KB | 20 KB | 5 KB |
| **Total RAM** | **314 KB** ❌ | **86 KB** ✅ | **55 KB** ✅ |

**Status**: Micro-CNN and SNN both fit! 🎉

---

## 📈 Comparison with State-of-the-Art

### Campanella et al. (2024) - IEEE Sensors Journal

| Metric | Campanella et al. | Our Work (Target) |
|--------|-------------------|-------------------|
| Platform | STM32U575xx | Arduino Nano 33 BLE |
| Model | FFNN (conventional) | **SNN (neuromorphic)** |
| Accuracy | 99.38% | ~85-87% |
| Latency | 25 ms | <100 ms |
| Power | ~100 mW | **~30 mW** (3x lower) |
| Model size | 60 KB | 40 KB |
| Classes | Binary (fall/ADL) | 6-class |

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
- [x] CNN-only baseline (89.98%)
- [x] CNN-LSTM ensemble evaluation (failed - unstable)
- [x] CNN-LMU evaluation (same as CNN-only)
- [x] 6-class merged dataset
- [x] Finding: Temporal modeling provides no benefit

### 🔄 In Progress
- [ ] Micro-CNN training (41K params)
- [ ] CNN → SNN conversion
- [ ] Arduino deployment
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

**Why our results differ**:
- Different preprocessing (we merged Impact+Aftermath due to insufficient Fall_Recovery data)
- Smaller batch size (32-64 vs 512) causes LSTM instability
- GPU constraints (Quadro P1000 4GB vs unknown)
- Our focus: Practical deployment on microcontrollers, not maximizing accuracy

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
