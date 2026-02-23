# 💓 IdxPPG - Foundational

> PPG-based heart rate estimation under sport conditions using bilateral analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Resources & Documentation](#resources--documentation)

---

## 🎯 Overview

IdxPPG is a cutting-edge solution for photoplethysmography (PPG) signal analysis, specifically optimized for high-motion sport environments. It leverages bilateral waveform analysis techniques to deliver accurate heart rate estimations even during intense physical activity.

**Key Features:**
- ✅ Robust PPG signal processing
- ✅ Bilateral analysis methodology
- ✅ Sport-condition optimization
- ✅ Advanced feature extraction

---

## 🚀 Quick Start

### Installation

#### Prerequisites
- Python 3.10 or higher
- Conda (recommended)

#### Setup Steps

```bash
# Create and activate conda environment
conda create -n idxppg python=3.10 -y
conda activate idxppg

# Install dependencies and download models
pip install -r requirements.txt
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Kzp8G0IFpLJadrguW2K_WbCMHYPChps4
```

---

## 📚 Resources & Documentation

### Core PPG Theory
| Topic | Resource |
|-------|----------|
| **PPG Fundamentals** | [Wearable PPG Chapter](https://peterhcharlton.github.io/publication/wearable_ppg_chapter/Wear_PPG_Chapter_20210323.pdf) |
| **Signal Processing** | [PPG Signal Processing Chapter](https://peterhcharlton.github.io/publication/ppg_sig_proc_chapter/PPG_sig_proc_Chapter_20210612.pdf) |

### Essential Toolkits

Extract ground truth labels such as HR, HRV, RR, and more:

- **[HeartPy](https://github.com/paulvangentcom/heartrate_analysis_python)** - Python Heart Rate Analysis Package (PPG & ECG)
- **[NeuroKit](https://github.com/neuropsychology/NeuroKit)** - Python Toolbox for Neurophysiological Signal Processing

### PPG Signal Quality & Assessment

- [Nature Article: PPG Signal Quality](https://www.nature.com/articles/s44328-024-00002-1)
- [ACM Digital Library: Signal Quality](https://dl.acm.org/doi/full/10.1145/3587256)

### Advanced Modeling & Feature Extraction

- [OpenReview: PPG Modeling](https://openreview.net/forum?id=aMAbseqdg7)
- [ACM: Feature Extraction Techniques](https://dl.acm.org/doi/abs/10.1145/3749494)
- [BP-Benchmark Repository](https://github.com/inventec-ai-center/bp-benchmark)

