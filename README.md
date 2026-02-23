IdxPPG - Foundational
=====

PPG-based heart rate estimation under sport conditions using bilateral analysis.

## Setup

```bash
conda create -n idxppg python=3.10 -y
conda activate idxppg
pip install -r requirements.txt
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Kzp8G0IFpLJadrguW2K_WbCMHYPChps4
```

## Docs


- PPG fundamentals: https://peterhcharlton.github.io/publication/wearable_ppg_chapter/Wear_PPG_Chapter_20210323.pdf

- PPG toolkits (mainly used to extract ground truth labels such as HR, HRV, RR, …):
    - [HeartPy](https://github.com/paulvangentcom/heartrate_analysis_python): Python Heart Rate Analysis Package, for both PPG and ECG signals
    - [Neurokit](https://github.com/neuropsychology/NeuroKit): The Python Toolbox for Neurophysiological Signal Processing
- PPG Signal Quality Index (SQI):
    - https://www.nature.com/articles/s44328-024-00002-1
    - https://dl.acm.org/doi/full/10.1145/3587256
    - https://peterhcharlton.github.io/publication/ppg_sig_proc_chapter/PPG_sig_proc_Chapter_20210612.pdf
- PPG modeling and feature extractions:
    - https://openreview.net/forum?id=aMAbseqdg7
    - https://dl.acm.org/doi/abs/10.1145/3749494
    - https://github.com/inventec-ai-center/bp-benchmark

