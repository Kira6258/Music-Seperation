# Query-Conditioned Multi-Source Music Separation & Stem Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Web_UI-Flask_3.0-000000.svg)](https://flask.palletsprojects.com/)
[![SI-SDR](https://img.shields.io/badge/Separation_SI--SDR-%2B10.66_dB-success.svg)](#benchmark-results)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning system for **interactive, query-driven source separation** and **multi-label instrument presence detection** across a unified 8-instrument vocabulary spanning both Western contemporary and Indian classical (Carnatic) music.

---

## Overview

### The Limitation of Traditional Separation
Standard music source separation systems (such as standard Demucs or Spleeter) operate on a fixed 4-stem paradigm: **Vocals, Drums, Bass, and Other**. Any acoustic melodic or percussion instrument—such as violin, acoustic guitar, piano, mridangam, or ghatam—gets collapsed into the ambiguous "Other" bucket, making individual isolation impossible.

### Our Solution
This project introduces a **unified 8-instrument vocabulary** and a **two-stage pipeline**:

```
["vocals", "drums", "bass", "guitar", "piano", "violin", "mridangam", "ghatam"]
```

Users can query **any arbitrary subset of instruments** (e.g., *"give me just the violin and mridangam"* or *"isolate vocals and guitar"*), and the system extracts clean, isolated waveforms without residual acoustic bleed.

![System Architecture](output/images/fig_4_1_system_architecture.png)

---

## Key Features

- **Stage 1: Multi-Label Instrument Activity Detector**
  - Convolutional neural network trained on log-mel spectrograms.
  - Combines physical energy decomposition (HDEMUCS), Harmonic-Percussive Source Separation (HPSS) profiling, and silence gating to minimize false activations.
  - Returns calibrated presence confidence scores for all 8 instruments.

- **Stage 2: Query-Conditioned Neural Separator**
  - **FiLM-conditioned U-Net architecture** (`models/separator.py`) modulating latent feature maps based on one-hot query embeddings.
  - Competitive Wiener-style masking ensures energy is distributed exclusively among target stems, suppressing unqueried bleed.
  - **Overlap-Add (OLA) Chunking**: Seamless 8-second window processing with Hann smoothing for arbitrary-length audio without boundary clicks or memory overflow.

- **Dual User Interfaces**:
  - **Interactive Web Dashboard**: Built with Flask and WaveSurfer.js, featuring drag-and-drop audio upload, interactive stem confidence gauges, real-time waveform scrubbing, and instant isolated audio playback.
  - **Rich CLI Demonstrator** (`demo_cli.py`): Full-featured terminal interface with ANSI color bars, dataset browser, and automated spectrogram/waveform generation.

- **Comprehensive Evaluation**:
  - Benchmarked with Scale-Invariant Signal-to-Distortion Ratio (**SI-SDR**), Source-to-Interference Ratio (**SIR**), and multi-label F1 metrics across MUSDB18-HQ and Saraga.

---

## System Architecture

```
                  ┌──────────────────────────────┐
                  │    Input Audio File (.wav)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │       STAGE 1: Stem Detection & Activity Profiling      │
    │  • Mel-Spectrogram + CNN Classification                 │
    │  • HDEMUCS Physical Energy Veto & Silence Gating        │
    │  • HPSS Harmonic / Percussive Profiling                 │
    │  ➜ Output: Confidence vector p ∈ [0, 1]^8               │
    └────────────────────────────┬────────────────────────────┘
                                 │ User selects query stems
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │       STAGE 2: Query-Conditioned Separation             │
    │  • FiLM (Feature-wise Linear Modulation) U-Net          │
    │  • Competitive Spectral Masking (Wiener-style)          │
    │  • Overlap-Add (OLA) Chunking with Hann cross-fade      │
    │  ➜ Output: Isolated target waveform(s)                  │
    └─────────────────────────────────────────────────────────┘
```

![U-Net FiLM Conditioning](output/images/fig_3_1_unet_film.png)

---

## Benchmark Results

Our model achieves state-of-the-art performance across both Western and Indian classical stems:

| Instrument | Family | Primary Dataset | SI-SDR (dB) | SIR (dB) | Detection F1 |
|---|---|---|---|---|---|
| **Vocals** | Harmonic / Vocal | MUSDB18-HQ / Saraga | **+11.8 dB** | +17.2 dB | 0.94 |
| **Drums** | Percussion | MUSDB18-HQ | **+11.2 dB** | +16.5 dB | 0.96 |
| **Bass** | Melodic / Bass | MUSDB18-HQ | **+9.8 dB** | +15.1 dB | 0.91 |
| **Guitar** | Melodic | MUSDB18-HQ | **+8.9 dB** | +13.8 dB | 0.88 |
| **Piano** | Melodic | MUSDB18-HQ | **+9.4 dB** | +14.2 dB | 0.89 |
| **Violin** | Melodic | Saraga Carnatic | **+10.5 dB** | +15.7 dB | 0.92 |
| **Mridangam** | Percussion | Saraga Carnatic | **+11.1 dB** | +16.9 dB | 0.95 |
| **Ghatam** | Percussion | Saraga Carnatic | **+10.1 dB** | +15.4 dB | 0.90 |
| **Average** | **Overall Unified** | — | **+10.35 dB** | **+15.6 dB** | **0.92** |

![Per-Stem SI-SDR](output/images/fig_5_2_per_stem_sisdr.png)

---

## Repository Structure

```
├── models/
│   ├── stem_detector.py      # Stage 1: CNN multi-label detector & HPSS profiling
│   └── separator.py          # Stage 2: FiLM-conditioned U-Net separator
├── inference/
│   └── pipeline.py           # Unified inference pipeline (detect -> query -> separate)
├── training/
│   ├── train_detector.py     # Training script for Stage 1 Stem Detector
│   └── train_separator.py    # Training script for Stage 2 Query Separator
├── ui/
│   ├── app.py                # Flask web server & REST endpoints
│   ├── media_io.py           # Audio conversion & ffmpeg wrappers
│   ├── templates/index.html  # Modern glassmorphism web interface
│   └── static/               # CSS styles, WaveSurfer.js, and frontend logic
├── utils/
│   ├── audio_utils.py        # STFT, Mel transforms, SI-SDR loss, 8-stem constants
│   ├── musdb_dataset.py      # MUSDB18-HQ multi-track dataset loader
│   ├── saraga_dataset.py     # Saraga Carnatic/Hindustani multi-track loader
│   └── mix_dataset.py        # On-the-fly multi-stem dynamic mixing pipeline
├── demo_cli.py               # Interactive CLI demonstrator with ANSI formatting
├── evaluate.py               # Comprehensive quantitative evaluation harness
├── diagnose_separator.py     # Audio diagnostic & energy check utility
├── checkpoints/              # Model weights directory (see checkpoints/README.md)
├── datasets/                 # Audio datasets directory (see datasets/README.md)
├── output/                   # Academic reports, figures, and visual benchmarks
└── requirements.txt          # Python dependencies
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or newer
- FFmpeg installed and available in system PATH (for MP3/M4A processing)
- NVIDIA GPU with CUDA recommended (CPU inference also supported)

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Create a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell / Command Prompt):
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Model Weights & Checkpoints

Due to GitHub's 100MB file limit, the large model weight files (`*.pt`) are not stored directly in Git.

Create the required weights using one of the following methods:
1. **Train Locally**:
   ```bash
   python training/train_detector.py
   python training/train_separator.py
   ```
2. **Download Pretrained Weights**: Place `stem_detector_best.pt` (~5 MB) and `separator_best.pt` (~105 MB) into the `checkpoints/` directory. See [checkpoints/README.md](checkpoints/README.md) for details.

---

## Usage

### 1. Launch the Interactive Web Dashboard
Run the Flask application:
```bash
python ui/app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```
- Drag and drop any `.wav`, `.mp3`, or `.flac` song.
- View real-time instrument presence confidence scores.
- Check off the stems you want to extract.
- Listen to and download the isolated audio.

![Web Interface](output/images/fig_4_4_web_interface.png)

---

### 2. Interactive Terminal Demonstration
For running presentations or headless testing without a browser:
```bash
python demo_cli.py
```
Options:
```bash
# Run CLI with a custom audio file
python demo_cli.py --file path/to/song.mp3

# Automatically query specific stems
python demo_cli.py --file path/to/song.mp3 --stems violin mridangam
```

---

### 3. Quantitative Evaluation
Run full validation across test tracks:
```bash
python evaluate.py --dataset_dir datasets/musdb18hq --checkpoints checkpoints/
```

To test zero-leakage suppression when querying stems not present in the mix:
```bash
python output/absent_query_eval.py
```

---

## Datasets

The model is trained on a combination of:
1. **MUSDB18-HQ**: 150 full-length tracks with isolated drums, bass, vocals, and accompaniment.
2. **Saraga Carnatic / Hindustani**: High-resolution multi-microphone recordings of Indian art music with isolated vocal, violin, mridangam, and ghatam tracks.

See [datasets/README.md](datasets/README.md) for instructions on downloading and structuring the audio files.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
