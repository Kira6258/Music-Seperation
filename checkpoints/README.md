# Model Checkpoints

This directory is where trained weights and checkpoints are loaded by the inference pipeline (`inference/pipeline.py`), web interface (`ui/app.py`), and CLI demo (`demo_cli.py`).

> **Note on Model Weights & GitHub**: Model weight files (`*.pt`) exceed GitHub's 100MB file limit (`separator_best.pt` is ~105 MB) and are excluded from Git tracking via `.gitignore`.

---

## Required Checkpoint Files

Place the following checkpoint files in this directory:

| Filename | Size | Description |
|---|---|---|
| `stem_detector_best.pt` | ~5 MB | Stage 1 Multi-label CNN classifier predicting instrument activity |
| `separator_best.pt` | ~105 MB | Stage 2 FiLM-conditioned U-Net separator for 8-instrument extraction |

Optional / Training Checkpoints:
- `separator_latest.pt`
- `stem_detector_latest.pt`
- `separator_v2_best.pt`

---

## How to Obtain Checkpoints

### Option A: Train Your Own Models
You can train the detector and separator using the included training scripts:

```bash
# 1. Train the Stage 1 Stem Detector
python training/train_detector.py

# 2. Train the Stage 2 Query-Conditioned Separator
python training/train_separator.py
```
Checkpoints will automatically be saved to `checkpoints/stem_detector_best.pt` and `checkpoints/separator_best.pt`.

### Option B: Download Pre-Trained Weights
If you have access to the pre-trained weights release (via GitHub Releases, HuggingFace, or Google Drive link provided by the authors):
1. Download `stem_detector_best.pt` and `separator_best.pt`.
2. Move them into this directory:
   ```
   checkpoints/
   ├── stem_detector_best.pt
   └── separator_best.pt
   ```
