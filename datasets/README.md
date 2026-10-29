# Datasets Setup Guide

This project trains and evaluates on a combination of Western and Indian Classical music datasets to form a unified 8-instrument vocabulary:
`["vocals", "drums", "bass", "guitar", "piano", "violin", "mridangam", "ghatam"]`.

Due to their large multi-gigabyte sizes and licensing, audio datasets are not tracked in Git.

---

## Directory Structure

Organize your datasets inside this folder as follows:

```
datasets/
├── musdb18hq/                 # MUSDB18-HQ dataset
│   ├── train/                 # Training songs (.wav stems)
│   └── test/                  # Test songs (.wav stems)
│
├── saraga/                    # Saraga Indian Art Music dataset
│   ├── carnatic/              # Carnatic stems: violin, mridangam, ghatam, vocals
│   └── hindustani/            # Hindustani stems and recordings
│
├── Philharmonia/              # Philharmonia instrument samples (optional)
│
└── custom_mixes/              # Put any audio tracks (.wav, .mp3, .flac) here
                               # to test immediately with the Web UI or CLI demo
```

---

## Dataset Details & Sources

### 1. MUSDB18-HQ
- **Website**: [https://sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html)
- **Description**: High-quality uncompressed stems for 150 full-length songs (Vocals, Drums, Bass, Other).
- **Stems utilized**: Vocals, Drums, Bass, Guitar, Piano.

### 2. Saraga Carnatic / Hindustani
- **Website**: [https://mtg.github.io/saraga/](https://mtg.github.io/saraga/)
- **Description**: Indian Art Music collection with multi-track recordings and instrument-specific channels.
- **Stems utilized**: Violin, Mridangam, Ghatam, Vocal.

### 3. Custom Mixes (`datasets/custom_mixes/`)
Drop any audio files (`.wav`, `.mp3`, `.flac`, `.m4a`) here. The Web UI and `demo_cli.py` will automatically detect them as selectable sample inputs for interactive separation.
