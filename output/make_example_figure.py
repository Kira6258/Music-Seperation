"""
make_example_figure.py
----------------------
Runs the real inference pipeline on one MUSDB18-HQ test excerpt and one Saraga
validation excerpt and draws log-magnitude spectrograms of

    mixture | our output | ground-truth stem

Output: output/images/fig_5_6_spectrogram_examples.png
Run:    python output/make_example_figure.py
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from inference.pipeline import MusicSeparationPipeline
from utils.audio_utils import load_audio, si_sdr, STEM_LABELS
from utils.saraga_dataset import SaragaCarnaticdataset
from training.train_separator import SARAGA_ROOT

OUT = ROOT / "output" / "images" / "fig_5_6_spectrogram_examples.png"
SR = 22050
N_FFT, HOP = 1024, 256
WIN = torch.hann_window(N_FFT)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})


def spec_db(x: torch.Tensor) -> np.ndarray:
    x = x.reshape(-1)
    s = torch.stft(x, N_FFT, HOP, window=WIN, return_complex=True).abs()
    return 20 * torch.log10(s + 1e-5).numpy()


def to22(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def run(pipe, mix22: torch.Tensor, stems_sel):
    tmp = Path(tempfile.gettempdir()) / "fig_example_in.wav"
    torchaudio.save(str(tmp), mix22.reshape(1, -1), SR)
    out = pipe.separate(str(tmp), stems_sel)
    wav, sr = torchaudio.load(out)
    return to22(wav, sr)


def main():
    pipe = MusicSeparationPipeline()
    rows = []   # (label, mixture, ours, truth)

    # ---- MUSDB18-HQ test track, 12 s excerpt
    test_root = ROOT / "datasets" / "musdb18hq" / "test"
    track = sorted(p for p in test_root.iterdir() if p.is_dir())[1]
    mix = load_audio(str(track / "mixture.wav"), offset_sec=60, max_duration_sec=12)
    for name in ("vocals", "drums"):
        truth = load_audio(str(track / f"{name}.wav"), offset_sec=60, max_duration_sec=12)
        ours = run(pipe, mix, [name])
        n = min(ours.shape[-1], truth.shape[-1], mix.shape[-1])
        rows.append((f"MUSDB18-HQ test: {name}", mix[..., :n], ours[..., :n], truth[..., :n]))

    # ---- Saraga validation clip, mridangam
    ds = SaragaCarnaticdataset(SARAGA_ROOT, split="valid", mode="separator", augment=False)
    mr = STEM_LABELS.index("mridangam")
    pick = None
    for i in range(len(ds)):
        m, t, s = ds[i]
        if int(s) == mr and t.pow(2).mean().sqrt() > 5e-3:
            pick = (m, t)
            break
    if pick is not None:
        m, t = pick
        ours = run(pipe, m, ["mridangam"])
        n = min(ours.shape[-1], t.shape[-1], m.shape[-1])
        rows.append(("Saraga validation: mridangam", m[..., :n], ours[..., :n], t[..., :n]))

    fig, axs = plt.subplots(len(rows), 3, figsize=(9.0, 2.45 * len(rows)), sharex=False)
    if len(rows) == 1:
        axs = axs[None, :]
    for r, (label, mix_, ours_, truth_) in enumerate(rows):
        sd = float(si_sdr(ours_.reshape(1, -1), truth_.reshape(1, -1)))
        panels = [("mixture", mix_), (f"our output  (SI-SDR {sd:+.1f} dB)", ours_), ("ground-truth stem", truth_)]
        specs = [spec_db(p[1]) for p in panels]
        vmax = max(s.max() for s in specs)
        for c, ((title, _), S) in enumerate(zip(panels, specs)):
            ax = axs[r, c]
            ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 80, vmax=vmax,
                      extent=[0, S.shape[1] * HOP / SR, 0, SR / 2000])
            ax.set_title(title if r > 0 or c > 0 else "mixture", fontsize=9)
            if c == 0:
                ax.set_ylabel(label + "\nfrequency (kHz)", fontsize=8.5)
            if r == len(rows) - 1:
                ax.set_xlabel("time (s)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)
    print([(r[0], round(float(si_sdr(r[2].reshape(1, -1), r[3].reshape(1, -1))), 2)) for r in rows])


if __name__ == "__main__":
    main()
