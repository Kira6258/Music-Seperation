"""
make_figures.py
---------------
Generates the figures used in the project report and writes them to
output/images/.  Diagrams are drawn with matplotlib patches; charts use the
measured values reported in Chapter 5 (validation / test numbers recorded
while training and evaluating the final checkpoints).

Run:  python output/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 100,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

NAVY, BLUE, LIGHT = "#1F3A5F", "#2E75B6", "#DCE6F2"
ORANGE, SAND = "#C55A11", "#FBE5D6"
GREEN, MINT = "#548235", "#E2EFDA"
GRAY, SILVER = "#7F7F7F", "#E7E6E6"
INK = "#222222"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=9, bold=False, lw=1.2, tc=INK, r=0.06):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.2, style="-|>", ls="-", rad=0.0, ms=11):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


# ──────────────────────────────────────────────────────────────────────────
# Figure 3.1  Query-conditioned U-Net with FiLM
# ──────────────────────────────────────────────────────────────────────────
def fig_3_1():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(-0.3, 6.5)
    ax.axis("off")

    # input path
    box(ax, 0.0, 4.4, 1.5, 0.9, "Mixture\n|STFT|  (F x T)", fc=SILVER, ec=GRAY, fs=8.5)
    box(ax, 0.0, 3.05, 1.5, 0.75, "log1p(10 |X|)", fc=SILVER, ec=GRAY, fs=8.5)
    arrow(ax, 0.75, 4.4, 0.75, 3.8)

    # encoder blocks (each carries a FiLM layer); drawn as one box
    enc_x = [1.95, 3.0, 4.05, 5.1]
    enc_y = [3.0, 2.15, 1.3, 0.45]
    chs = [64, 128, 256, 512]
    for i, (x, y, c) in enumerate(zip(enc_x, enc_y, chs)):
        box(ax, x, y, 1.0, 0.85, f"Enc {i+1}\n{c} ch\n+ FiLM", fc=LIGHT, ec=NAVY, fs=8)
    arrow(ax, 1.5, 3.42, 1.95, 3.42, lw=1.0, ms=9)
    for i in range(3):
        arrow(ax, enc_x[i] + 0.5, enc_y[i], enc_x[i + 1] + 0.1, enc_y[i + 1] + 0.7, lw=1.0, ms=9, rad=-0.15)

    # bottleneck
    box(ax, 6.35, 0.2, 1.15, 0.7, "Bottleneck\n512 ch, 3x3", fc=LIGHT, ec=NAVY, fs=8)
    arrow(ax, enc_x[3] + 1.0, enc_y[3] + 0.3, 6.35, 0.55, lw=1.0, ms=9)

    # decoder blocks
    dec_x = [7.75, 8.8, 9.85]
    dec_y = [0.45, 1.3, 2.15]
    dchs = [256, 128, 64]
    for x, y, c, n in zip(dec_x, dec_y, dchs, ["Dec 4", "Dec 3", "Dec 2"]):
        box(ax, x, y, 1.0, 0.85, f"{n}\n{c} ch", fc=MINT, ec=GREEN, fs=8)
    arrow(ax, 7.5, 0.55, 7.75, 0.75, lw=1.0, ms=9)
    for i in range(2):
        arrow(ax, dec_x[i] + 0.5, dec_y[i] + 0.85, dec_x[i + 1] + 0.1, dec_y[i + 1] + 0.12, lw=1.0, ms=9, rad=-0.15)
    box(ax, 9.65, 3.0, 1.6, 0.85, "Dec 1 (64 ch)\n1x1 conv\nsigmoid", fc=MINT, ec=GREEN, fs=8)
    arrow(ax, dec_x[2] + 0.5, dec_y[2] + 0.85, 10.2, 3.0, lw=1.0, ms=9, rad=-0.1)

    # skip connections (same level, concatenated before the decoder block)
    for ei, dx, dy in [(2, 8.8, 1.3), (1, 9.85, 2.15), (0, 9.65, 3.0)]:
        arrow(ax, enc_x[ei] + 1.0, enc_y[ei] + 0.42, dx, dy + 0.42, color=GRAY, lw=1.0, ls=(0, (4, 3)), ms=8)
    ax.text(7.0, 2.7, "skip connections\n(concatenate)", fontsize=8, color=GRAY, ha="center", style="italic")

    # query path
    box(ax, 2.75, 5.55, 1.6, 0.6, "Stem ID (0 to 7)", fc=SAND, ec=ORANGE, fs=8.5)
    box(ax, 5.0, 5.55, 2.1, 0.6, "Embedding  e  (64-d)", fc=SAND, ec=ORANGE, fs=8.5, bold=True)
    arrow(ax, 4.35, 5.85, 5.0, 5.85)
    for x, y in zip(enc_x, enc_y):
        arrow(ax, 5.0 if x < 4.5 else 5.7, 5.55, x + 0.5, y + 0.85, color=ORANGE, lw=0.9, ls=(0, (2, 2)), ms=7)
    ax.text(7.25, 5.85, "FiLM: gamma, beta = FC(e)\nat every encoder level", fontsize=8, color=ORANGE, ha="left", va="center")

    # output
    box(ax, 9.3, 4.55, 2.05, 1.0, "mask M in [0, 1]\nM x |X|, mixture phase,\niSTFT", fc=SILVER, ec=GRAY, fs=8)
    arrow(ax, 10.45, 3.85, 10.45, 4.55)

    save(fig, "fig_3_1_unet_film.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 3.2  Competitive mask (schematic)
# ──────────────────────────────────────────────────────────────────────────
def fig_3_2():
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    # left: independent masks do not add to one
    labels = ["bin 1", "bin 2", "bin 3", "bin 4", "bin 5", "bin 6"]
    srcs = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    cols = [BLUE, ORANGE, GREEN, "#7030A0", "#BF9000", GRAY]
    # illustrative magnitudes of the six Demucs outputs in six bins
    A = np.array([
        [0.90, 0.10, 0.05, 0.20, 0.05, 0.05],
        [0.10, 0.85, 0.05, 0.10, 0.05, 0.10],
        [0.05, 0.05, 0.90, 0.10, 0.05, 0.10],
        [0.30, 0.05, 0.05, 0.60, 0.40, 0.20],
        [0.05, 0.05, 0.05, 0.30, 0.70, 0.20],
        [0.25, 0.20, 0.10, 0.30, 0.25, 0.60],
    ]).T  # (sources, bins)
    W = A ** 2
    share = W / W.sum(0, keepdims=True)
    ax = axs[0]
    bottom = np.zeros(6)
    for i, s in enumerate(srcs):
        ax.bar(labels, share[i], bottom=bottom, color=cols[i], label=s, width=0.72, edgecolor="white", lw=0.6)
        bottom += share[i]
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of the bin  (W_s / sum W)")
    ax.set_title("Each bin is divided among all sources", fontsize=10)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.tick_params(axis="x", labelsize=8)

    # right: selecting guitar keeps only its share
    ax = axs[1]
    sel = 3
    keep = share[sel]
    ax.bar(labels, np.ones(6), color=SILVER, width=0.72, label="mixture (1.0)")
    ax.bar(labels, keep, color="#7030A0", width=0.72, label="kept: guitar share")
    ax.set_ylim(0, 1.0)
    ax.set_title("Selected stem keeps only its share", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    ax.tick_params(axis="x", labelsize=8)
    fig.text(0.5, -0.02, "Illustrative values. Weights W_s = |source_s|^2, so a bin dominated by another stem is almost fully removed.",
             ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig_3_2_competitive_mask.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 4.1  System architecture
# ──────────────────────────────────────────────────────────────────────────
def fig_4_1():
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    box(ax, 0.1, 2.7, 1.6, 1.0, "Input\naudio / video", fc=SILVER, ec=GRAY, fs=9, bold=True)
    box(ax, 2.3, 2.55, 2.0, 1.3, "Stem detector\nlog-mel CNN\n(trained, 0.42 M)", fc=LIGHT, ec=NAVY, fs=9)
    box(ax, 5.0, 2.7, 1.7, 1.0, "User selects\nstems", fc=SILVER, ec=GRAY, fs=9, bold=True)
    arrow(ax, 1.7, 3.2, 2.3, 3.2)
    arrow(ax, 4.3, 3.2, 5.0, 3.2)
    ax.text(4.65, 3.5, "present\nstems", fontsize=7.5, ha="center", color=GRAY)

    # backends
    box(ax, 7.4, 4.0, 2.4, 1.45, "Demucs-6 backend\n(pretrained)\ndrums, bass, vocals,\nguitar, piano, other", fc=SILVER, ec=GRAY, fs=8.5)
    box(ax, 7.4, 0.95, 2.4, 1.45, "Query-conditioned U-Net\n(trained, 8.8 M)\nviolin, mridangam,\nghatam masks", fc=SAND, ec=ORANGE, fs=8.5)
    arrow(ax, 6.7, 3.45, 7.4, 4.6)
    arrow(ax, 6.7, 2.95, 7.4, 1.85)

    box(ax, 10.3, 2.35, 1.6, 1.7, "Competitive\nmask\n(W = |s|^2)\n+ loudness\nmakeup", fc=MINT, ec=GREEN, fs=8.5, bold=True)
    arrow(ax, 9.8, 4.6, 10.3, 3.7)
    arrow(ax, 9.8, 1.85, 10.3, 2.7)

    box(ax, 10.3, 0.25, 1.6, 0.8, "Selected stems\n(WAV / video)", fc=SILVER, ec=GRAY, fs=8.5, bold=True)
    arrow(ax, 11.1, 2.35, 11.1, 1.05)

    # legend
    ax.add_patch(FancyBboxPatch((0.1, 0.2), 0.3, 0.2, boxstyle="round,pad=0,rounding_size=0.03", fc=SAND, ec=ORANGE))
    ax.text(0.5, 0.3, "trained in this project", fontsize=8, va="center")
    ax.add_patch(FancyBboxPatch((3.0, 0.2), 0.3, 0.2, boxstyle="round,pad=0,rounding_size=0.03", fc=LIGHT, ec=NAVY))
    ax.text(3.4, 0.3, "trained detector", fontsize=8, va="center")
    ax.add_patch(FancyBboxPatch((5.2, 0.2), 0.3, 0.2, boxstyle="round,pad=0,rounding_size=0.03", fc=SILVER, ec=GRAY))
    ax.text(5.6, 0.3, "pretrained / interface", fontsize=8, va="center")

    for xx, t in [(3.3, "Stage 1: detect"), (5.85, "Stage 2: choose"), (9.1, "Stage 3: separate and clean up")]:
        ax.text(xx, 6.1, t, ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
    save(fig, "fig_4_1_system_architecture.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 4.2  Datasets
# ──────────────────────────────────────────────────────────────────────────
def fig_4_2():
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    names = ["MUSDB18-HQ", "Saraga Carnatic"]
    train, val, test = [84, 132], [16, 36], [50, 0]
    x = np.arange(2)
    ax.bar(x, train, color=BLUE, width=0.55, label="train")
    ax.bar(x, val, bottom=train, color=ORANGE, width=0.55, label="validation")
    ax.bar(x, test, bottom=np.array(train) + np.array(val), color=GRAY, width=0.55, label="test")
    for i in range(2):
        ax.text(i, train[i] / 2, str(train[i]), ha="center", va="center", color="white", fontsize=9)
        ax.text(i, train[i] + val[i] / 2, str(val[i]), ha="center", va="center", color="white", fontsize=9)
    ax.text(0, train[0] + val[0] + test[0] / 2, "50", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("tracks")
    ax.set_title("Track split", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0, 215)

    ax = axs[1]
    stems = ["vocals", "drums", "bass", "guitar", "piano", "violin", "mridangam", "ghatam"]
    M = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0],   # MUSDB
        [1, 0, 0, 0, 0, 1, 1, 1],   # Saraga
    ])
    ax.pcolormesh(M, cmap=matplotlib.colors.ListedColormap([SILVER, BLUE]), edgecolors="white", linewidth=2, vmin=0, vmax=1)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_xticklabels(stems, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["MUSDB18-HQ", "Saraga"], fontsize=9)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.text(4.0, 0.5, "single 'other' stem:\nlabels masked", ha="center", va="center", fontsize=7.5, color=INK)
    ax.text(7.5, 1.5, "46\ntracks", ha="center", va="center", fontsize=7.5, color="white")
    ax.set_title("Stems with ground truth (blue)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_4_2_datasets.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 4.3  Stem detector
# ──────────────────────────────────────────────────────────────────────────
def fig_4_3():
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    x = 0.1
    items = [("Log-mel\n1 x 128 x T", SILVER, GRAY, 1.35)]
    blocks = [32, 64, 128, 256]
    for c in blocks:
        items.append((f"Conv 3x3\nBN, ReLU\nMaxPool\n{c} ch", LIGHT, NAVY, 1.2))
    items.append(("Global\naverage\npool", LIGHT, NAVY, 1.05))
    items.append(("FC 128\nReLU\nDropout 0.3", LIGHT, NAVY, 1.2))
    items.append(("FC 8\nsigmoid", MINT, GREEN, 1.0))
    xs = []
    for t, fc, ec, w in items:
        box(ax, x, 1.0, w, 1.5, t, fc=fc, ec=ec, fs=8)
        xs.append((x, w))
        x += w + 0.32
    for (x0, w0), (x1, _) in zip(xs[:-1], xs[1:]):
        arrow(ax, x0 + w0, 1.75, x1, 1.75, ms=9)
    last_x, last_w = xs[-1]
    ax.text(last_x + last_w / 2, 0.55, "p(stem present)\nper-class threshold", ha="center", fontsize=8, color=GREEN)
    ax.text(6.0, 3.05, "Windows of 6 s, logits averaged over the song; guitar and piano heads are masked in training",
            ha="center", fontsize=8.5, color=GRAY)
    save(fig, "fig_4_3_detector.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 5.1  Validation progress
# ──────────────────────────────────────────────────────────────────────────
def fig_5_1():
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    stages = ["Leak removed,\nSaraga targets\nmisaligned", "Saraga trained\non exact\nstem sum", "Longer clips\n(120 s / 60 s)", "8.8 M model,\n150 s / 90 s,\n150 epochs"]
    vals = [-8.48, 1.81, 2.15, 2.53]
    cols = [ORANGE, BLUE, BLUE, NAVY]
    b = ax.bar(range(4), vals, color=cols, width=0.6)
    ax.axhline(0, color=INK, lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.35 if v > 0 else -0.9), f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel("best validation SI-SDR (dB)")
    ax.set_ylim(-10.5, 4.5)
    ax.set_title("Validation SI-SDR at each stage of the project", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig_5_1_validation_progress.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 5.2  Per-stem SI-SDR: mixture vs model vs oracle
# ──────────────────────────────────────────────────────────────────────────
def fig_5_2():
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    data = {
        "MUSDB18-HQ": (["vocals", "drums", "bass"],
                       [(-3.6, 4.2, 10.1), (-6.5, 2.3, 6.5), (-6.6, 0.2, 4.2)]),
        "Saraga": (["vocals", "mridangam", "violin"],
                   [(0.8, 5.6, 9.7), (-2.4, 5.2, 5.8), (-5.9, 1.1, 3.8)]),
    }
    for ax, (title, (stems, vals)) in zip(axs, data.items()):
        n = len(stems)
        x = np.arange(n)
        w = 0.26
        v = np.array(vals)
        ax.bar(x - w, v[:, 0], w, color=GRAY, label="raw mixture")
        ax.bar(x, v[:, 1], w, color=NAVY, label="our U-Net")
        ax.bar(x + w, v[:, 2], w, color="#9DC3E6", label="ideal mask (oracle reference)")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(stems, fontsize=9)
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel("SI-SDR (dB)")
    axs[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_5_2_per_stem_sisdr.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 5.3  Detector F1
# ──────────────────────────────────────────────────────────────────────────
def fig_5_3():
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    stems = ["vocals", "drums", "bass", "violin", "mridangam", "ghatam"]
    f1 = [0.85, 0.89, 0.88, 0.78, 0.74, 0.65]
    cols = [BLUE] * 3 + [ORANGE] * 3
    ax.bar(stems, f1, color=cols, width=0.6)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5)
    macro = float(np.mean(f1))
    ax.axhline(macro, color=INK, lw=1.0, ls="--")
    ax.text(5.45, macro + 0.015, f"macro-F1 = {macro:.3f}", ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("validation F1")
    ax.set_title("Stem detector: F1 per instrument (blue Western, orange Carnatic)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_5_3_detector_f1.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 5.4  Ours vs best-of-6 Demucs on Carnatic stems
# ──────────────────────────────────────────────────────────────────────────
def fig_5_4():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    stems = ["vocals", "violin", "mridangam"]
    dem = [4.50, -1.85, -4.35]
    ours = [5.28, 0.98, 0.06]
    x = np.arange(3)
    w = 0.34
    ax.bar(x - w / 2, dem, w, color=GRAY, label="best of six Demucs outputs")
    ax.bar(x + w / 2, ours, w, color=NAVY, label="our trained U-Net")
    ax.axhline(0, color=INK, lw=0.8)
    for i in range(3):
        ax.text(x[i] - w / 2, dem[i] + (0.2 if dem[i] > 0 else -0.55), f"{dem[i]:.2f}", ha="center", fontsize=8.5)
        ax.text(x[i] + w / 2, ours[i] + 0.2, f"{ours[i]:.2f}", ha="center", fontsize=8.5)
        ax.text(x[i], 6.4, f"+{ours[i]-dem[i]:.1f} dB", ha="center", fontsize=10, fontweight="bold", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(stems)
    ax.set_ylim(-6, 7.4)
    ax.set_ylabel("SI-SDR vs true stem (dB)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("Carnatic validation tracks (12 tracks, 20 s windows)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_5_4_vs_demucs.png")


# ──────────────────────────────────────────────────────────────────────────
# Figure 5.5  SIR: U-Net alone vs final system
# ──────────────────────────────────────────────────────────────────────────
def fig_5_5():
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    stems = ["vocals", "drums", "bass", "guitar + piano"]
    unet = [16.1, 5.3, 4.5, -1.7]
    final = [28.2, 27.6, 11.0, 20.2]
    x = np.arange(4)
    w = 0.34
    ax.bar(x - w / 2, unet, w, color=GRAY, label="U-Net masks alone")
    ax.bar(x + w / 2, final, w, color=NAVY, label="final system (competitive mask)")
    ax.axhline(0, color=INK, lw=0.8)
    for i in range(4):
        ax.text(x[i] - w / 2, unet[i] + (0.5 if unet[i] >= 0 else -1.7), f"{unet[i]:.1f}", ha="center", fontsize=8.5)
        ax.text(x[i] + w / 2, final[i] + 0.5, f"{final[i]:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(stems)
    ax.set_ylabel("SIR (dB), higher = less bleed")
    ax.set_ylim(-4, 33)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Leakage of unselected stems (4 MUSDB test tracks, 40 s each)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_5_5_sir.png")


if __name__ == "__main__":
    fig_3_1()
    fig_3_2()
    fig_4_1()
    fig_4_2()
    fig_4_3()
    fig_5_1()
    fig_5_2()
    fig_5_3()
    fig_5_4()
    fig_5_5()
