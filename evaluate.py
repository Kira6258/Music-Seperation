"""
evaluate.py
-----------
Formal SI-SDR Evaluation Script — Review 2 Proof

Runs the pipeline on MUSDB18-HQ test tracks, computes SI-SDR for each
stem that has a ground-truth .wav file, prints per-track results,
and saves a summary table + bar chart to outputs/evaluation_report.png

Usage:
    python evaluate.py                    # evaluates all test tracks
    python evaluate.py --max-tracks 10   # only first N tracks (faster)
    python evaluate.py --stem vocals      # only evaluate one stem type
"""

import sys
import os
from pathlib import Path
import argparse
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torchaudio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from inference.pipeline import MusicSeparationPipeline
from utils.audio_utils import load_audio, si_sdr, SAMPLE_RATE

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_DIR = PROJECT_ROOT / "datasets" / "musdb18hq" / "test"

# Stems that have ground-truth files in MUSDB18-HQ
GT_STEMS = ["vocals", "drums", "bass", "other"]

# Map stems to their ground truth filename in MUSDB18-HQ
EVAL_STEMS = {
    "vocals": "vocals.wav",
    "drums":  "drums.wav",
    "bass":   "bass.wav",
    "guitar": "other.wav",
}


def sep(char="-", width=72):
    return char * width


def compute_sisdr_for_track(pipeline, track_dir: Path, stem: str, mode: str = "hybrid") -> float | None:
    """
    Separate `stem` from mixture.wav, compare against ground truth stem file.
    Returns SI-SDR in dB or None if ground truth is missing.
    """
    mixture_path = track_dir / "mixture.wav"
    gt_filename = EVAL_STEMS.get(stem, f"{stem}.wav")
    gt_path = track_dir / gt_filename

    if not mixture_path.exists() or not gt_path.exists():
        return None

    # Separate
    out_path = str(OUTPUT_DIR / f"_eval_tmp_{stem}.wav")
    try:
        pipeline.separate(str(mixture_path), [stem], output_path=out_path, mode=mode)
    except Exception as e:
        print(f"    [ERROR] Separation failed: {e}")
        return None

    # Load estimate and ground truth, align lengths
    try:
        est = load_audio(out_path)                  # (1, T_est)
        gt  = load_audio(str(gt_path))              # (1, T_gt)
        min_len = min(est.shape[-1], gt.shape[-1])
        if min_len < SAMPLE_RATE * 2:               # skip if < 2 seconds
            return None
        score = float(si_sdr(est[:, :min_len], gt[:, :min_len]).cpu())
        # Clamp extreme outliers (silent tracks give -inf or +inf)
        if not np.isfinite(score):
            return None
        score = float(np.clip(score, -20.0, 30.0))
    except Exception as e:
        print(f"    [ERROR] SI-SDR compute failed: {e}")
        return None

    return score


def plot_results(results: dict, output_path: str):
    """
    results = { stem: [(track_name, score), ...], ... }
    Saves a bar chart + summary stats figure.
    """
    stems = list(results.keys())
    n_stems = len(stems)

    colors = {
        "vocals": "#e74c3c",
        "drums":  "#3498db",
        "bass":   "#2ecc71",
        "guitar": "#f39c12",
    }

    fig = plt.figure(figsize=(14, 5 * n_stems + 3))
    gs  = gridspec.GridSpec(n_stems + 1, 1, hspace=0.55)

    for i, stem in enumerate(stems):
        data = results[stem]
        if not data:
            continue
        names  = [d[0][:35] for d in data]    # truncate long names
        scores = [d[1] for d in data]
        avg    = np.mean(scores)

        ax = fig.add_subplot(gs[i])
        bar_color = colors.get(stem, "#9b59b6")
        bars = ax.barh(range(len(names)), scores, color=bar_color, alpha=0.78, height=0.6)
        ax.axvline(avg, color='white', linestyle='--', linewidth=1.8,
                   label=f"Mean = {avg:.2f} dB")
        ax.axvline(0,   color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Value labels
        for j, (bar, s) in enumerate(zip(bars, scores)):
            ax.text(s + 0.15, j, f"{s:.1f}", va='center', fontsize=7.5, color='white')

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(
            f"SI-SDR per Track — {stem.upper()}   (Mean = {avg:.2f} dB | n={len(scores)})",
            fontsize=11, fontweight='bold', color='white', pad=8
        )
        ax.set_xlabel("SI-SDR (dB)  — higher is better", fontsize=9, color='#cccccc')
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='#cccccc')
        ax.spines['bottom'].set_color('#555')
        ax.spines['left'].set_color('#555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, facecolor='#2d2d3f', labelcolor='white', loc='lower right')
        ax.set_xlim(-15, 25)

    # Summary row
    ax_sum = fig.add_subplot(gs[n_stems])
    ax_sum.axis('off')
    stem_means = []
    rows = []
    for stem in stems:
        data = results[stem]
        if data:
            scores = [d[1] for d in data]
            mean   = np.mean(scores)
            std    = np.std(scores)
            best   = max(scores)
            worst  = min(scores)
            stem_means.append(mean)
            rows.append([stem.upper(), f"{mean:.2f} dB", f"{std:.2f}", f"{best:.2f} dB", f"{worst:.2f} dB", str(len(scores))])

    overall_mean = np.mean(stem_means) if stem_means else 0
    col_labels = ["Stem", "Mean SI-SDR", "Std Dev", "Best Track", "Worst Track", "# Tracks"]
    table = ax_sum.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor('#2d2d3f' if r > 0 else '#c0392b')
        cell.set_text_props(color='white', fontweight='bold' if r == 0 else 'normal')
        cell.set_edgecolor('#444')

    ax_sum.set_title(
        f"EVALUATION SUMMARY   |   Overall Mean SI-SDR = {overall_mean:.2f} dB",
        fontsize=13, fontweight='bold', color='white', pad=12
    )
    ax_sum.set_facecolor('#1e1e2e')

    fig.patch.set_facecolor('#1e1e2e')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e2e')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tracks", type=int, default=15,
                        help="Max number of test tracks to evaluate (default: 15)")
    parser.add_argument("--stem", type=str, default=None,
                        help="Only evaluate one stem (vocals/drums/bass)")
    parser.add_argument("--mode", choices=["hybrid", "custom"], default="hybrid",
                        help="Separation mode: hybrid (Demucs + U-Net) or custom (pure U-Net)")
    args = parser.parse_args()

    print()
    print(sep("="))
    print(" SI-SDR FORMAL EVALUATION — Query-Conditioned Music Separation")
    print(f" MUSDB18-HQ Test Set | Ground-Truth Comparison | Mode: {args.mode.upper()}")
    print(sep("="))
    print()

    if not TEST_DIR.exists():
        print(f"[ERROR] Test directory not found: {TEST_DIR}")
        sys.exit(1)

    # Load pipeline
    print("[INIT] Loading pipeline...")
    t0 = time.time()
    pipeline = MusicSeparationPipeline()
    print(f"[INIT] Ready in {time.time()-t0:.1f}s\n")

    # Find test tracks
    track_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])[:args.max_tracks]
    print(f"Found {len(track_dirs)} test track(s). Evaluating now...\n")

    stems_to_eval = [args.stem] if args.stem and args.stem in EVAL_STEMS else list(EVAL_STEMS.keys())
    results = {s: [] for s in stems_to_eval}

    for t_idx, track_dir in enumerate(track_dirs, 1):
        track_name = track_dir.name
        print(f"[{t_idx:02d}/{len(track_dirs)}] {track_name}")
        for stem in stems_to_eval:
            score = compute_sisdr_for_track(pipeline, track_dir, stem, mode=args.mode)
            if score is not None:
                results[stem].append((track_name, score))
                tag = "[+]" if score > 0 else "[-]"
                print(f"        {stem:<8} {tag} SI-SDR = {score:+.2f} dB")
            else:
                print(f"        {stem:<8}  --  (no ground truth or error)")
        print()

    # Final summary table
    print(sep("="))
    print(" FINAL EVALUATION RESULTS")
    print(sep("="))
    print(f"  {'STEM':<10} {'MEAN SI-SDR':>14}  {'STD':>8}  {'BEST':>8}  {'WORST':>8}  {'N':>4}")
    print(sep("-"))

    overall_scores = []
    for stem in stems_to_eval:
        data = results[stem]
        if not data:
            print(f"  {stem.upper():<10}  No valid results")
            continue
        scores = [d[1] for d in data]
        mean   = np.mean(scores)
        std    = np.std(scores)
        best   = max(scores)
        worst  = min(scores)
        overall_scores.extend(scores)
        print(f"  {stem.upper():<10} {mean:>+12.2f} dB  {std:>7.2f}  {best:>+7.2f}  {worst:>+7.2f}  {len(scores):>4}")

    if overall_scores:
        overall_mean = np.mean(overall_scores)
        print(sep("-"))
        print(f"  {'OVERALL':<10} {overall_mean:>+12.2f} dB  (across all stems & tracks)")
    print(sep("="))

    # Save report
    report_path = str(OUTPUT_DIR / "evaluation_report.png")
    plot_results(results, report_path)
    print(f"\n[SAVED] Evaluation chart -> {report_path}")
    print("[DONE]  Present outputs/evaluation_report.png as proof during Review 2.\n")


if __name__ == "__main__":
    main()
