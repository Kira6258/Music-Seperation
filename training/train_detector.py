"""
train_detector.py
-----------------
Training script for the StemDetector (multi-label CNN classifier).

Usage:
    python training/train_detector.py                 # train from scratch (recommended)
    python training/train_detector.py --dry-run       # one batch only
    python training/train_detector.py --epochs 40

What changed vs. the old version (why the detector was unreliable):
  * labels are computed on the exact crop fed to the network
  * guitar/piano are label-masked (MUSDB only has an undifferentiated "other")
  * synthetic cross-dataset mixtures break the "dataset == label" shortcut
  * validation has NO train/valid leakage (MUSDB `valid` was a copy of `train`)
  * model selection is by macro-F1 (not loss / exact-match accuracy)
  * per-class decision thresholds are tuned on validation and stored in the
    checkpoint (`thresholds`) so the pipeline can use them
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from models.stem_detector import StemDetector, StemDetectorLoss
from utils.musdb_dataset import MUSDB18Dataset
from utils.saraga_dataset import SaragaCarnaticdataset
from utils.mix_dataset import SyntheticMixDetectorDataset, UNTRAINED_STEMS
from utils.audio_utils import STEM_LABELS, STEM_TO_IDX

NUM_WORKERS = 0  # Windows-safe; audio is cached in RAM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MUSDB_ROOT   = PROJECT_ROOT / "datasets" / "musdb18hq"
SARAGA_ROOT  = PROJECT_ROOT / "datasets" / "saraga" / "carnatic" / "saraga1.5_carnatic"
CHECKPOINTS  = PROJECT_ROOT / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

TRAINED_IDX = [i for i, s in enumerate(STEM_LABELS) if s not in UNTRAINED_STEMS]


def get_args():
    p = argparse.ArgumentParser(description="Train Stem Detector")
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--p-synthetic", type=float, default=0.6,
                   help="fraction of training samples that are synthetic mixtures")
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--resume",      type=str,   default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: per-class F1 + threshold tuning
# ─────────────────────────────────────────────────────────────────────────────

def _f1(pred, tgt):
    tp = float((pred * tgt).sum())
    fp = float((pred * (1 - tgt)).sum())
    fn = float(((1 - pred) * tgt).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-8)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_p, all_t, all_m = [], [], []

    for mel, presence, mask in loader:
        mel, presence, mask = mel.to(device), presence.to(device), mask.to(device)
        logits = model(mel)
        total_loss += criterion(logits, presence, mask).item()
        n_batches += 1
        all_p.append(torch.sigmoid(logits).cpu())
        all_t.append(presence.cpu())
        all_m.append(mask.cpu())

    P = torch.cat(all_p).numpy()
    T = torch.cat(all_t).numpy()
    M = torch.cat(all_m).numpy()

    thresholds = [0.5] * len(STEM_LABELS)
    f1s = {}
    for c in TRAINED_IDX:
        valid = M[:, c] > 0
        p, t = P[valid, c], T[valid, c]
        if t.sum() == 0:                      # class absent from val set
            continue
        best_f, best_th = -1.0, 0.5
        for th in np.arange(0.10, 0.91, 0.05):
            f = _f1((p >= th).astype(np.float32), t)
            if f > best_f:
                best_f, best_th = f, float(th)
        thresholds[c] = round(best_th, 2)
        f1s[STEM_LABELS[c]] = (_f1((p >= 0.5).astype(np.float32), t), best_f, best_th)

    macro_f1 = float(np.mean([v[1] for v in f1s.values()])) if f1s else 0.0
    return total_loss / max(n_batches, 1), macro_f1, f1s, thresholds


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_detector] Device: {device}")
    print(f"[train_detector] Stems:  {STEM_LABELS}")
    print(f"[train_detector] Label-masked (untrained) stems: {UNTRAINED_STEMS}")

    print("\nLoading datasets...")
    musdb_train  = MUSDB18Dataset(MUSDB_ROOT, split="train", mode="detector")
    musdb_valid  = MUSDB18Dataset(MUSDB_ROOT, split="valid", mode="detector")
    saraga_train = SaragaCarnaticdataset(SARAGA_ROOT, split="train", mode="detector")
    saraga_valid = SaragaCarnaticdataset(SARAGA_ROOT, split="valid", mode="detector")

    real_train = ConcatDataset([musdb_train, saraga_train])
    real_val   = ConcatDataset([musdb_valid, saraga_valid])

    train_dataset = SyntheticMixDetectorDataset(
        real_train, [musdb_train, saraga_train], p_synthetic=args.p_synthetic)
    # validation = real val items + a fixed-size block of synthetic val mixes
    synth_val = SyntheticMixDetectorDataset(
        real_val, [musdb_valid, saraga_valid], length=300, p_synthetic=1.0)
    val_dataset = ConcatDataset([real_val, synth_val])

    print(f"Train examples/epoch: {len(train_dataset)} | Val examples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    model = StemDetector().to(device)
    criterion = StemDetectorLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch, best_f1 = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        start_epoch = ckpt.get("epoch", 0)
        best_f1 = ckpt.get("val_f1", -1.0)
        print(f"Resumed from epoch {start_epoch}, val_f1={best_f1:.4f}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")

        for mel, presence, mask in pbar:
            mel, presence, mask = mel.to(device), presence.to(device), mask.to(device)

            optimizer.zero_grad()
            loss = criterion(model(mel), presence, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if args.dry_run:
                print("\n[dry-run] Single batch passed. Exiting.")
                return

        val_loss, val_f1, f1s, thresholds = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch+1:03d} | train_loss={running/max(n,1):.4f} | "
              f"val_loss={val_loss:.4f} | val_macroF1={val_f1:.3f}")
        for s, (f05, fbest, th) in f1s.items():
            print(f"    {s:10s} F1@0.5={f05:.3f}  F1@best={fbest:.3f}  (thr={th:.2f})")

        state = {
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss, "val_f1": val_f1, "thresholds": thresholds,
        }
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(state, CHECKPOINTS / "stem_detector_best.pt")
            print(f"  [SAVED] Best detector (macro-F1 {val_f1:.3f})")
        torch.save(state, CHECKPOINTS / "stem_detector_latest.pt")

    print(f"\nTraining complete. Best val macro-F1: {best_f1:.3f}")


if __name__ == "__main__":
    main()
