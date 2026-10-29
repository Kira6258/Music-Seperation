"""
train_separator.py
------------------
Training script for the QueryConditionedSeparator (U-Net + FiLM).

Usage:
    python training/train_separator.py
    python training/train_separator.py --dry-run
    python training/train_separator.py --epochs 100 --batch-size 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from models.separator import QueryConditionedSeparator
from utils.musdb_dataset import MUSDB18Dataset
from utils.saraga_dataset import SaragaCarnaticdataset
from utils.audio_utils import (
    stft, istft, si_sdr, si_sdr_loss, STEM_LABELS, N_FFT, HOP_LENGTH, SEGMENT_SAMPLES
)

# num_workers=0 required on Windows — multiprocessing spawn in scripts causes hangs.
# Audio is cached in RAM so 0 workers is fast enough.
NUM_WORKERS = 0

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
MUSDB_ROOT        = PROJECT_ROOT / "datasets" / "musdb18hq"
SARAGA_ROOT       = PROJECT_ROOT / "datasets" / "saraga" / "carnatic" / "saraga1.5_carnatic"
PHILHARMONIA_ROOT = PROJECT_ROOT / "datasets" / "Philharmonia"
CHECKPOINTS       = PROJECT_ROOT / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser(description="Train Query-Conditioned Separator")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--neg-prob",   type=float, default=0.2,
                        help="fraction of absent-query (silent target) samples")
    parser.add_argument("--tag",        type=str,   default="v2",
                        help="checkpoints are written to separator_<tag>_best.pt, "
                             "so the app's current model is never overwritten")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--musdb-seconds", type=float, default=150.0)
    parser.add_argument("--saraga-seconds", type=float, default=90.0)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--resume",     type=str,   default=None)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Batch → STFT magnitude
# ─────────────────────────────────────────────────────────────────────────────

def batch_to_mag(waveforms: torch.Tensor, device: torch.device):
    """
    Convert batch of waveforms (B, 1, T) to STFT magnitude (B, 1, F, T_spec).
    Vectorized directly on GPU in a single kernel call.
    """
    w = waveforms.to(device)
    if w.dim() == 3:
        w = w.squeeze(1)  # (B, T)
    window = torch.hann_window(N_FFT, device=device)
    spec = torch.stft(w, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    mag = spec.abs().unsqueeze(1)   # (B, 1, F, T_spec)
    return mag, spec


def mag_to_wav(estimated_mag: torch.Tensor,
               original_specs: torch.Tensor,
               original_length: int,
               device: torch.device) -> torch.Tensor:
    """
    Apply estimated magnitude mask to original phase, then vectorized iSTFT on GPU.
    Returns: (B, original_length) tensor directly on GPU in float32.
    """
    estimated_mag = estimated_mag.float()
    original_specs = original_specs.to(torch.complex64)
    phase = original_specs / (original_specs.abs() + 1e-8)
    masked = estimated_mag.squeeze(1) * phase   # (B, F, T_spec)
    window = torch.hann_window(N_FFT, device=device)
    wavs = torch.istft(masked, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       window=window, length=original_length)  # (B, original_length)
    return wavs


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation steps
# ─────────────────────────────────────────────────────────────────────────────

LOG_SCALE = 100.0   # log1p(LOG_SCALE * |X|): makes quiet (hiss/bleed) bins matter


def run_batch(model, mixture_wav, target_wav, stem_idx, device):
    """
    Forward pass for one batch fully on GPU.
    Returns: (total_loss, rel_l1, sdr_val, est_batch)

    Loss is scale-free so no single term dominates (the old 0.1*dB SI-SDR term
    was ~100x larger than the L1 term and, being scale-invariant, let the mask
    drift to arbitrary gains / a hissy noise floor):
      rel_l1       : |est-tgt| normalised by the mixture's mean magnitude
      loss_log     : L1 in log-magnitude -> penalises residual noise in quiet bins
      loss_mask    : MSE to the ideal ratio mask
      loss_interf  : mask energy over non-target bins
      loss_sdr     : SI-SDR, only on samples whose target is non-silent
    Absent-query samples (all-zero target) therefore teach the model to output
    silence instead of noise.
    """
    mix_mag, mix_specs = batch_to_mag(mixture_wav, device)
    tgt_mag, _ = batch_to_mag(target_wav, device)
    stem_idx = stem_idx.to(device)

    target = target_wav.to(device).float()
    if target.dim() == 3:
        target = target.squeeze(1)

    mask = model(mix_mag, stem_idx)           # (B, 1, F, T)
    est_mag = mix_mag * mask

    mix_scale = mix_mag.mean(dim=(1, 2, 3), keepdim=True) + 1e-6
    rel_l1 = ((est_mag - tgt_mag).abs() / mix_scale).mean()

    loss_log = F.l1_loss(torch.log1p(LOG_SCALE * est_mag),
                         torch.log1p(LOG_SCALE * tgt_mag))

    ideal_mask = torch.clamp(tgt_mag / (mix_mag + 1e-8), 0.0, 1.0)
    loss_mask = F.mse_loss(mask, ideal_mask)

    background_mag = F.relu(mix_mag - tgt_mag) / mix_scale
    loss_interf = (mask * background_mag).mean()

    est_batch = mag_to_wav(est_mag, mix_specs, SEGMENT_SAMPLES, device)

    # SI-SDR only where a target exists (it is undefined for an empty target)
    has_target = target.pow(2).mean(dim=-1).sqrt() > 1e-4
    if has_target.any():
        sdr_each = si_sdr(est_batch[has_target], target[has_target], reduction="none")
        sdr_val = sdr_each.mean()
        loss_sdr = -torch.clamp(sdr_each, min=-30.0, max=30.0).mean()   # in dB
    else:
        sdr_val = torch.zeros((), device=device)
        loss_sdr = torch.zeros((), device=device)

    total_loss = (rel_l1 + 0.5 * loss_log + 0.5 * loss_mask
                  + 0.2 * loss_interf + 0.05 * loss_sdr)   # 0.05/dB: waveform-level pressure
    return total_loss, rel_l1, sdr_val, est_batch


def evaluate(model, loader, device, max_batches=20):
    """Quick validation over a subset of the val set."""
    model.eval()
    total_l1 = 0.0
    total_sdr = 0.0
    n = 0
    with torch.no_grad():
        for i, (mix, tgt, s_idx) in enumerate(loader):
            if i >= max_batches:
                break
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                _, l1_loss, sdr_val, _ = run_batch(model, mix, tgt, s_idx, device)
            if not (torch.isnan(l1_loss) or torch.isnan(sdr_val)):
                total_l1 += l1_loss.item()
                total_sdr += sdr_val.item()
                n += 1
    return (total_l1 / max(n, 1)), (total_sdr / max(n, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_separator] Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets...")
    # neg_prob: fraction of "absent query" samples (zero target) in training.
    # Validation uses neg_prob=0 so val SI-SDR stays meaningful.
    musdb_train  = MUSDB18Dataset(MUSDB_ROOT, split="train", mode="separator", neg_prob=args.neg_prob,
                                max_seconds=args.musdb_seconds)
    musdb_valid  = MUSDB18Dataset(MUSDB_ROOT, split="valid", mode="separator", augment=False,
                                  max_seconds=args.musdb_seconds)
    saraga_train = SaragaCarnaticdataset(SARAGA_ROOT, split="train", mode="separator", neg_prob=args.neg_prob,
                                       max_duration_sec=args.saraga_seconds)
    saraga_valid = SaragaCarnaticdataset(SARAGA_ROOT, split="valid", mode="separator", augment=False,
                                         max_duration_sec=args.saraga_seconds)

    train_sets = [musdb_train, saraga_train]
    val_sets   = [musdb_valid, saraga_valid]

    train_dataset = ConcatDataset(train_sets)
    val_dataset   = ConcatDataset(val_sets)

    print(f"Train examples: {len(train_dataset)} (MUSDB: {len(musdb_train)}, Saraga: {len(saraga_train)})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    print(f"Val examples:   {len(val_dataset)}   (MUSDB: {len(musdb_valid)}, Saraga: {len(saraga_valid)})")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = QueryConditionedSeparator(base_channels=args.base_channels).to(device)
    config = {"base_channels": args.base_channels}

    # Use mixed precision on GPU
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    start_epoch = 0
    best_val_sdr = float("-inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = ckpt["model"]
        try:
            model.load_state_dict(state_dict)
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    pass
            start_epoch = ckpt.get("epoch", 0)
            best_val_sdr = ckpt.get("val_sdr", float("-inf"))
            print(f"Resumed from epoch {start_epoch}")
        except RuntimeError:
            cur_state = model.state_dict()
            for k, v in state_dict.items():
                if k in cur_state and cur_state[k].shape == v.shape:
                    cur_state[k] = v
                elif k == "stem_embedding.weight":
                    n_common = min(cur_state[k].shape[0], v.shape[0])
                    cur_state[k][:n_common] = v[:n_common]
            model.load_state_dict(cur_state)
            print("[train_separator] Preloaded backbone weights from checkpoint (partial match).")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_l1 = 0.0
        running_sdr = 0.0
        n_train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (mix, tgt, s_idx) in enumerate(pbar):
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss, l1_loss, sdr_val, _ = run_batch(model, mix, tgt, s_idx, device)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if not (torch.isnan(l1_loss) or torch.isnan(sdr_val)):
                running_l1 += l1_loss.item()
                running_sdr += sdr_val.item()
                n_train_batches += 1
                pbar.set_postfix(L1=f"{l1_loss.item():.4f}", SDR=f"{sdr_val.item():.2f}dB")

            if args.dry_run:
                print(f"\n[dry-run] Batch OK. L1 loss = {l1_loss.item():.4f}, SI-SDR = {sdr_val.item():.2f} dB")
                print("[dry-run] Single batch passed. Exiting.")
                return

        train_l1  = running_l1 / max(n_train_batches, 1)
        train_sdr = running_sdr / max(n_train_batches, 1)
        val_l1, val_sdr = evaluate(model, val_loader, device)
        scheduler.step()

        # GPU memory report
        gpu_mem = ""
        if device.type == "cuda":
            alloc  = torch.cuda.memory_allocated(device) / 1024**2
            reserv = torch.cuda.memory_reserved(device)  / 1024**2
            gpu_mem = f" | GPU {alloc:.0f}/{reserv:.0f} MB"

        print(f"\nEpoch {epoch+1:03d} | "
              f"train relL1={train_l1:.4f} (SDR={train_sdr:.2f}dB) | "
              f"val relL1={val_l1:.4f} (SDR={val_sdr:.2f}dB){gpu_mem}")

        if val_sdr > best_val_sdr:
            best_val_sdr = val_sdr
            ckpt_path = CHECKPOINTS / f"separator_{args.tag}_best.pt"
            torch.save({
                "epoch":     epoch + 1,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  val_l1,
                "val_sdr":   val_sdr, "config": config,
            }, ckpt_path)
            print(f"  [SAVED] Best separator (val SI-SDR {val_sdr:.2f} dB) -> {ckpt_path}")

        torch.save({
            "epoch":   epoch + 1,
            "model":   model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_l1,
            "val_sdr":  val_sdr, "config": config,
        }, CHECKPOINTS / f"separator_{args.tag}_latest.pt")

    print(f"\nDone. Best val SI-SDR: {best_val_sdr:.2f} dB")


if __name__ == "__main__":
    main()
