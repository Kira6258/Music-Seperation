"""
diagnose_separator.py
---------------------
Tells you WHY separator SI-SDR is low, per dataset and per stem.

For every validation example it computes SI-SDR (dB) of:
  mix    : the raw mixture used as the estimate (trivial baseline)
  oracle : mixture * ideal-ratio-mask (the CEILING for any mask model here)
  model  : your trained separator

How to read it:
  * oracle low (< ~5 dB) for a dataset/stem  -> mixture and stem are misaligned
    or at different levels: the DATA is the problem, not the model.
  * oracle high but model ~= mix             -> model is under-trained / collapsed.
  * model clearly > mix                      -> it is really separating.

Usage:
    python diagnose_separator.py                 # uses checkpoints/separator_best.pt
    python diagnose_separator.py --n 60
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from models.separator import QueryConditionedSeparator
from utils.musdb_dataset import MUSDB18Dataset
from utils.saraga_dataset import SaragaCarnaticdataset
from utils.audio_utils import si_sdr, STEM_LABELS, SEGMENT_SAMPLES
from training.train_separator import (
    batch_to_mag, mag_to_wav, MUSDB_ROOT, SARAGA_ROOT, CHECKPOINTS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="examples per dataset")
    ap.add_argument("--ckpt", default=str(CHECKPOINTS / "separator_best.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    bc = ckpt.get("config", {}).get("base_channels", 48)
    model = QueryConditionedSeparator(base_channels=bc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt} (epoch {ckpt.get('epoch')}, val_sdr {ckpt.get('val_sdr')})")

    sets = {
        "MUSDB":  MUSDB18Dataset(MUSDB_ROOT, split="valid", mode="separator", augment=False),
        "Saraga": SaragaCarnaticdataset(SARAGA_ROOT, split="valid", mode="separator", augment=False),
    }

    rows = defaultdict(lambda: {"mix": [], "oracle": [], "model": [], "mask": []})

    with torch.no_grad():
        for dname, ds in sets.items():
            idxs = np.random.RandomState(0).permutation(len(ds))[: args.n]
            for i in idxs:
                mix, tgt, s_idx = ds[int(i)]
                if tgt.pow(2).mean().sqrt() < 1e-4:
                    continue
                mix_b = mix.unsqueeze(0)                 # (1,1,T)
                tgt_b = tgt.unsqueeze(0)
                mix_mag, mix_spec = batch_to_mag(mix_b, device)
                tgt_mag, _ = batch_to_mag(tgt_b, device)
                t = tgt.to(device).float().reshape(1, -1)

                mask = model(mix_mag, torch.tensor([s_idx], device=device))
                irm = torch.clamp(tgt_mag / (mix_mag + 1e-8), 0.0, 1.0)

                est_model  = mag_to_wav(mix_mag * mask, mix_spec, SEGMENT_SAMPLES, device)
                est_oracle = mag_to_wav(mix_mag * irm,  mix_spec, SEGMENT_SAMPLES, device)
                est_mix    = mix.to(device).float().reshape(1, -1)

                key = (dname, STEM_LABELS[s_idx])
                rows[key]["mix"].append(float(si_sdr(est_mix, t)))
                rows[key]["oracle"].append(float(si_sdr(est_oracle, t)))
                rows[key]["model"].append(float(si_sdr(est_model, t)))
                rows[key]["mask"].append(float(mask.mean()))

    print(f"\n{'dataset':8s} {'stem':10s} {'n':>3s} | {'mix':>7s} {'oracle':>7s} {'model':>7s} | "
          f"{'model med':>9s} | {'mean mask':>9s}")
    print("-" * 78)
    for (d, s), r in sorted(rows.items()):
        print(f"{d:8s} {s:10s} {len(r['model']):3d} | "
              f"{np.mean(r['mix']):7.2f} {np.mean(r['oracle']):7.2f} {np.mean(r['model']):7.2f} | "
              f"{np.median(r['model']):9.2f} | {np.mean(r['mask']):9.3f}")


if __name__ == "__main__":
    main()
