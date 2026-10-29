"""
absent_query_eval.py
--------------------
Measures what the final system returns when asked for a stem that is not in the
song (violin, mridangam, ghatam on Western test tracks), per track and per stem.

Level is the RMS of the returned audio relative to the RMS of the input mixture,
in dB (more negative = quieter).  Vocals (present) is included as a reference.
Run: python output/absent_query_eval.py
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchaudio

from inference.pipeline import MusicSeparationPipeline
from utils.audio_utils import load_audio

STEMS = ["violin", "mridangam", "ghatam", "vocals"]
N_TRACKS, SECONDS, OFFSET = 4, 30, 60


def rms_db(x: torch.Tensor) -> float:
    return float(20 * torch.log10(x.pow(2).mean().sqrt() + 1e-12))


def main():
    pipe = MusicSeparationPipeline()
    test = ROOT / "datasets" / "musdb18hq" / "test"
    tracks = sorted(p for p in test.iterdir() if p.is_dir())[:N_TRACKS]
    rows = {}
    for td in tracks:
        mix = load_audio(str(td / "mixture.wav"), offset_sec=OFFSET, max_duration_sec=SECONDS)
        tmp = Path(tempfile.gettempdir()) / "absent_in.wav"
        torchaudio.save(str(tmp), mix.reshape(1, -1), 22050)
        mix_db = rms_db(mix)
        rows[td.name] = {}
        for st in STEMS:
            out = pipe.separate(str(tmp), [st])
            wav, _ = torchaudio.load(out)
            rows[td.name][st] = round(rms_db(wav) - mix_db, 1)
        print(td.name, rows[td.name], flush=True)
    summary = {}
    for st in STEMS:
        v = np.array([rows[t][st] for t in rows])
        summary[st] = {"mean": round(float(v.mean()), 1), "sd": round(float(v.std(ddof=1)), 1),
                       "min": round(float(v.min()), 1), "max": round(float(v.max()), 1)}
    print(json.dumps(summary, indent=1))
    out = Path(__file__).resolve().parent / "absent_query_results.json"
    out.write_text(json.dumps({"per_track": rows, "summary": summary}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
