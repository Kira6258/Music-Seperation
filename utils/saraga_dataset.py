"""
saraga_dataset.py
-----------------
PyTorch Dataset for Saraga 1.5 Carnatic.

Stem vocabulary from this dataset:
  - vocals    (*.multitrack-vocal.mp3)
  - violin    (*.multitrack-violin.mp3)
  - mridangam (left + right channels summed from
                *.multitrack-mridangam-left.mp3 +
                *.multitrack-mridangam-right.mp3)
  - ghatam    (*.multitrack-ghatam.mp3)

Note: Saraga Hindustani has no stem files — only used as mixture context.

Mixture is reconstructed by summing available stem files.

Each __getitem__ returns ONE training example:
  Separator mode: (mixture_segment, target_stem_segment, stem_label_index)
  Detector mode:  (mel_spec, presence_vector, label_mask)
"""

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.audio_utils import (
    load_audio,
    random_segment,
    fixed_segment,
    stem_is_present,
    waveform_to_mel,
    get_mel_transform,
    SEGMENT_SAMPLES,
    STEM_LABELS,
    STEM_TO_IDX,
    NUM_STEMS,
    SAMPLE_RATE,
)


# ─── Saraga Carnatic stem file patterns ──────────────────────────────────────
# Maps our unified stem name → list of file suffixes to load (and sum if >1)
SARAGA_STEM_PATTERNS = {
    "vocals":    ["multitrack-vocal"],
    "violin":    ["multitrack-violin"],
    "mridangam": ["multitrack-mridangam-left", "multitrack-mridangam-right"],
    "ghatam":    ["multitrack-ghatam"],
}
SARAGA_STEMS = list(SARAGA_STEM_PATTERNS.keys())


def _find_stem_file(track_dir: Path, suffix: str) -> Optional[Path]:
    """Find the .mp3 file in track_dir whose name contains `suffix`."""
    files = [f for f in track_dir.iterdir()
             if f.suffix == ".mp3" and "mixture" not in f.name.lower()]
    for f in files:                       # exact match first: "...-vocal.mp3"
        if f.name.endswith(suffix + ".mp3"):
            return f
    for f in files:
        if suffix in f.name:
            return f
    return None


def _load_saraga_stem(track_dir: Path, stem_name: str,
                      max_duration_sec: Optional[float] = 120.0) -> Optional[torch.Tensor]:
    """
    Load a Saraga stem by name. If the stem has two mic channels
    (e.g., mridangam left+right), load both and sum them.
    Capped at max_duration_sec to avoid memory exhaustion on live concerts.
    Returns (1, T) tensor or None if not found.
    """
    patterns = SARAGA_STEM_PATTERNS[stem_name]
    waveforms = []
    for suffix in patterns:
        fpath = _find_stem_file(track_dir, suffix)
        if fpath is not None:
            waveforms.append(load_audio(fpath, max_duration_sec=max_duration_sec))

    if not waveforms:
        return None

    # Align lengths and sum
    min_len = min(w.shape[-1] for w in waveforms)
    waveforms = [w[:, :min_len] for w in waveforms]
    combined = torch.stack(waveforms, dim=0).sum(dim=0)  # (1, T)
    # Normalise after summing
    peak = combined.abs().max()
    if peak > 1.0:
        combined = combined / peak
    return combined


def _load_saraga_mixture(track_dir: Path,
                         max_duration_sec: Optional[float] = 120.0) -> Optional[torch.Tensor]:
    """
    Try to load the main mixture .mp3 (the full concert recording).
    Pattern: the mp3 file that does NOT contain 'multitrack' in its name.
    Capped at max_duration_sec to avoid memory exhaustion on live concerts.
    """
    for f in track_dir.iterdir():
        if f.suffix == ".mp3" and "multitrack" not in f.name.lower():
            return load_audio(f, max_duration_sec=max_duration_sec)
    return None


ENV_RATE_HOP = 220        # 22050/220 ~= 100 Hz envelope
MAX_LAG_SEC = 10.0
MIN_ALIGN_CORR = 0.35     # below this the mixture file is unusable -> rebuild from stems


def _envelope(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool1d(x.abs().reshape(1, 1, -1), ENV_RATE_HOP, ENV_RATE_HOP).reshape(-1)


def _align_mixture(mix: torch.Tensor, ref: torch.Tensor):
    """
    Find the time offset between the released mixture and the sum of the stems
    by cross-correlating 100 Hz loudness envelopes.

    Returns (aligned_mix | None, lag_seconds, correlation). `aligned_mix` is None
    when the correlation is too low to trust (mixture is a different recording).
    """
    n = min(mix.shape[-1], ref.shape[-1])
    if n < SAMPLE_RATE * 5:
        return mix, 0.0, 0.0
    a = _envelope(mix[:, :n]).float()
    b = _envelope(ref[:, :n]).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm() + 1e-8
    L = a.numel()
    nfft = 1 << (2 * L - 1).bit_length()
    cc = torch.fft.irfft(torch.fft.rfft(a, nfft) * torch.fft.rfft(b, nfft).conj(), nfft)
    max_lag = int(MAX_LAG_SEC * 100)
    lags = torch.arange(-max_lag, max_lag + 1)
    vals = cc[lags % nfft]                  # cc[k] = sum_t a[t+k] b[t]
    best = int(vals.argmax())
    lag = int(lags[best])
    corr = float(vals[best] / denom)
    if corr < MIN_ALIGN_CORR:
        return None, lag / 100.0, corr
    shift = lag * ENV_RATE_HOP              # >0: mixture is delayed vs stems
    if shift > 0:
        mix = mix[:, shift:]
    elif shift < 0:
        mix = F.pad(mix, (-shift, 0))
    return mix, lag / 100.0, corr


def _collect_saraga_tracks(root: Path) -> list[Path]:
    """
    Recursively find all leaf track directories that have at least
    one multitrack stem file.
    """
    tracks = []
    for p in root.rglob("*.mp3"):
        if "multitrack" in p.name:
            parent = p.parent
            if parent not in tracks:
                tracks.append(parent)
    return sorted(tracks)


class SaragaCarnaticdataset(Dataset):
    """
    Dataset for Saraga 1.5 Carnatic music.

    Modes:
      'separator' – returns (mixture_segment, target_segment, stem_idx)
      'detector'  – returns (mel_spec, presence_vector)

    Audio is preloaded into RAM at init to avoid repeated MP3 decoding overhead.
    """

    def __init__(self,
                 root: str | Path,
                 split: str = "all",
                 mode: str = "separator",
                 segment_samples: int = SEGMENT_SAMPLES,
                 augment: bool = True,
                 preload: bool = True,
                 max_duration_sec: float = 60.0,
                 neg_prob: float = 0.0,
                 separator_mix: str = "stems"):
        """
        Args:
            root:  Path to saraga/carnatic/saraga1.5_carnatic/
            split: 'all', 'train', or 'valid' (deterministic track-level split)
            mode:  'separator' or 'detector'
            segment_samples: audio clip length in samples
            augment: random gain augmentation
            preload: load all audio into RAM at init (strongly recommended)
            max_duration_sec: max seconds to load per track (default 30s).
                              Prevents loading 70+ minute concerts into RAM.
        """
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.segment_samples = segment_samples
        self.augment = augment
        self.preload = preload
        self.max_duration_sec = max_duration_sec
        # "stems": separator input = exact sum of the stems (consistent with the
        #          targets). "real": the released concert mixture. The released
        #          mix is a processed console mix whose waveform does NOT match
        #          the mic stems (ideal-mask SI-SDR was about -15..-23 dB), so it
        #          cannot be used to learn masks.
        self.separator_mix = separator_mix
        self.neg_prob = neg_prob   # separator: prob. of an absent-query (zero-target) sample
        self.mel_transform = get_mel_transform()

        if not self.root.exists():
            raise FileNotFoundError(f"Saraga root not found: {self.root}")

        all_tracks = _collect_saraga_tracks(self.root)
        if not all_tracks:
            raise RuntimeError(f"No Saraga tracks found under {self.root}")

        # Deterministic track-level split (80% train / 20% val) to eliminate data leakage
        if split in ("train", "val", "valid"):
            import hashlib
            val_tracks = []
            train_tracks = []
            for t in all_tracks:
                h = int(hashlib.md5(t.name.encode("utf-8")).hexdigest(), 16)
                if (h % 100) < 20:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            if split in ("val", "valid"):
                self.tracks = val_tracks if val_tracks else all_tracks[:max(1, len(all_tracks)//5)]
            else:
                self.tracks = train_tracks if train_tracks else all_tracks[max(1, len(all_tracks)//5):]
        else:
            self.tracks = all_tracks

        print(f"[SaragaCarnaticdataset] split={split}: {len(self.tracks)} tracks, mode={mode}")

        # ── In-memory cache: {track_dir: {stem_name: waveform | None}} ───────
        self._cache: dict = {}
        if preload:
            self._preload_audio()

        if mode == "separator":
            # One example per (track, available_stem)
            self.examples = []
            for track in self.tracks:
                for stem in SARAGA_STEMS:
                    wav = self._get_stem(track, stem)
                    if wav is not None and stem_is_present(wav):
                        self.examples.append((track, stem))
        else:
            # For detector mode, include full mixtures + solo stem tracks for robust detection
            self.examples = [(track, "mixture") for track in self.tracks]
            for track in self.tracks:
                for stem in SARAGA_STEMS:
                    wav = self._get_stem(track, stem)
                    if wav is not None and stem_is_present(wav):
                        self.examples.append((track, stem))

    # ── Preloading ────────────────────────────────────────────────────────────

    def _preload_audio(self):
        """Load Saraga tracks (mixture + stems) into RAM once, capped at max_duration_sec."""
        from tqdm import tqdm
        print(f"  Preloading {len(self.tracks)} Saraga tracks into RAM (capped at {self.max_duration_sec:.0f}s each)...")
        stats = []
        for track_dir in tqdm(self.tracks, desc="  Caching Saraga", leave=False):
            entry = {}
            # Mixture
            need_mix = not (self.mode == "separator" and self.separator_mix == "stems")
            mix = (_load_saraga_mixture(track_dir, max_duration_sec=self.max_duration_sec)
                   if need_mix else None)
            # Stems
            for stem in SARAGA_STEMS:
                entry[stem] = _load_saraga_stem(track_dir, stem, max_duration_sec=self.max_duration_sec)

            # The released mixture is NOT sample-aligned with the multitracks.
            # Align it to the stem sum; if it cannot be aligned, drop it so the
            # mixture is rebuilt from the stems (exact by construction).
            parts = [w for w in (entry[s] for s in SARAGA_STEMS) if w is not None]
            if mix is not None and parts:
                m = min(p.shape[-1] for p in parts)
                ref = torch.stack([p[:, :m] for p in parts]).sum(0)
                mix, lag, corr = _align_mixture(mix, ref)
                stats.append((lag, corr, mix is not None))
            entry["mixture"] = mix
            self._cache[track_dir] = entry
        print(f"  Saraga cache ready. ({len(self._cache)} tracks in RAM)")
        if stats:
            kept = sum(1 for _, _, ok in stats if ok)
            lags = sorted(abs(l) for l, _, ok in stats if ok)
            corrs = sorted(c for _, c, _ in stats)
            print(f"  Mixture alignment: {kept}/{len(stats)} usable "
                  f"(rest rebuilt from stems) | median |lag|="
                  f"{lags[len(lags)//2] if lags else 0:.2f}s | "
                  f"median corr={corrs[len(corrs)//2]:.2f}")

    def _get_stem(self, track_dir: Path, stem: str):
        """Get a stem waveform from cache or load from disk."""
        if self._cache and track_dir in self._cache:
            return self._cache[track_dir].get(stem)
        return _load_saraga_stem(track_dir, stem, max_duration_sec=self.max_duration_sec)

    def _get_mixture(self, track_dir: Path):
        """Get the mixture waveform from cache or reconstruct."""
        if self._cache and track_dir in self._cache:
            return self._cache[track_dir].get("mixture")   # None -> caller rebuilds from stems
        return _load_saraga_mixture(track_dir, max_duration_sec=self.max_duration_sec)

    def _reconstruct_mixture(self, track_dir: Path) -> torch.Tensor:
        """Build mixture by summing available stems (used when no mixture file exists)."""
        parts = [self._get_stem(track_dir, s) for s in SARAGA_STEMS]
        parts = [p for p in parts if p is not None]
        if not parts:
            return torch.zeros(1, self.segment_samples)
        min_len = min(p.shape[-1] for p in parts)
        parts = [p[:, :min_len] for p in parts]
        return torch.stack(parts).sum(0)   # no renormalisation: keep stem/mix ratio exact


    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        if self.mode == "separator":
            return self._get_separator_item(idx)
        else:
            return self._get_detector_item(idx)

    # ── Separator mode ──────────────────────────────────────────────────────

    def _get_separator_item(self, idx: int):
        """
        Returns:
            mixture_seg: (1, T)
            target_seg:  (1, T)
            stem_idx:    int
        """
        track_dir, stem_name = self.examples[idx]

        # Load from cache
        target  = self._get_stem(track_dir, stem_name)
        if target is None:
            target = torch.zeros(1, self.segment_samples)

        if self.separator_mix == "stems":
            mixture = self._reconstruct_mixture(track_dir)
        else:
            mixture = self._get_mixture(track_dir)
            if mixture is None:
                mixture = self._reconstruct_mixture(track_dir)

        # BUG FIX: align BOTH to the shorter length BEFORE choosing the random
        # crop offset. Previously the mixture could be 60 min while target was
        # only 30 s, so a random offset in mixture would sample completely
        # different music from the target — making IRM = random noise.
        min_len = min(mixture.shape[-1], target.shape[-1])
        mixture = mixture[:, :min_len]
        target  = target[:, :min_len]

        # Random crop — offset is now guaranteed valid for both
        start = random.randint(0, max(0, min_len - self.segment_samples))
        end   = start + self.segment_samples
        if end > min_len:
            pad = end - min_len
            mixture = F.pad(mixture, (0, pad))
            target  = F.pad(target,  (0, pad))

        mixture_seg = mixture[:, start:end]
        target_seg  = target[:, start:end]

        # Absent-query negative: ask for a stem that is not in this track
        # (guitar/piano/drums/bass, or a Carnatic stem without a file).
        if self.neg_prob > 0 and random.random() < self.neg_prob:
            present = {s for s in SARAGA_STEMS
                       if self._get_stem(track_dir, s) is not None}
            absent = [s for s in STEM_LABELS if s not in present]
            if absent:
                stem_name = random.choice(absent)
                target_seg = torch.zeros_like(mixture_seg)

        if self.augment:
            gain = 10 ** (random.uniform(-6, 6) / 20.0)
            mixture_seg = mixture_seg * gain
            target_seg  = target_seg  * gain

        stem_idx = STEM_TO_IDX[stem_name]
        return mixture_seg, target_seg, stem_idx

    # ── Detector mode ───────────────────────────────────────────────────────

    def get_stem_clip(self, stem_name: str):
        """Random segment of a clean stem from a random track (synthetic mixing)."""
        if stem_name not in SARAGA_STEMS:
            return None
        for _ in range(10):
            track = random.choice(self.tracks)
            wav = self._get_stem(track, stem_name)
            if wav is not None and stem_is_present(wav):
                return random_segment(wav, self.segment_samples)
        return None

    def _get_detector_item(self, idx: int):
        """
        Returns:
            mel_spec:     (1, N_MELS, T)
            presence_vec: (NUM_STEMS,)  binary label per stem
            label_mask:   (NUM_STEMS,)  all ones (every label is known here)

        Presence is computed on the SAME crop that is fed to the network.
        """
        track_dir, stem_target = self.examples[idx]

        presence   = torch.zeros(NUM_STEMS, dtype=torch.float32)
        label_mask = torch.ones(NUM_STEMS, dtype=torch.float32)

        if stem_target == "mixture":
            wav = self._get_mixture(track_dir)
            if wav is None:
                wav = self._reconstruct_mixture(track_dir)
        else:
            wav = self._get_stem(track_dir, stem_target)
            if wav is None:
                wav = torch.zeros(1, self.segment_samples)

        T = wav.shape[-1]
        start = random.randint(0, max(0, T - self.segment_samples))
        wav_seg = fixed_segment(wav, self.segment_samples, start)

        if stem_target == "mixture":
            for s in SARAGA_STEMS:
                w = self._get_stem(track_dir, s)
                if w is not None and stem_is_present(
                        fixed_segment(w, self.segment_samples, start)):
                    presence[STEM_TO_IDX[s]] = 1.0
        else:
            presence[STEM_TO_IDX[stem_target]] = 1.0

        mel = waveform_to_mel(wav_seg, self.mel_transform)
        return mel, presence, label_mask
