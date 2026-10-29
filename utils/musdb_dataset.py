"""
musdb_dataset.py
----------------
PyTorch Dataset for MUSDB18HQ.

Stem vocabulary from this dataset:
  - vocals  (vocals.wav)
  - drums   (drums.wav)
  - bass    (bass.wav)
  - other   (other.wav) -> an undifferentiated mix of guitar/piano/synths.
            Used as the separator target for BOTH "guitar" and "piano", but its
            guitar/piano presence is UNKNOWN for the detector (label-masked).

Performance:
  Audio is loaded into RAM once per track (during dataset init with preload=True)
  and reused across all batches — eliminating repeated disk I/O which would
  otherwise make each batch take several minutes.

Separator mode item : (mixture_segment, target_segment, stem_idx)
Detector mode item  : (mel_spec, presence_vector, label_mask)
"""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.audio_utils import (
    load_audio,
    random_segment,
    fixed_segment,
    stem_is_present,
    waveform_to_mel,
    get_mel_transform,
    SEGMENT_SAMPLES,
    STEM_TO_IDX,
    NUM_STEMS,
)

# MUSDB18HQ stem mappings for specific instrument queries
MUSDB_STEM_MAPPING = {
    "vocals": "vocals.wav",
    "drums":  "drums.wav",
    "bass":   "bass.wav",
    "guitar": "other.wav",  # other.wav predominantly contains guitar/accompaniment in rock/pop
    "piano":  "other.wav",  # other.wav contains keyboard/polyphonic accompaniment in pop/rock
}
MUSDB_STEMS = ["vocals", "drums", "bass", "guitar", "piano"]

# Stems that can never appear in MUSDB18 -> valid "absent query" negatives.
MUSDB_ABSENT_STEMS = ["violin", "mridangam", "ghatam"]
# MUSDB has only an undifferentiated "other" stem, so guitar/piano presence is
# UNKNOWN there. The detector loss must ignore those two labels on MUSDB audio.
UNKNOWN_ON_MUSDB = ["guitar", "piano"]
# Stems with a clean, individually-labelled file (used for synthetic mixing).
MUSDB_CLEAN_STEMS = ["vocals", "drums", "bass"]

# Seconds of each track kept in RAM (was 60). Audio is cached as float16.
MAX_SECONDS = 120.0


class MUSDB18Dataset(Dataset):
    """
    Dataset for MUSDB18HQ.

    Modes:
      'separator'  – returns (mixture_segment, target_segment, stem_idx)
      'detector'   – returns (mel_spec, presence_vector, label_mask)
    """

    def __init__(self,
                 root: str | Path,
                 split: str = "train",
                 mode: str = "separator",
                 segment_samples: int = SEGMENT_SAMPLES,
                 augment: bool = True,
                 preload: bool = True,
                 neg_prob: float = 0.0,
                 max_seconds: float = None):
        """
        Args:
            root:    Path to musdb18hq/ folder.
            split:   'train', 'valid', or 'test'.
            mode:    'separator' or 'detector'.
            segment_samples: number of audio samples per training clip.
            augment: if True, apply random gain augmentation.
            preload: load all audio into RAM at init time (strongly recommended).
            neg_prob: (separator mode) probability of returning an "absent
                     query" example: a stem that is not in the mixture, with an
                     all-zero target, so the model learns to output silence
                     instead of noise when asked for something that isn't there.
        """
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.segment_samples = segment_samples
        self.augment = augment
        self.preload = preload
        self.neg_prob = neg_prob
        self.max_seconds = max_seconds or MAX_SECONDS
        self.mel_transform = get_mel_transform()

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"MUSDB18HQ split directory not found: {split_dir}")

        # The local `valid` folder is a COPY of 16 `train` tracks. Drop them from
        # `train` so validation is not measured on training data (leakage).
        valid_names = set()
        if split == "train" and (self.root / "valid").exists():
            valid_names = {p.name for p in (self.root / "valid").iterdir() if p.is_dir()}

        self.tracks = []
        for track_dir in sorted(split_dir.iterdir()):
            if not track_dir.is_dir() or track_dir.name in valid_names:
                continue
            if (track_dir / "mixture.wav").exists() and (track_dir / "vocals.wav").exists():
                self.tracks.append(track_dir)

        if not self.tracks:
            raise RuntimeError(f"No valid MUSDB18HQ tracks found in {split_dir}")

        print(f"[MUSDB18Dataset] {split}: {len(self.tracks)} tracks, mode={mode}")

        # ── In-memory audio cache ──────────────────────────────────────────
        self._cache: dict = {}
        if preload:
            self._preload_audio()

        if mode == "separator":
            self.examples = [
                (track, stem)
                for track in self.tracks
                for stem in MUSDB_STEMS
            ]
        else:
            # Detector: mixtures + solo stems (+ raw "other")
            self.examples = [(track, "mixture") for track in self.tracks]
            for track in self.tracks:
                for stem in MUSDB_CLEAN_STEMS + ["other"]:
                    wav = self._get_audio(track, stem)
                    if stem_is_present(wav):
                        self.examples.append((track, stem))

    # ── Preloading ──────────────────────────────────────────────────────────

    def _preload_audio(self):
        """Load all track audio into RAM once (first 60s of each track)."""
        from tqdm import tqdm
        print(f"  Preloading {len(self.tracks)} tracks into RAM (fast cache)...")
        for track_dir in tqdm(self.tracks, desc="  Caching", leave=False):
            entry = {}
            for fname in ["mixture.wav", "vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
                fpath = track_dir / fname
                if fpath.exists():
                    entry[fname] = load_audio(fpath, max_duration_sec=self.max_seconds).half()
            self._cache[track_dir] = entry
        print(f"  Cache ready. ({len(self._cache)} tracks in RAM)")

    def _get_audio(self, track_dir: Path, stem_name: str) -> torch.Tensor:
        """Get waveform from cache (fast) or disk (fallback)."""
        filename = MUSDB_STEM_MAPPING.get(stem_name, f"{stem_name}.wav")
        if self._cache and track_dir in self._cache:
            if filename in self._cache[track_dir]:
                return self._cache[track_dir][filename].float()
        return load_audio(track_dir / filename, max_duration_sec=self.max_seconds)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        if self.mode == "separator":
            return self._get_separator_item(idx)
        return self._get_detector_item(idx)

    # ── Separator mode ──────────────────────────────────────────────────────

    def _get_separator_item(self, idx: int):
        track_dir, stem_name = self.examples[idx]

        mixture = self._get_audio(track_dir, "mixture")
        target  = self._get_audio(track_dir, stem_name)

        min_len = min(mixture.shape[-1], target.shape[-1])
        mixture = mixture[:, :min_len]
        target  = target[:, :min_len]

        start = random.randint(0, max(0, min_len - self.segment_samples))
        end   = start + self.segment_samples
        if end > min_len:
            pad = end - min_len
            mixture = torch.nn.functional.pad(mixture, (0, pad))
            target  = torch.nn.functional.pad(target,  (0, pad))

        mixture_seg = mixture[:, start:end]
        target_seg  = target[:, start:end]

        # Absent-query negative: ask for a stem that is not in the mixture.
        if self.neg_prob > 0 and random.random() < self.neg_prob:
            stem_name = random.choice(MUSDB_ABSENT_STEMS)
            target_seg = torch.zeros_like(mixture_seg)

        if self.augment:
            gain = 10 ** (random.uniform(-6, 6) / 20.0)
            mixture_seg = mixture_seg * gain
            target_seg  = target_seg  * gain

        return mixture_seg, target_seg, STEM_TO_IDX[stem_name]

    # ── Detector mode ───────────────────────────────────────────────────────

    def get_stem_clip(self, stem_name: str):
        """Random segment of a clean stem from a random track (synthetic mixing)."""
        if stem_name not in MUSDB_CLEAN_STEMS:
            return None
        track = random.choice(self.tracks)
        return random_segment(self._get_audio(track, stem_name), self.segment_samples)

    def _get_detector_item(self, idx: int):
        """
        Returns:
            mel_spec:     (1, N_MELS, T)
            presence_vec: (NUM_STEMS,)  binary label per stem
            label_mask:   (NUM_STEMS,)  1 = label known, 0 = ignore in loss

        Presence is computed on the SAME crop that is fed to the network. (The
        old code labelled a random 6 s crop using the whole 60 s clip, so a
        stem silent in the crop was still labelled present.)
        """
        track_dir, stem_target = self.examples[idx]

        presence   = torch.zeros(NUM_STEMS, dtype=torch.float32)
        label_mask = torch.ones(NUM_STEMS, dtype=torch.float32)

        wav_full = self._get_audio(track_dir, stem_target)
        T = wav_full.shape[-1]
        start = random.randint(0, max(0, T - self.segment_samples))
        wav_seg = fixed_segment(wav_full, self.segment_samples, start)

        if stem_target == "mixture":
            for stem_name in MUSDB_CLEAN_STEMS:
                crop = fixed_segment(self._get_audio(track_dir, stem_name),
                                     self.segment_samples, start)
                if stem_is_present(crop):
                    presence[STEM_TO_IDX[stem_name]] = 1.0
            for s_name in UNKNOWN_ON_MUSDB:
                label_mask[STEM_TO_IDX[s_name]] = 0.0
        elif stem_target == "other":
            for s_name in UNKNOWN_ON_MUSDB:
                label_mask[STEM_TO_IDX[s_name]] = 0.0
        else:
            presence[STEM_TO_IDX[stem_target]] = 1.0

        mel = waveform_to_mel(wav_seg, self.mel_transform)
        return mel, presence, label_mask
