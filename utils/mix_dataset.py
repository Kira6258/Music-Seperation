"""
mix_dataset.py
--------------
Synthetic-mixture wrapper for training the stem detector.

Why: with only real MUSDB / Saraga mixtures the detector can solve the task by
recognising the *dataset* (MP3 concert recording => violin/mridangam, studio
WAV => drums/bass) instead of the instruments. Here clean stems from BOTH
corpora are randomly summed, so e.g. a MUSDB vocal can sit on top of a Saraga
mridangam. Labels are exact because we know precisely which stems were added.

guitar/piano have no clean source in either corpus, so their labels are
masked out of the loss (label_mask = 0) in every sample.
"""

import random

import torch
from torch.utils.data import Dataset

from utils.audio_utils import (
    STEM_TO_IDX, NUM_STEMS, SEGMENT_SAMPLES,
    waveform_to_mel, get_mel_transform,
)

MIXABLE_STEMS = ["vocals", "drums", "bass", "violin", "mridangam", "ghatam"]
UNTRAINED_STEMS = ["guitar", "piano"]


class SyntheticMixDetectorDataset(Dataset):
    """
    Args:
        real_dataset:  a (Concat)Dataset yielding (mel, presence, label_mask)
        stem_sources:  datasets exposing get_stem_clip(stem_name) -> (1, T) | None
        length:        epoch length (defaults to len(real_dataset))
        p_synthetic:   probability of returning a synthetic mix instead of a real item
    """

    def __init__(self, real_dataset, stem_sources, length=None, p_synthetic=0.6):
        self.real = real_dataset
        self.sources = list(stem_sources)
        self.length = length or len(real_dataset)
        self.p_synthetic = p_synthetic
        self.mel_transform = get_mel_transform()

    def __len__(self):
        return self.length

    def _clip(self, stem):
        srcs = self.sources[:]
        random.shuffle(srcs)
        for src in srcs:
            clip = src.get_stem_clip(stem)
            if clip is not None:
                return clip
        return None

    def _synthetic(self):
        k = random.randint(1, 4)
        chosen = random.sample(MIXABLE_STEMS, k)
        mix = torch.zeros(1, SEGMENT_SAMPLES)
        presence = torch.zeros(NUM_STEMS, dtype=torch.float32)

        for stem in chosen:
            clip = self._clip(stem)
            if clip is None:
                continue
            gain = 10 ** (random.uniform(-8, 3) / 20.0)
            mix = mix + clip[:, :SEGMENT_SAMPLES] * gain
            presence[STEM_TO_IDX[stem]] = 1.0

        if presence.sum() == 0:           # nothing could be loaded -> fall back
            return None

        peak = mix.abs().max()
        if peak > 0.95:
            mix = mix / peak * 0.92

        label_mask = torch.ones(NUM_STEMS, dtype=torch.float32)
        for s in UNTRAINED_STEMS:
            label_mask[STEM_TO_IDX[s]] = 0.0

        mel = waveform_to_mel(mix, self.mel_transform)
        return mel, presence, label_mask

    def __getitem__(self, idx):
        if random.random() < self.p_synthetic:
            item = self._synthetic()
            if item is not None:
                return item
        return self.real[idx % len(self.real)]
