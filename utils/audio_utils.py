"""
audio_utils.py
--------------
Shared audio helpers:
  - Mel-spectrogram computation
  - SI-SDR (Scale-Invariant SDR) loss
  - Audio loading / saving (handles both .wav and .mp3)
  - Stem mixing (sum multiple stems into one combined waveform)
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 22050        # unified sample rate across both datasets
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
SEGMENT_SECONDS = 6        # audio clip length for training
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

# Unified stem vocabulary (8 stems: MUSDB18-HQ + Saraga Carnatic)
STEM_LABELS = [
    "vocals",        # 0  – Voice (MUSDB18 / Saraga)
    "drums",         # 1  – Drums / Kit (MUSDB18)
    "bass",          # 2  – Bass (electric / acoustic, MUSDB18)
    "guitar",        # 3  – Guitar (acoustic / electric, MUSDB18)
    "piano",         # 4  – Piano / Keys (MUSDB18)
    "violin",        # 5  – Carnatic / Western Violin (Saraga)
    "mridangam",     # 6  – Mridangam (Saraga Carnatic)
    "ghatam",        # 7  – Ghatam (Saraga Carnatic)
]
NUM_STEMS = len(STEM_LABELS)
STEM_TO_IDX = {s: i for i, s in enumerate(STEM_LABELS)}
# Stem family groups for grouped selection (e.g. "all_percussion" → drums + mridangam + ghatam)
STEM_FAMILIES = {
    "percussion":  ["drums", "mridangam", "ghatam"],
    "melodic":     ["guitar", "piano", "violin"],
    "rhythm":      ["drums", "bass", "mridangam", "ghatam"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE,
               mono: bool = True,
               offset_sec: float = 0.0,
               max_duration_sec: Optional[float] = None) -> torch.Tensor:
    """
    Load an audio file (wav or mp3) and resample to target_sr.
    Supports partial loading via offset_sec and max_duration_sec to avoid
    allocating gigabytes of RAM on multi-hour live recordings.
    Returns: waveform tensor of shape (1, T) if mono=True, else (C, T).
    """
    path = str(path)
    frame_offset = 0
    num_frames = -1

    if offset_sec > 0.0 or max_duration_sec is not None:
        try:
            info = torchaudio.info(path)
            orig_sr = info.sample_rate
            if offset_sec > 0.0:
                frame_offset = int(offset_sec * orig_sr)
            if max_duration_sec is not None:
                num_frames = int(max_duration_sec * orig_sr)
        except Exception:
            frame_offset = 0
            num_frames = -1

    waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)

    # If seeking beyond file produced empty tensor, fallback to beginning
    if waveform.shape[-1] == 0 and frame_offset > 0:
        waveform, sr = torchaudio.load(path, frame_offset=0, num_frames=num_frames)

    # Convert to mono by averaging channels
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform  # shape: (1, T)


def save_audio(waveform: torch.Tensor, path: str | Path,
               sr: int = SAMPLE_RATE) -> None:
    """
    Save a waveform tensor to a .wav file.
    waveform: (1, T) or (T,)
    Normalizes to prevent clipping. Applies gentle gain (max 1.8x) to
    audible stems. Does NOT boost signals below 0.08 peak (silence/bleed).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.cpu().float()

    peak = waveform.abs().max()
    if peak > 0.95:
        # Prevent digital clipping distortion
        waveform = (waveform / peak) * 0.92
    elif peak >= 0.08:
        # Audible stem: apply gentle gain (max 1.8x) so it's comfortably audible
        gain = min(1.8, 0.85 / (peak + 1e-6))
        waveform = waveform * gain
    # If peak < 0.08, it's silence / residual bleed — DO NOT amplify noise

    torchaudio.save(path, waveform, sample_rate=sr)


def mix_stems(stem_waveforms: list[torch.Tensor]) -> torch.Tensor:
    """
    Sum a list of stem waveforms (all same shape) into one combined waveform.
    Only normalizes to prevent clipping. Never boosts quiet signals.
    """
    if not stem_waveforms:
        raise ValueError("No stems provided to mix.")
    mixed = torch.stack(stem_waveforms, dim=0).sum(dim=0)  # (1, T)
    peak = mixed.abs().max()
    if peak > 0.95:
        mixed = (mixed / peak) * 0.92
    return mixed


# ─────────────────────────────────────────────────────────────────────────────
# Mel Spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def get_mel_transform(sr: int = SAMPLE_RATE,
                      n_fft: int = N_FFT,
                      hop_length: int = HOP_LENGTH,
                      n_mels: int = N_MELS) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )


def waveform_to_mel(waveform: torch.Tensor,
                    mel_transform: T.MelSpectrogram = None,
                    log_scale: bool = True) -> torch.Tensor:
    """
    Convert waveform (1, T) to log-mel spectrogram (1, N_MELS, frames).
    """
    if mel_transform is None:
        mel_transform = get_mel_transform()
    mel = mel_transform(waveform)              # (1, N_MELS, frames)
    if log_scale:
        mel = torch.log1p(mel * 1e4)           # compress dynamic range
    return mel


def compute_rms_energy(waveform: torch.Tensor) -> float:
    """
    Compute the RMS energy of a waveform tensor.
    Used to determine if a stem is 'present' in a track.
    """
    return float(waveform.pow(2).mean().sqrt())


def stem_is_present(waveform: torch.Tensor,
                    threshold: float = 0.005) -> bool:
    """
    Returns True if the stem waveform has meaningful energy above threshold.
    """
    return compute_rms_energy(waveform) > threshold


# ─────────────────────────────────────────────────────────────────────────────
# STFT for the Separator (magnitude + phase)
# ─────────────────────────────────────────────────────────────────────────────

def stft(waveform: torch.Tensor,
         n_fft: int = N_FFT,
         hop_length: int = HOP_LENGTH) -> torch.Tensor:
    """
    Compute STFT and return complex tensor of shape (..., freq, time).
    """
    # waveform: (1, T) → (T,)
    w = waveform.squeeze(0)
    window = torch.hann_window(n_fft, device=waveform.device)
    return torch.stft(w, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True)


def istft(spec: torch.Tensor,
          n_fft: int = N_FFT,
          hop_length: int = HOP_LENGTH,
          length: int = None) -> torch.Tensor:
    """
    Inverse STFT from complex tensor (freq, time) → waveform (1, T).
    """
    window = torch.hann_window(n_fft, device=spec.device)
    kwargs = dict(n_fft=n_fft, hop_length=hop_length,
                  window=window)
    if length is not None:
        kwargs["length"] = length
    wav = torch.istft(spec, **kwargs)
    return wav.unsqueeze(0)  # (1, T)


# ─────────────────────────────────────────────────────────────────────────────
# SI-SDR Loss
# ─────────────────────────────────────────────────────────────────────────────

def si_sdr(estimate: torch.Tensor, target: torch.Tensor,
           eps: float = 1e-8, reduction: str = "mean") -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    Higher is better. Both tensors should be (B, T) or (T,).

    Returns SI-SDR in dB: scalar mean over batch (reduction="mean") or the
    per-sample vector (reduction="none").
    Always computed in float32 to prevent float16 overflow over long waveform lengths.
    """
    estimate = estimate.float()
    target = target.float()

    if estimate.dim() == 3:   # (B, 1, T) → (B, T)
        estimate = estimate.squeeze(1)
        target = target.squeeze(1)

    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Projection of estimate onto target
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target * target).sum(dim=-1, keepdim=True) + eps
    scale = dot / target_energy
    projection = scale * target

    noise = estimate - projection
    proj_energy = projection.pow(2).sum(dim=-1) + eps
    noise_energy = noise.pow(2).sum(dim=-1) + eps
    ratio = torch.clamp(proj_energy / noise_energy, min=1e-7, max=1e7)
    si_sdr_val = 10 * torch.log10(ratio)
    si_sdr_val = torch.nan_to_num(si_sdr_val, nan=-30.0, posinf=30.0, neginf=-30.0)
    if reduction == "none":
        return si_sdr_val
    return si_sdr_val.mean()


def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative SI-SDR for use as a minimisation loss."""
    return -si_sdr(estimate, target)


def spectrogram_l1_loss(est_mag: torch.Tensor, tgt_mag: torch.Tensor) -> torch.Tensor:
    """L1 loss between estimated and target magnitude spectrograms."""
    return F.l1_loss(est_mag, tgt_mag)


# ─────────────────────────────────────────────────────────────────────────────
# Segment helpers
# ─────────────────────────────────────────────────────────────────────────────

def random_segment(waveform: torch.Tensor,
                   segment_len: int = SEGMENT_SAMPLES) -> torch.Tensor:
    """
    Randomly crop a fixed-length segment from a waveform.
    If the waveform is shorter, pad with zeros.
    waveform: (1, T)
    Returns: (1, segment_len)
    """
    T = waveform.shape[-1]
    if T < segment_len:
        pad = segment_len - T
        waveform = F.pad(waveform, (0, pad))
    else:
        start = torch.randint(0, T - segment_len + 1, (1,)).item()
        waveform = waveform[:, start: start + segment_len]
    return waveform


def fixed_segment(waveform: torch.Tensor,
                  segment_len: int = SEGMENT_SAMPLES,
                  start: int = 0) -> torch.Tensor:
    """Extract a fixed segment starting at `start`."""
    T = waveform.shape[-1]
    end = start + segment_len
    if end > T:
        waveform = F.pad(waveform, (0, end - T))
    return waveform[:, start:end]
