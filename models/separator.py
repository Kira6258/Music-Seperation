"""
separator.py
------------
Query-Conditioned U-Net with FiLM Conditioning for music source separation.

Architecture:
  - Encoder: 4-level U-Net that processes the STFT magnitude of the mixture
  - FiLM conditioning: at each encoder level, a learned (γ, β) is derived
    from the stem query embedding and applied via affine feature modulation
  - Decoder: symmetric 4-level upsampling with skip connections
  - Output: estimated magnitude mask → reconstructed waveform

Query type: integer stem index → learned embedding (simple & trainable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.audio_utils import NUM_STEMS, N_FFT

INPUT_LOG_SCALE = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# FiLM Conditioning
# ─────────────────────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Given a conditioning vector, produces channel-wise scale (γ) and shift (β)
    applied to feature maps: output = γ * feature + β
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, num_channels * 2)  # → [γ, β]

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, H, W)
        cond: (B, cond_dim)
        """
        gamma_beta = self.fc(cond)            # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder blocks
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Conv → BN → LeakyReLU → stride-2 (downsampling)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """ConvTranspose2d → BN → ReLU → optional Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Query-Conditioned U-Net Separator
# ─────────────────────────────────────────────────────────────────────────────

class QueryConditionedSeparator(nn.Module):
    """
    U-Net with FiLM conditioning for query-conditioned music source separation.

    Input:
        mixture_mag: (B, 1, F, T)  — STFT magnitude spectrogram
        stem_idx:    (B,)          — integer stem index (the query)

    Output:
        mask:        (B, 1, F, T)  — multiplicative mask in [0,1]

    The separated magnitude is: mixture_mag * mask
    Phase from the original mixture is reused for waveform reconstruction.
    """

    def __init__(self,
                 num_stems: int = NUM_STEMS,
                 embed_dim: int = 64,
                 base_channels: int = 48):
        super().__init__()
        self.num_stems = num_stems
        self.embed_dim = embed_dim

        # ── Query Embedding ────────────────────────────────────────────────
        self.stem_embedding = nn.Embedding(num_stems, embed_dim)

        # ── Encoder ────────────────────────────────────────────────────────
        # Input: (B, 1, F, T)
        ch = base_channels
        self.enc1 = EncoderBlock(1,      ch)      # → (B, 32,  F/2, T/2)
        self.enc2 = EncoderBlock(ch,     ch*2)    # → (B, 64,  F/4, T/4)
        self.enc3 = EncoderBlock(ch*2,   ch*4)    # → (B, 128, F/8, T/8)
        self.enc4 = EncoderBlock(ch*4,   ch*8)    # → (B, 256, F/16,T/16)

        # ── FiLM layers (one per encoder level) ───────────────────────────
        self.film1 = FiLM(embed_dim, ch)
        self.film2 = FiLM(embed_dim, ch*2)
        self.film3 = FiLM(embed_dim, ch*4)
        self.film4 = FiLM(embed_dim, ch*8)

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch*8, ch*8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch*8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Decoder (with skip connections, so in_ch doubles) ─────────────
        self.dec4 = DecoderBlock(ch*8,      ch*4, dropout=0.2)
        self.dec3 = DecoderBlock(ch*4 + ch*4, ch*2, dropout=0.1)
        self.dec2 = DecoderBlock(ch*2 + ch*2, ch)
        self.dec1 = DecoderBlock(ch + ch,     ch)

        # ── Output: sigmoid mask ───────────────────────────────────────────
        self.output_conv = nn.Sequential(
            nn.Conv2d(ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self,
                mixture_mag: torch.Tensor,
                stem_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture_mag: (B, 1, F, T) — magnitude spectrogram of mixture
            stem_idx:    (B,)         — integer indices into STEM_LABELS
        Returns:
            mask:        (B, 1, F, T) — soft mask ∈ [0, 1]
        """
        # Query embedding
        cond = self.stem_embedding(stem_idx)  # (B, embed_dim)

        # Log-compress the raw magnitude before the encoder. Raw STFT magnitudes
        # span ~5 orders of magnitude, so quiet bins (where hiss/bleed lives) were
        # invisible to the network. The mask is still applied to the RAW magnitude.
        x = torch.log1p(mixture_mag * INPUT_LOG_SCALE)

        # Encoder with FiLM
        e1 = self.film1(self.enc1(x), cond)
        e2 = self.film2(self.enc2(e1), cond)
        e3 = self.film3(self.enc3(e2), cond)
        e4 = self.film4(self.enc4(e3), cond)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = self._pad_and_cat(d4, e3)
        d3 = self.dec3(d4)
        d3 = self._pad_and_cat(d3, e2)
        d2 = self.dec2(d3)
        d2 = self._pad_and_cat(d2, e1)
        d1 = self.dec1(d2)

        # Resize to match input if needed (due to strided convs on odd dims)
        d1 = F.interpolate(d1, size=mixture_mag.shape[2:], mode="bilinear",
                           align_corners=False)

        return self.output_conv(d1)

    @staticmethod
    def _pad_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Pad x to match skip's spatial dimensions, then concatenate on channel dim.
        Handles potential size mismatches from strided convolutions.
        """
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([x, skip], dim=1)

    @torch.no_grad()
    def separate(self,
                 mixture_mag: torch.Tensor,
                 stem_idx: torch.Tensor) -> torch.Tensor:
        """Inference helper: returns the masked magnitude."""
        self.eval()
        mask = self.forward(mixture_mag, stem_idx)
        return mixture_mag * mask
