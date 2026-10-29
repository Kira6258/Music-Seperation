"""
stem_detector.py
----------------
Multi-label CNN classifier that detects which named stems are present
in a music mixture.

Input:  Log-mel spectrogram  (B, 1, N_MELS=128, T_frames)
Output: Per-stem sigmoid     (B, NUM_STEMS=8)

Stems: [vocals, drums, bass, guitar, piano, violin, mridangam, ghatam]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.audio_utils import NUM_STEMS, N_MELS


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → optional MaxPool."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Stem Detector
# ─────────────────────────────────────────────────────────────────────────────

class StemDetector(nn.Module):
    """
    Lightweight CNN for multi-label stem detection.

    Architecture:
      4 × ConvBlock (with max-pooling)  →  Global Average Pooling  →  FC → sigmoid

    Designed to run fast at inference time (stem detection is Stage 1).
    """

    def __init__(self,
                 n_mels: int = N_MELS,
                 num_stems: int = NUM_STEMS,
                 base_channels: int = 32):
        super().__init__()
        self.num_stems = num_stems

        self.encoder = nn.Sequential(
            ConvBlock(1,               base_channels,     pool=True),   # 128→64
            ConvBlock(base_channels,   base_channels * 2, pool=True),   # 64→32
            ConvBlock(base_channels*2, base_channels * 4, pool=True),   # 32→16
            ConvBlock(base_channels*4, base_channels * 8, pool=True),   # 16→8
        )

        # Global average pool collapses (H, W) → scalar per channel
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_stems),
            # No activation here — BCEWithLogitsLoss expects raw logits
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 1, N_MELS, T)  — log-mel spectrogram
        Returns:
            logits: (B, num_stems)  — raw logits (pass through sigmoid for probs)
        """
        x = self.encoder(mel)   # (B, 256, H', W')
        x = self.gap(x)         # (B, 256, 1, 1)
        return self.classifier(x)   # (B, num_stems)

    @torch.no_grad()
    def predict(self, mel: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        """
        Inference convenience method.
        Returns binary presence mask (B, num_stems).
        """
        self.eval()
        logits = self.forward(mel)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()

    @torch.no_grad()
    def predict_probs(self, mel: torch.Tensor) -> torch.Tensor:
        """Returns raw probabilities (B, num_stems) — no thresholding."""
        self.eval()
        return torch.sigmoid(self.forward(mel))


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class StemDetectorLoss(nn.Module):
    """
    Weighted, label-masked Binary Cross-Entropy with logits.

    label_mask (B, num_stems): 1 = label is known, 0 = ignore. Needed because
    MUSDB has no separate guitar/piano ground truth, so those two labels must
    not contribute gradient on MUSDB audio.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        if pos_weight is None:
            # [vocals, drums, bass, guitar, piano, violin, mridangam, ghatam]
            weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0]
            pos_weight = torch.tensor(weights[:NUM_STEMS], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                label_mask: torch.Tensor = None) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        if label_mask is None:
            return loss.mean()
        loss = loss * label_mask
        return loss.sum() / label_mask.sum().clamp(min=1.0)
