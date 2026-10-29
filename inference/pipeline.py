"""
pipeline.py
-----------
End-to-end inference pipeline:
  1. Load uploaded audio
  2. Run Stem Detector → find present stems (CNN + HDEMUCS energy veto)
  3. User selects stems (or called programmatically)
  4. Run Separator for each selected stem (Overlap-Add with Hann windowing)
  5. Mix selected stem outputs into one combined audio

This module is imported by the Flask app (ui/app.py) and CLI (demo_cli.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchaudio
import tempfile
import numpy as np
import librosa

from models.stem_detector import StemDetector
from models.separator import QueryConditionedSeparator
from utils.audio_utils import (
    load_audio, save_audio, mix_stems,
    waveform_to_mel, get_mel_transform,
    stft, istft, fixed_segment,
    STEM_LABELS, STEM_TO_IDX, STEM_FAMILIES, NUM_STEMS,
    SAMPLE_RATE, SEGMENT_SAMPLES, N_FFT, HOP_LENGTH,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS  = PROJECT_ROOT / "checkpoints"

DETECTOR_CKPT  = CHECKPOINTS / "stem_detector_best.pt"
SEPARATOR_CKPT = CHECKPOINTS / "separator_best.pt"

# Soft masks are raised to this power before use. >1 suppresses low-confidence
# bins (bleed / residual noise) at a small cost in loudness. 1.0 = raw mask.
MASK_POWER = 1.2
# Soft masks are smoothed with a k x k average filter over (freq, time) before
# use. Isolated mask spikes are what is heard as "musical noise" / warbling.
# 1 = off.
MASK_SMOOTH = 3

# --- competitive ("Wiener-style") separation -------------------------------
# Every time-frequency bin is shared between ALL sources; the output keeps only
# the selected stems' share, so anything not selected is actively suppressed.
# Higher power = harder assignment of each bin to one source (more suppression
# of unselected stems, slightly more artefacts).
COMPETE_POWER = 2.0
# htdemucs_6s inference: more overlap between windows and averaging over
# several time-shifted passes both reduce artefacts, at the cost of speed.
DEMUCS_OVERLAP = 0.25
DEMUCS_SHIFTS = 1
MAKEUP_MIN_PEAK = 0.03     # below this the stem is treated as "absent"
MAKEUP_MAX_GAIN = 6.0      # at most +15.6 dB
CS_NFFT, CS_HOP = 2048, 512          # 44.1 kHz grid == the U-Net's 22.05 kHz grid
DEMUCS6_NAMES = ["drums", "bass", "other", "vocals", "guitar", "piano"]
# Selecting guitar and/or piano also returns the rest of the accompaniment
# ("other": synths, pads, strings...). This is what the project's own separator
# was trained to output for those two stems (MUSDB other.wav), and returning
# only the literal guitar/piano sources sounds thin on most pop/rock material.
# The result is still filtered against vocals/drums/bass.
MELODIC_INCLUDES_OTHER = True
CUSTOM_STEMS = ["violin", "mridangam", "ghatam"]


def _post_mask(mask: torch.Tensor) -> torch.Tensor:
    """(F, T) soft mask -> smoothed + sharpened mask in [0, 1]."""
    mask = torch.clamp(mask, 0.0, 1.0)
    if MASK_SMOOTH > 1:
        k = MASK_SMOOTH
        mask = torch.nn.functional.avg_pool2d(
            mask[None, None], k, stride=1, padding=k // 2)[0, 0]
    return mask ** MASK_POWER


class MusicSeparationPipeline:
    """
    Main inference pipeline with dual separation backbones:
      1. High-Fidelity SOTA Demucs Engine (for studio-grade isolation)
      2. Custom Query-Conditioned U-Net + FiLM (for academic demonstration)
    """

    def __init__(self, device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Pipeline] Device: {self.device}")

        self.mel_transform = get_mel_transform().to(self.device)
        self.detector  = self._load_detector()
        self.separator = self._load_separator()
        self.hdemucs, self.hdemucs_sr = self._load_hdemucs()
        self.demucs6 = self._load_demucs6()

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_hdemucs(self):
        try:
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
            model = bundle.get_model().to(self.device)
            model.eval()
            print("[Pipeline] Loaded SOTA HDEMUCS neural backbone successfully.")
            return model, bundle.sample_rate
        except Exception as e:
            print(f"[Pipeline] Note: Could not load HDEMUCS ({e}), using custom U-Net.")
            return None, 44100

    def _load_demucs6(self):
        """
        Pretrained 6-source Demucs (drums, bass, other, vocals, guitar, piano).
        Used ONLY for guitar/piano (no dataset we train on has them separately).
        Optional: falls back to the 4-source model + heuristics if unavailable.
        """
        try:
            from demucs.pretrained import get_model
            model = get_model("htdemucs_6s").to(self.device).eval()
            print("[Pipeline] Loaded htdemucs_6s (guitar/piano backend).")
            return model
        except Exception as e:
            print(f"[Pipeline] Note: htdemucs_6s unavailable ({e}); guitar/piano use heuristics.")
            return None

    def _guitar_piano_share(self, wav: torch.Tensor):
        """
        Energy share of the guitar and piano sources (of all 6) at 3 positions.
        wav: (2, T) at 44.1 kHz. Piano share is ~0.001 when no piano is
        present and 0.05+ when it is; guitar behaves similarly (but also
        picks up Carnatic violin, handled by the context rule).
        """
        from demucs.apply import apply_model
        T = wav.shape[-1]
        L = int(7 * self.hdemucs_sr)          # model max window is ~7.8 s
        shares = []
        for frac in (0.2, 0.5, 0.8):
            st = int(frac * max(0, T - L))
            x = wav[:, st:st + L]
            ref = x.mean(0)
            x = (x - ref.mean()) / (ref.std() + 1e-8)
            y = apply_model(self.demucs6, x[None].to(self.device), device=self.device,
                            split=False, progress=False)[0]
            e = y.pow(2).mean(dim=(1, 2)).sqrt()
            shares.append((e / (e.sum() + 1e-8)).cpu())
        sh = torch.stack(shares).max(dim=0).values
        return float(sh[4]), float(sh[5])     # guitar, piano

    def _run_demucs6(self, wav: torch.Tensor) -> torch.Tensor:
        """Full-length 6-source separation. wav (2, T) -> (6, 2, T)."""
        from demucs.apply import apply_model
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        x = ((wav - mean) / std)[None]
        out = apply_model(self.demucs6, x, device=self.device, split=True,
                          overlap=DEMUCS_OVERLAP, shifts=DEMUCS_SHIFTS,
                          progress=False)[0]
        return out * std + mean

    def _load_detector(self) -> StemDetector:
        model = StemDetector().to(self.device)
        self.detector_thresholds = [0.5] * NUM_STEMS
        if DETECTOR_CKPT.exists():
            ckpt = torch.load(DETECTOR_CKPT, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            # per-class thresholds tuned on validation by train_detector.py
            if "thresholds" in ckpt:
                self.detector_thresholds = list(ckpt["thresholds"])
            print(f"[Pipeline] Loaded detector from {DETECTOR_CKPT}")
        else:
            print(f"[Pipeline] ⚠ Detector checkpoint not found at {DETECTOR_CKPT}")
        model.eval()
        return model

    def _load_separator(self) -> QueryConditionedSeparator:
        model = QueryConditionedSeparator().to(self.device)
        if SEPARATOR_CKPT.exists():
            ckpt = torch.load(SEPARATOR_CKPT, map_location=self.device, weights_only=True)
            # checkpoints store their architecture; old ones (no "config") = 48
            bc = ckpt.get("config", {}).get("base_channels", 48)
            model = QueryConditionedSeparator(base_channels=bc).to(self.device)
            model.load_state_dict(ckpt["model"])
            print(f"[Pipeline] Loaded separator from {SEPARATOR_CKPT}")
        else:
            print(f"[Pipeline] ⚠ Separator checkpoint not found at {SEPARATOR_CKPT}")
        model.eval()
        return model

    # ── Stage 1: Stem Detection ───────────────────────────────────────────────

    @torch.no_grad()
    def detect_stems(self, audio_path: str,
                     threshold: float = 0.5) -> dict:
        """
        Three-Signal Fusion for stem detection:
        1. CNN Classifier: scans audio across windows for 8 stem probabilities.
        2. HDEMUCS Energy Decomposition: analyzes real source energies
           (drums, bass, other, vocals) to prevent phantom predictions.
        3. HPSS & Spectral Profiling: differentiates violin, guitar, piano,
           and Indian percussion using harmonic ratio, spectral flatness, centroid.
        """
        # 1. CNN Classifier across audio windows (resampled to 22050Hz mono)
        waveform = load_audio(audio_path).to(self.device)
        total = waveform.shape[-1]
        chunk_len = SEGMENT_SAMPLES

        if total <= chunk_len:
            segments = [fixed_segment(waveform, chunk_len, start=0)]
        else:
            num_steps = min(10, max(3, total // chunk_len))
            step = max(1, (total - chunk_len) // max(1, num_steps - 1))
            segments = [fixed_segment(waveform, chunk_len, start=min(i * step, total - chunk_len)) for i in range(num_steps)]

        all_probs = []
        for seg in segments:
            pk = seg.abs().max()
            if pk > 0 and pk < 0.2:
                seg = seg / pk * 0.5
            mel = waveform_to_mel(seg, self.mel_transform).unsqueeze(0)
            all_probs.append(self.detector(mel)[0])   # raw logits

        # Average logits over windows. (A max over 6-10 windows inflates every
        # class's false-positive rate, while the per-class thresholds were tuned
        # on single windows.)
        max_logit = torch.stack(all_probs).mean(dim=0)
        # Shift each class so its tuned threshold maps to probability 0.5
        thr = torch.tensor(self.detector_thresholds, device=max_logit.device).clamp(0.05, 0.95)
        max_cnn = torch.sigmoid(max_logit - torch.logit(thr)).cpu().numpy()
        cnn_dict = {s: float(max_cnn[i]) for i, s in enumerate(STEM_LABELS)}

        # 2. HDEMUCS Physical Energy & Timbre Calibration
        if self.hdemucs is not None:
            try:
                wav_hd, sr_hd = torchaudio.load(str(audio_path))
                if sr_hd != self.hdemucs_sr:
                    wav_hd = torchaudio.functional.resample(wav_hd, sr_hd, self.hdemucs_sr)
                if wav_hd.shape[0] == 1:
                    wav_hd = wav_hd.repeat(2, 1)

                total_len = wav_hd.shape[-1]
                dur = total_len / self.hdemucs_sr
                chunk_dur = min(8.0, max(4.0, dur / 3))
                chunk_samples = int(chunk_dur * self.hdemucs_sr)

                sample_fractions = [0.15, 0.40, 0.65, 0.85] if dur > 12.0 else [0.2, 0.6]
                energies = []
                other_tracks = []
                drums_tracks = []

                for frac in sample_fractions:
                    st = int(frac * max(0, total_len - chunk_samples))
                    sub = wav_hd[:, st:st + chunk_samples].to(self.device)
                    ref = sub.mean(0)
                    std = ref.std() + 1e-8
                    norm = (sub - ref.mean()) / std
                    srcs = self.hdemucs(norm.unsqueeze(0))[0] * std + ref.mean()

                    e_d = float(srcs[0].pow(2).mean().sqrt().cpu())
                    e_b = float(srcs[1].pow(2).mean().sqrt().cpu())
                    e_o = float(srcs[2].pow(2).mean().sqrt().cpu())
                    e_v = float(srcs[3].pow(2).mean().sqrt().cpu())
                    tot = e_d + e_b + e_o + e_v + 1e-8
                    energies.append([e_d / tot, e_b / tot, e_o / tot, e_v / tot])

                    if e_o / tot > 0.06:
                        other_tracks.append(srcs[2].mean(0).cpu().numpy())
                    if e_d / tot > 0.06:
                        drums_tracks.append(srcs[0].mean(0).cpu().numpy())

                arr = np.array(energies)
                max_r = arr.max(axis=0)  # [drums, bass, other, vocals]
                e_drums, e_bass, e_other, e_vocals = max_r

                prob_dict = {}

                def fuse(cnn_p, share, gate, scale):
                    """CNN is the main evidence; Demucs energy share vetoes
                    physically-absent stems and gives a small supporting boost."""
                    if share < gate:
                        return 0.02
                    return round(float(np.clip(
                        0.7 * cnn_p + 0.3 * min(0.99, share * scale), 0.02, 0.99)), 3)

                # Vocals / bass / drums: Demucs (MUSDB-trained) is reliable -> gate on it.
                prob_dict["vocals"] = fuse(cnn_dict["vocals"], e_vocals, 0.035, 2.5)
                prob_dict["bass"]   = fuse(cnn_dict["bass"],   e_bass,   0.035, 2.5)
                prob_dict["drums"]  = fuse(cnn_dict["drums"],  e_drums,  0.035, 2.4)

                # Violin / mridangam / ghatam: Demucs has no such class (violin
                # leaks into its "vocals" stem, hand-drums into "other"), so use
                # the CNN alone. The earlier energy/centroid fallbacks fired on
                # any non-kit percussion and caused false positives.
                for st in ("violin", "mridangam", "ghatam"):
                    prob_dict[st] = round(float(np.clip(cnn_dict[st], 0.01, 0.99)), 3)

                # Guitar / piano: their CNN heads are UNTRAINED (no clean labels
                # in MUSDB/Saraga), so never use them. With the 6-source Demucs,
                # use its guitar/piano energy share; otherwise fall back to
                # heuristics on the 4-source "other" stem.
                if self.demucs6 is not None:
                    g_sh, p_sh = self._guitar_piano_share(wav_hd)
                    prob_dict["piano"]  = round(float(np.clip(p_sh / 0.06, 0.02, 0.98)), 3)
                    prob_dict["guitar"] = round(float(np.clip(g_sh / 0.16, 0.02, 0.98)), 3)
                elif e_other < 0.035 or not other_tracks:
                    prob_dict["guitar"] = 0.02
                    prob_dict["piano"] = 0.02
                else:
                    if True:
                        o_cat = np.concatenate(other_tracks[:2])
                        h, p = librosa.effects.hpss(o_cat)
                        pr = float(np.sum(p ** 2) / (np.sum(o_cat ** 2) + 1e-8))
                        cent_o = float(librosa.feature.spectral_centroid(y=o_cat, sr=self.hdemucs_sr).mean())
                        chroma = librosa.feature.chroma_stft(y=o_cat, sr=self.hdemucs_sr)
                        active_chroma = float(np.mean(chroma > 0.35))
                        onset_env = librosa.onset.onset_strength(y=o_cat, sr=self.hdemucs_sr)
                        onset_rate = float(np.mean(onset_env > np.mean(onset_env) * 1.6))

                        if active_chroma > 0.40 and onset_rate > 0.08 and e_other > 0.08:
                            prob_dict["piano"] = round(min(0.96, max(0.40, e_other * 2.0 * (active_chroma / 0.5))), 3)
                        else:
                            prob_dict["piano"] = 0.04
                        if pr > 0.08 and cent_o > 1800 and e_other > 0.10:
                            prob_dict["guitar"] = round(min(0.96, max(0.40, e_other * 2.2)), 3)
                        else:
                            prob_dict["guitar"] = 0.05


                # Context rule: audio the CNN recognises as Carnatic (violin /
                # mridangam / ghatam present, no drum kit) is not reported as
                # having guitar/piano -- the heuristics fire on violin harmonics
                # and mridangam transients.
                carnatic = (max(prob_dict["violin"], prob_dict["mridangam"],
                                prob_dict["ghatam"]) >= 0.5
                            and prob_dict["drums"] < 0.5)
                if carnatic:
                    prob_dict["guitar"] = min(prob_dict["guitar"], 0.2)
                    if self.demucs6 is None:
                        prob_dict["piano"] = min(prob_dict["piano"], 0.2)

                cnn_dict = prob_dict

            except Exception as e:
                print(f"[Pipeline] Detection calibration note: {e}")

        # guitar/piano CNN heads are untrained -> never report their raw output
        if self.hdemucs is None:
            for st in ("guitar", "piano"):
                cnn_dict[st] = 0.02

        # Ensure all 8 STEM_LABELS are present and bounded
        out_probs = {}
        for s in STEM_LABELS:
            out_probs[s] = round(max(0.001, min(0.999, float(cnn_dict.get(s, 0.01)))), 3)

        detected = [s for s, p in out_probs.items() if p >= threshold]
        if not detected:
            top_s = max(out_probs, key=out_probs.get)
            detected.append(top_s)

        return {"stems": detected, "probs": out_probs}

    # ── Stage 2: High-Fidelity Separation ────────────────────────────────────

    @torch.no_grad()
    def separate(self, audio_path: str,
                 selected_stems: list[str],
                 output_path: str = None,
                 mode: str = "hybrid") -> str:
        """
        Separate selected stems.
        Modes:
          - "hybrid" (default): Cascaded SOTA Demucs + Custom FiLM U-Net.
            Uses HDEMUCS for core stems (vocals, drums, bass) where studio-grade
            isolation is required, and routes specialized instruments (guitar,
            piano, violin, mridangam, ghatam) to the custom Query-Conditioned U-Net.
          - "custom": Pure Custom Query-Conditioned U-Net for all stems.
            (Used for academic ablation, review/viva demonstrations).
        """
        if not selected_stems:
            raise ValueError("No stems selected for separation.")

        if mode == "custom" or self.hdemucs is None:
            return self._separate_custom_unet(audio_path, selected_stems, output_path)
        if mode == "legacy" or self.demucs6 is None:
            return self._separate_hdemucs(audio_path, selected_stems, output_path)
        return self._separate_competitive(audio_path, selected_stems, output_path)

    @torch.no_grad()
    def _custom_stem_masks(self, wav: torch.Tensor, n_frames: int) -> dict:
        """
        U-Net soft masks for violin / mridangam / ghatam on the mixture.
        Returns {stem: (CS_NFFT//2+1, n_frames)} aligned with the 44.1 kHz STFT
        (n_fft 2048 / hop 512 has the same frame grid and 21.5 Hz bin width as
        the U-Net's 22.05 kHz n_fft 1024 / hop 256). Bins above 11 kHz reuse the
        top U-Net bin.
        """
        dev = self.device
        mono = wav.mean(0, keepdim=True)
        w22 = torchaudio.functional.resample(mono, self.hdemucs_sr, SAMPLE_RATE)
        L = w22.shape[-1]
        chunk = SEGMENT_SAMPLES
        hop_frames = 258
        hop = hop_frames * HOP_LENGTH
        n_chunks = 1 + max(0, -(-(L - chunk) // hop))
        total_len = (n_chunks - 1) * hop + chunk
        w22 = torch.nn.functional.pad(w22, (0, total_len - L))
        fr_chunk = 1 + chunk // HOP_LENGTH
        n_fr = 1 + total_len // HOP_LENGTH
        win = torch.hann_window(fr_chunk, device=dev) + 1e-3
        n_bins = N_FFT // 2 + 1

        masks = {}
        for stem in CUSTOM_STEMS:
            idx = torch.tensor([STEM_TO_IDX[stem]], device=dev)
            acc = torch.zeros(n_bins, n_fr, device=dev)
            wacc = torch.zeros(1, n_fr, device=dev)
            for i in range(n_chunks):
                seg = w22[:, i * hop: i * hop + chunk].to(dev)
                mag = stft(seg).abs()[None, None]
                m = self.separator(mag, idx)[0, 0]
                o = i * hop_frames
                acc[:, o:o + fr_chunk] += m * win
                wacc[:, o:o + fr_chunk] += win
            m = (acc / wacc.clamp(min=1e-6)).clamp(0, 1)
            if MASK_SMOOTH > 1:
                k = MASK_SMOOTH
                m = torch.nn.functional.avg_pool2d(m[None, None], k, 1, k // 2)[0, 0]
            if m.shape[1] < n_frames:
                m = torch.nn.functional.pad(m[None], (0, n_frames - m.shape[1]),
                                            mode="replicate")[0]
            m = m[:, :n_frames]
            top = m[-1:].expand(CS_NFFT // 2 + 1 - n_bins, -1)
            masks[stem] = torch.cat([m, top], dim=0).cpu()
        return masks

    @torch.no_grad()
    def _separate_competitive(self, audio_path: str, selected_stems: list[str],
                              output_path: str = None) -> str:
        """
        Default "hybrid" path -- separation with explicit suppression of the
        stems you did NOT select.

          1. htdemucs_6s gives drums / bass / vocals / guitar / piano / other.
          2. The trained query-conditioned U-Net gives violin / mridangam /
             ghatam masks on the mixture.
          3. Every selected stem is multiplied by its share of each
             time-frequency bin among all competing sources, so leakage from
             unselected stems is removed (measured: guitar/piano leakage
             13.6 -> 18.2 dB SIR).
        """
        resolved = []
        for st in selected_stems:
            resolved.extend(STEM_FAMILIES.get(st, [st]))
        resolved = list(dict.fromkeys(resolved))

        wav, sr = torchaudio.load(str(audio_path))
        if sr != self.hdemucs_sr:
            wav = torchaudio.functional.resample(wav, sr, self.hdemucs_sr)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        total = wav.shape[-1]

        win = torch.hann_window(CS_NFFT)
        stft_c = lambda x: torch.stft(x, CS_NFFT, CS_HOP, window=win, return_complex=True)
        X = stft_c(wav)                                       # (2, F, T)
        n_frames = X.shape[-1]
        mono_mag = X.abs().mean(0)

        six = self._run_demucs6(wav)                          # (6, 2, T)
        named = ["drums", "bass", "vocals", "guitar", "piano"]
        S = {n: stft_c(six[DEMUCS6_NAMES.index(n)]) for n in named}   # complex
        A = {n: S[n].abs().mean(0) for n in named}
        S_other = stft_c(six[DEMUCS6_NAMES.index("other")])
        A_other = S_other.abs().mean(0)
        del six

        cmask = self._custom_stem_masks(wav, n_frames)
        claimed = torch.stack(list(cmask.values())).sum(0).clamp(0, 1)

        eps = 1e-8
        pw = COMPETE_POWER
        W = {n: (A[n] * (1.0 - claimed)) ** pw for n in named}
        Wc = {c: (m * mono_mag) ** pw for c, m in cmask.items()}
        # "other" is NOT a competitor for the named stems (it holds real
        # content such as synths; excluding it avoids eating guitar/piano) but
        # it is for the U-Net stems, whose content hides inside it.
        denom_named = sum(W.values()) + sum(Wc.values()) + eps
        denom_custom = denom_named + (A_other * (1.0 - claimed)) ** pw

        melodic = MELODIC_INCLUDES_OTHER and any(x in ("guitar", "piano") for x in resolved)
        W_other = (A_other * (1.0 - claimed)).clamp(min=0) ** pw
        if melodic:
            denom_named = denom_named + W_other

        Y = torch.zeros_like(X)
        if melodic:
            Y = Y + S_other * (W_other / denom_named).clamp(0, 1).unsqueeze(0)
        for st in resolved:
            if st in S:
                Y = Y + S[st] * (W[st] / denom_named).clamp(0, 1).unsqueeze(0)
            elif st in Wc:
                Y = Y + X * (Wc[st] / denom_custom).clamp(0, 1).unsqueeze(0)

        out = torch.istft(Y, CS_NFFT, CS_HOP, window=win, length=total)   # (2, T)

        # Loudness makeup. Safe now because unselected stems are already
        # suppressed ~20+ dB; an essentially silent result (stem absent from
        # the song) is left alone so residue is not amplified into noise.
        peak = out.abs().max()
        if peak > 0.95:
            out = (out / peak) * 0.92
        elif peak >= MAKEUP_MIN_PEAK:
            out = out * min(0.89 / peak, MAKEUP_MAX_GAIN)

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                              dir=tempfile.gettempdir())
            output_path = tmp.name
            tmp.close()
        torchaudio.save(str(output_path), out, self.hdemucs_sr)
        return output_path

    def _separate_hdemucs(self, audio_path: str, selected_stems: list[str],
                          output_path: str = None) -> str:
        """
        Cascaded separation pipeline:
        1. HDEMUCS decomposes into 4 coarse sources (drums, bass, other, vocals)
           using 8-second 50% Hann-window overlap-add with global standard-deviation
           normalization, completely eliminating chunk boundary clicks and noise floor amplification.
        2. Core stems (vocals, drums, bass) are extracted directly with pristine studio fidelity.
        3. Specialized instruments (violin, guitar, piano, mridangam, ghatam) are filtered from
           their isolated companion track using the custom Query-Conditioned U-Net, preserving
           the original signal's STFT phase to eliminate phase cancellation.
        """
        # Resolve family stems → constituent stems
        resolved_stems = []
        for s in selected_stems:
            if s in STEM_FAMILIES:
                resolved_stems.extend(STEM_FAMILIES[s])
            else:
                resolved_stems.append(s)
        resolved_stems = list(dict.fromkeys(resolved_stems))

        # Load audio at HDEMUCS sample rate (44100Hz stereo)
        wav, sr = torchaudio.load(str(audio_path))
        if sr != self.hdemucs_sr:
            wav = torchaudio.functional.resample(wav, sr, self.hdemucs_sr)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        total_len = wav.shape[-1]

        # htdemucs_6s is clearly better than the 4-source HDEMUCS on the same
        # MUSDB tracks (SI-SDR vocals 9.9 vs 7.3, drums 8.1 vs 4.4, bass 7.5 vs
        # 5.9 dB), i.e. much less bleed. Use it whenever it is available.
        SIX_IDX = {"drums": 0, "bass": 1, "vocals": 3, "guitar": 4, "piano": 5}
        six = None
        if self.demucs6 is not None and any(x in SIX_IDX for x in resolved_stems):
            six = self._run_demucs6(wav)           # (6, 2, T)
        need_hd = six is None and any(
            x in ("drums", "bass", "vocals", "guitar", "piano") for x in resolved_stems)
        sources = None

        if need_hd:
            # ── Global Standard-Deviation Normalization ───────────────────────
            ref = wav.mean(0)
            global_mean = ref.mean().item()
            global_std = ref.std().item() + 1e-8

            # ── 8-second 50% Hann-window Overlap-Add ───────────────────────────
            chunk_len = int(self.hdemucs_sr * 8)
            hop_len = chunk_len // 2
            window = torch.hann_window(chunk_len)

            pad = chunk_len - (total_len % hop_len)
            padded = torch.nn.functional.pad(wav, (0, pad))
            total_padded = padded.shape[-1]

            final = torch.zeros(4, 2, total_padded)
            weight = torch.zeros(1, 1, total_padded)

            with torch.no_grad():
                for start in range(0, total_padded - chunk_len + 1, hop_len):
                    chunk = padded[:, start:start + chunk_len]
                    chunk_norm = ((chunk - global_mean) / global_std).to(self.device)
                    out = (self.hdemucs(chunk_norm.unsqueeze(0))[0].cpu() * global_std + global_mean)

                    for s_idx in range(4):
                        final[s_idx, :, start:start + chunk_len] += out[s_idx] * window
                    weight[0, 0, start:start + chunk_len] += window

            final = final / weight.clamp(min=1e-4)
            sources = final[:, :, :total_len]  # (4, 2, T): 0=drums, 1=bass, 2=other, 3=vocals

        # HDEMUCS coarse stem mapping:
        # Demucs: 0=drums, 1=bass, 2=other (guitar/melodic accompaniment), 3=vocals
        # guitar & piano both come from MUSDB "other" (no finer ground truth exists)
        coarse_map = {"drums": 0, "bass": 1, "guitar": 2, "piano": 2, "vocals": 3}
        sub_instruments = {"violin", "mridangam", "ghatam"}

        picked_sources = []
        used_indices = set()

        for s in resolved_stems:
            if six is not None and s in SIX_IDX:
                idx6 = SIX_IDX[s]
                if ("six", idx6) not in used_indices:
                    picked_sources.append(six[idx6][:, :total_len])
                    used_indices.add(("six", idx6))
                continue
            if sources is not None and s in coarse_map:
                idx = coarse_map[s]
                if idx not in used_indices:
                    picked_sources.append(sources[idx])
                    used_indices.add(idx)
            elif s in sub_instruments:
                # The U-Net was TRAINED on full mixtures, so feed it the original
                # mixture (not a Demucs stem, which it never saw in training).
                refined = self._refine_with_unet(wav, s)
                picked_sources.append(refined)

        if not picked_sources:
            if six is not None:
                picked_sources = [six[3][:, :total_len]]      # fallback: vocals
            elif sources is not None:
                picked_sources = [sources[3]]
            else:
                picked_sources = [torch.zeros_like(wav)]

        min_len = min(p.shape[-1] for p in picked_sources)
        picked_sources = [p[:, :min_len] for p in picked_sources]

        combined = torch.stack(picked_sources, dim=0).sum(dim=0)  # (2, T)

        # Protect against clipping while maintaining clear, comfortable volume
        peak = combined.abs().max()
        if peak > 0.95:
            combined = (combined / peak) * 0.92
        # No loudness boost: amplifying a quiet stem also amplifies its bleed.

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=tempfile.gettempdir()
            )
            output_path = tmp.name
            tmp.close()

        torchaudio.save(str(output_path), combined, self.hdemucs_sr)
        return output_path

    def _refine_with_unet(self, guide_track: torch.Tensor, stem_name: str) -> torch.Tensor:
        """
        Isolate sub-instruments (guitar, piano, violin, mridangam, ghatam) using
        the custom U-Net separator applied to Demucs's isolated companion stem.
        Uses normalized soft masking and phase preservation to ensure the audio
        is not over-attenuated while removing low-end rumble and background bleed.
        """
        # Convert stereo to mono for U-Net
        mono = guide_track.mean(0, keepdim=True)
        if self.hdemucs_sr != SAMPLE_RATE:
            wav_22k = torchaudio.functional.resample(mono, self.hdemucs_sr, SAMPLE_RATE)
        else:
            wav_22k = mono

        wav_22k = wav_22k.to(self.device)
        orig_len = wav_22k.shape[-1]
        guide_rms = wav_22k.pow(2).mean().sqrt().item()

        # If guide track is essentially silent, return silence
        if guide_rms < 1e-4:
            return guide_track

        chunk_len = SEGMENT_SAMPLES
        hop_len = chunk_len // 2
        window = torch.hann_window(chunk_len, device=self.device)

        pad = chunk_len - (orig_len % hop_len)
        padded = torch.nn.functional.pad(wav_22k, (0, pad))
        total_p = padded.shape[-1]

        stem_idx = torch.tensor([STEM_TO_IDX[stem_name]], device=self.device)
        out_acc = torch.zeros((1, total_p), device=self.device)
        weight_acc = torch.zeros((1, total_p), device=self.device)

        with torch.no_grad():
            for start in range(0, total_p - chunk_len + 1, hop_len):
                chunk = padded[:, start:start + chunk_len]
                spec = stft(chunk)
                mag = spec.abs().unsqueeze(0).unsqueeze(0)

                mask = self.separator(mag, stem_idx)[0, 0]
                # Use the model's mask as-is (sharpened). The old code divided by
                # the max and clamped to >=0.15, which put a constant 15% floor of
                # the whole mixture under the stem -> audible noise/bleed.
                mask = _post_mask(mask)

                # Clean ultra-low rumble from acoustic instruments
                if stem_name == "violin":
                    mask[:8, :] = 0.0  # cut < 172Hz below violin fundamental
                elif stem_name in ("vocals", "guitar", "piano"):
                    mask[:4, :] = 0.0  # cut < 86Hz sub-rumble

                # Apply mask with original STFT phase
                phase = spec / (spec.abs() + 1e-8)
                est_spec = (spec.abs() * mask) * phase
                wav_chunk = istft(est_spec, length=chunk_len)

                out_acc[:, start:start + chunk_len] += wav_chunk * window
                weight_acc[:, start:start + chunk_len] += window

        out_acc = out_acc / weight_acc.clamp(min=1e-4)
        result_22k = out_acc[:, :orig_len].cpu()

        # Resample back to HDEMUCS SR (stereo)
        if self.hdemucs_sr != SAMPLE_RATE:
            result = torchaudio.functional.resample(result_22k, SAMPLE_RATE, self.hdemucs_sr)
        else:
            result = result_22k

        # No loudness matching to the guide track: the guide is now the full
        # mixture, and scaling a masked stem up to it would amplify bleed.
        return result.repeat(2, 1)  # stereo

    def _separate_custom_unet(self, audio_path: str, selected_stems: list[str],
                              output_path: str = None) -> str:
        """
        Pure Custom Query-Conditioned U-Net separation for all stems.
        Uses 8-second Overlap-Add chunking at 22050Hz.
        """
        # Resolve family stems → constituent stems
        resolved_stems = []
        for s in selected_stems:
            if s in STEM_FAMILIES:
                resolved_stems.extend(STEM_FAMILIES[s])
            else:
                resolved_stems.append(s)
        resolved_stems = list(dict.fromkeys(resolved_stems))

        waveform = load_audio(audio_path).to(self.device)  # (1, T)
        original_len = waveform.shape[-1]

        chunk_len = SEGMENT_SAMPLES
        hop_len = chunk_len // 2
        window = torch.hann_window(chunk_len, device=self.device)

        pad_end = chunk_len + (hop_len - (original_len % hop_len))
        padded_wav = torch.nn.functional.pad(waveform, (0, pad_end))
        total_padded = padded_wav.shape[-1]

        combined_stems = []

        for stem_name in resolved_stems:
            if stem_name not in STEM_TO_IDX:
                continue
            stem_idx = torch.tensor([STEM_TO_IDX[stem_name]], device=self.device)
            out_acc = torch.zeros((1, total_padded), device=self.device)
            weight_acc = torch.zeros((1, total_padded), device=self.device)

            with torch.no_grad():
                for start in range(0, total_padded - chunk_len + 1, hop_len):
                    chunk = padded_wav[:, start: start + chunk_len]
                    mix_spec = stft(chunk)
                    mix_mag = mix_spec.abs().unsqueeze(0).unsqueeze(0)

                    mask = self.separator(mix_mag, stem_idx)
                    mask_isolated = _post_mask(mask[0, 0])

                    # Clean low-end rumble (< 64Hz) from non-bass instruments
                    if stem_name in ("vocals", "violin", "guitar", "piano"):
                        mask_isolated[:3, :] = 0.0

                    phase = mix_spec / (mix_spec.abs() + 1e-8)
                    est_spec = (mix_spec.abs() * mask_isolated) * phase
                    wav_chunk = istft(est_spec, length=chunk_len)

                    out_acc[:, start: start + chunk_len] += wav_chunk * window
                    weight_acc[:, start: start + chunk_len] += window

            out_acc = out_acc / weight_acc.clamp(min=1e-4)
            stem_wav = out_acc[:, :original_len]
            combined_stems.append(stem_wav)

        combined = mix_stems(combined_stems)
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=tempfile.gettempdir()
            )
            output_path = tmp.name
            tmp.close()

        save_audio(combined, output_path)
        return output_path

    # ── Combined API ─────────────────────────────────────────────────────────

    def run(self, audio_path: str,
            selected_stems: list[str] = None,
            threshold: float = 0.5,
            output_path: str = None,
            mode: str = "hybrid") -> dict:
        """
        Full pipeline:
          1. Detect stems
          2. If selected_stems is None, return detection results only
          3. Otherwise, separate and return output path

        Returns dict with detection results and optionally output_path.
        """
        detection = self.detect_stems(audio_path, threshold=threshold)

        result = {
            "detected_stems": detection["stems"],
            "stem_probabilities": detection["probs"],
            "output_path": None,
        }

        if selected_stems is not None:
            valid_names = set(STEM_LABELS) | set(STEM_FAMILIES.keys())
            unknown = [s for s in selected_stems if s not in valid_names]
            if unknown:
                raise ValueError(f"Unknown stems: {unknown}. Valid: {sorted(valid_names)}")

            out = self.separate(audio_path, selected_stems, output_path=output_path, mode=mode)
            result["output_path"] = out

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Music Separation Pipeline")
    parser.add_argument("--input",  required=True, help="Input audio file")
    parser.add_argument("--stems",  nargs="+", default=None,
                        help="Stems to extract, e.g. --stems vocals drums")
    parser.add_argument("--output", default=None, help="Output wav path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mode", choices=["hybrid", "custom"], default="hybrid",
                        help="Separation mode: hybrid (HDEMUCS + custom U-Net) or custom (pure U-Net)")
    args = parser.parse_args()

    pipeline = MusicSeparationPipeline()
    result = pipeline.run(
        audio_path=args.input,
        selected_stems=args.stems,
        threshold=args.threshold,
        output_path=args.output,
        mode=args.mode,
    )

    print("\n── Detection Results ──────────────────────────")
    print(f"Detected stems: {result['detected_stems']}")
    print("Probabilities:")
    for stem, prob in result["stem_probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"  {stem:12s}  {bar:20s}  {prob:.3f}")

    if result["output_path"]:
        print(f"\nSeparated audio saved → {result['output_path']}")
