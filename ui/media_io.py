"""
media_io.py
-----------
Helpers that turn a VIDEO file into an audio file the existing pipeline can
read, and put separated audio back into the video. Nothing here touches the
models or the separation code.

  extract_audio()  video file  -> 44.1 kHz stereo WAV (ffmpeg)
  mux_video()      original video + separated WAV -> new video file

ffmpeg comes from the `imageio-ffmpeg` package (a bundled binary), so no system
install is needed. A system ffmpeg on PATH is used as a fallback.
"""

import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

MAX_SECONDS = 12 * 60                 # longest clip accepted (seconds)
FFMPEG_TIMEOUT = 600                  # seconds


class MediaError(ValueError):
    """A problem the user can understand (bad link, no audio track, too long)."""


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise MediaError("ffmpeg is not available. Run: pip install imageio-ffmpeg")


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)


# ─────────────────────────────────────────────────────────────────────────────
# Video file -> WAV
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(src, dst, max_seconds: int = MAX_SECONDS) -> Path:
    """Decode the first audio track of `src` to a 44.1 kHz stereo 16-bit WAV."""
    src, dst = Path(src), Path(dst)
    cmd = [ffmpeg_exe(), "-nostdin", "-y", "-i", str(src), "-vn", "-map", "0:a:0",
           "-ac", "2", "-ar", "44100", "-t", str(max_seconds), "-c:a", "pcm_s16le", str(dst)]
    r = _run(cmd)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 1024:
        raise MediaError("No readable audio track was found in this file.")
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Original video + separated audio -> new video
# ─────────────────────────────────────────────────────────────────────────────

def mux_video(video, audio_wav, dst) -> Path:
    """Keep the video stream untouched and replace its sound with `audio_wav`."""
    video, audio_wav, dst = Path(video), Path(audio_wav), Path(dst)
    cmd = [ffmpeg_exe(), "-nostdin", "-y", "-i", str(video), "-i", str(audio_wav),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", str(dst)]
    r = _run(cmd)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 1024:
        raise MediaError("Could not build the video (it may have no video stream).")
    return dst
