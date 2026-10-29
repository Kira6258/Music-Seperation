"""
app.py  —  Flask Web UI for Music Separation
--------------------------------------------
Routes:
  GET  /              → serve index.html
  POST /detect        → upload audio, return detected stems + probabilities
  POST /separate      → separate selected stems, return audio file
  GET  /audio/<name>  → serve a separated audio file

Added for video input (see media_io.py; models are untouched):
  POST /remux        → put the separated audio back into the uploaded video
  GET  /video/<id>    → serve that rebuilt video
"""

import sys
import os
import uuid
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import (Flask, request, jsonify, render_template,
                   send_file, abort)

from inference.pipeline import MusicSeparationPipeline
from utils.audio_utils import STEM_LABELS, STEM_FAMILIES
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import media_io

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# Temp directory for uploaded + separated audio
UPLOAD_DIR = Path(tempfile.gettempdir()) / "music_sep_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "music_sep_outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"} | media_io.VIDEO_EXTS

# uploads can now be videos, so allow larger request bodies (1 GB)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

# Singleton pipeline (loads models once at startup)
pipeline: MusicSeparationPipeline = None


def get_pipeline() -> MusicSeparationPipeline:
    global pipeline
    if pipeline is None:
        pipeline = MusicSeparationPipeline()
    return pipeline


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", all_stems=STEM_LABELS)


@app.route("/detect", methods=["POST"])
def detect():
    """
    Receives uploaded audio, runs stem detection.
    Returns JSON:
      {
        "upload_id": "uuid",
        "detected_stems": ["vocals", "drums"],
        "stem_probabilities": {"vocals": 0.94, ...}
      }
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    f = request.files["audio"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    # Save upload with a UUID to avoid conflicts
    upload_id = str(uuid.uuid4())
    suffix = Path(f.filename).suffix.lower()
    upload_path = UPLOAD_DIR / f"{upload_id}{suffix}"
    f.save(str(upload_path))

    # Video upload: keep the video for later, and continue with its audio track
    # as an ordinary WAV, so everything below is unchanged.
    has_video = False
    if suffix in media_io.VIDEO_EXTS:
        video_path = UPLOAD_DIR / f"{upload_id}_video{suffix}"
        upload_path.rename(video_path)
        suffix = ".wav"
        upload_path = UPLOAD_DIR / f"{upload_id}{suffix}"
        try:
            media_io.extract_audio(video_path, upload_path)
        except media_io.MediaError as e:
            return jsonify({"error": str(e)}), 400
        has_video = True

    try:
        pipe = get_pipeline()
        detection = pipe.detect_stems(str(upload_path), threshold=0.5)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "upload_id":         upload_id,
        "upload_suffix":     suffix,
        "detected_stems":    detection["stems"],
        "stem_probabilities": detection["probs"],
        "has_video":         has_video,
    })


@app.route("/separate", methods=["POST"])
def separate():
    """
    Receives JSON: { "upload_id": "...", "upload_suffix": ".mp3", "selected_stems": ["vocals"] }
    Returns JSON:  { "output_id": "uuid" }   — then client calls /audio/<output_id>
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    upload_id     = data.get("upload_id")
    upload_suffix = data.get("upload_suffix", ".wav")
    selected      = data.get("selected_stems", [])

    if not upload_id:
        return jsonify({"error": "upload_id required"}), 400
    if not selected:
        return jsonify({"error": "Select at least one stem"}), 400

    # Validate stems against all individual stems and family groups
    valid_names = set(STEM_LABELS) | set(STEM_FAMILIES.keys())
    invalid = [s for s in selected if s not in valid_names]
    if invalid:
        return jsonify({"error": f"Unknown stems: {invalid}"}), 400

    upload_path = UPLOAD_DIR / f"{upload_id}{upload_suffix}"
    if not upload_path.exists():
        return jsonify({"error": "Uploaded file not found. Please re-upload."}), 404

    output_id   = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"{output_id}.wav"
    mode        = data.get("mode", "hybrid")

    try:
        pipe = get_pipeline()
        pipe.separate(str(upload_path), selected, str(output_path), mode=mode)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify({"output_id": output_id})


@app.route("/audio/<output_id>")
def serve_audio(output_id: str):
    """Serve a separated audio file for download / browser playback."""
    # Security: only allow safe filenames
    safe_id = "".join(c for c in output_id if c.isalnum() or c == "-")
    audio_path = OUTPUT_DIR / f"{safe_id}.wav"
    if not audio_path.exists():
        abort(404)
    return send_file(str(audio_path), mimetype="audio/wav",
                     as_attachment=False)


@app.route("/original/<upload_id>")
def serve_original(upload_id: str):
    """Serve the original uploaded audio file for A/B comparison."""
    safe_id = "".join(c for c in upload_id if c.isalnum() or c == "-")
    for ext in ALLOWED_EXTENSIONS:
        p = UPLOAD_DIR / f"{safe_id}{ext}"
        if p.exists():
            mime_types = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".flac": "audio/flac",
                ".ogg": "audio/ogg",
                ".m4a": "audio/mp4",
            }
            return send_file(str(p), mimetype=mime_types.get(ext, "audio/wav"),
                             as_attachment=False)
    abort(404)


@app.route("/remux", methods=["POST"])
def remux():
    """
    Receives JSON {"upload_id": "...", "output_id": "..."} and returns
    {"video_id": "..."}: the uploaded video with the separated audio as its sound.
    """
    data = request.get_json(silent=True) or {}
    safe = lambda s: "".join(c for c in str(s) if c.isalnum() or c == "-")
    upload_id, output_id = safe(data.get("upload_id", "")), safe(data.get("output_id", ""))

    videos = sorted(UPLOAD_DIR.glob(f"{upload_id}_video.*")) if upload_id else []
    audio = OUTPUT_DIR / f"{output_id}.wav"
    if not videos or not audio.exists():
        return jsonify({"error": "No video is associated with this upload."}), 404

    src = videos[0]
    ext = ".mp4" if src.suffix.lower() in (".mp4", ".m4v", ".mov") else ".mkv"
    video_id = str(uuid.uuid4())
    try:
        media_io.mux_video(src, audio, OUTPUT_DIR / f"{video_id}{ext}")
    except media_io.MediaError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"video_id": video_id, "ext": ext})


@app.route("/video/<video_id>")
def serve_video(video_id: str):
    """Serve a rebuilt video (separated audio muxed into the original picture)."""
    safe_id = "".join(c for c in video_id if c.isalnum() or c == "-")
    for ext, mime in ((".mp4", "video/mp4"), (".mkv", "video/x-matroska")):
        p = OUTPUT_DIR / f"{safe_id}{ext}"
        if p.exists():
            return send_file(str(p), mimetype=mime, as_attachment=True,
                             download_name=f"separated_video{ext}")
    abort(404)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading models...")
    get_pipeline()  # pre-load on startup
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
