/**
 * main.js — StemQuery Frontend Logic
 * ────────────────────────────────────
 * Handles:
 *   1. Drag-and-drop / browse file upload
 *   2. POST /detect  → show stem cards + probability bars
 *   3. Stem selection (checkbox-style cards)
 *   4. POST /separate → play + download output audio
 */

// ── Stem metadata (emoji + display name) ──────────────────────────────────
const STEM_META = {
  vocals:    { emoji: "🎤", label: "Vocals" },
  drums:     { emoji: "🥁", label: "Drums" },
  bass:      { emoji: "🎸", label: "Bass" },
  guitar:    { emoji: "🪕", label: "Guitar" },
  piano:     { emoji: "🎹", label: "Piano" },
  violin:    { emoji: "🎻", label: "Violin" },
  mridangam: { emoji: "🪘", label: "Mridangam" },
  ghatam:    { emoji: "🏺", label: "Ghatam" },
};
const STEM_ORDER = [
  "vocals", "drums", "bass", "guitar", "piano", "violin", "mridangam", "ghatam"
];

// ── State ──────────────────────────────────────────────────────────────────
let uploadId     = null;
let uploadSuffix = null;
let selectedFile = null;
let detectedStems   = [];
let selectedStems   = new Set();

// ── DOM Refs ───────────────────────────────────────────────────────────────
const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const browseBtn     = document.getElementById("browse-btn");
const clearFileBtn  = document.getElementById("clear-file");
const detectBtn     = document.getElementById("detect-btn");
const separateBtn   = document.getElementById("separate-btn");
const selectedFileEl = document.getElementById("selected-file");
const fileNameEl    = document.getElementById("file-name");

const secUpload  = document.getElementById("section-upload");
const secDetect  = document.getElementById("section-detect");
const secOutput  = document.getElementById("section-output");

const stemsGrid   = document.getElementById("stems-grid");
const probBars    = document.getElementById("prob-bars");
const selHint     = document.getElementById("selection-hint");

const downloadBtn   = document.getElementById("download-btn");
const outputDesc    = document.getElementById("output-desc");
const newSepBtn     = document.getElementById("new-separation-btn");
const newUploadBtn  = document.getElementById("new-upload-btn");

// WaveSurfer A/B Player DOM Refs
const abStatusPill   = document.getElementById("ab-status-pill");
const abStatusText   = document.getElementById("ab-status-text");
const sepTrackTitle  = document.getElementById("sep-track-title");
const trackBlockOrig = document.getElementById("track-block-orig");
const trackBlockSep  = document.getElementById("track-block-sep");
const soloOrigBtn    = document.getElementById("solo-orig-btn");
const soloSepBtn     = document.getElementById("solo-sep-btn");

const btnPlayPause   = document.getElementById("btn-play-pause");
const playIcon       = document.getElementById("play-icon");
const btnRestart     = document.getElementById("btn-restart");
const timeCurrentEl  = document.getElementById("time-current");
const timeTotalEl    = document.getElementById("time-total");

const toggleAbOrig   = document.getElementById("toggle-ab-orig");
const toggleAbSep    = document.getElementById("toggle-ab-sep");
const toggleAbFlip   = document.getElementById("toggle-ab-flip");
const btnVolume      = document.getElementById("btn-volume");
const volumeSlider   = document.getElementById("volume-slider");

// WaveSurfer Audio State
let wsOrig = null;
let wsSep  = null;
let activeTrack = "sep"; // "sep" or "orig"
let isPlaying   = false;
let currentVolume = 1.0;
let isMuted       = false;

const loadingOverlay = document.getElementById("loading-overlay");
const loadingTitle   = document.getElementById("loading-title");
const loadingSub     = document.getElementById("loading-sub");
const errorToast     = document.getElementById("error-toast");
const toastMsg       = document.getElementById("toast-msg");
const toastClose     = document.getElementById("toast-close");


// ── Helpers ────────────────────────────────────────────────────────────────
function showLoading(title, sub = "This may take a few seconds.") {
  loadingTitle.textContent = title;
  loadingSub.textContent   = sub;
  loadingOverlay.classList.remove("hidden");
}
function hideLoading() { loadingOverlay.classList.add("hidden"); }

function showError(msg) {
  toastMsg.textContent = msg;
  errorToast.classList.remove("hidden");
  setTimeout(() => errorToast.classList.add("hidden"), 6000);
}
toastClose.addEventListener("click", () => errorToast.classList.add("hidden"));

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  fileNameEl.textContent = file.name;
  selectedFileEl.classList.remove("hidden");
  detectBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  selectedFileEl.classList.add("hidden");
  detectBtn.disabled = true;
}


// ── Drag-and-drop ──────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave",  () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
clearFileBtn.addEventListener("click", clearFile);


// ── Step 1 → Detect ────────────────────────────────────────────────────────
detectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  showLoading("Analysing audio…", "Detecting which instruments are present.");

  const formData = new FormData();
  formData.append("audio", selectedFile);

  try {
    const res  = await fetch("/detect", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || "Detection failed.");
    }

    uploadId     = data.upload_id;
    uploadSuffix = data.upload_suffix;
    detectedStems = data.detected_stems;
    hasVideo      = !!data.has_video;

    renderStemGrid(data.detected_stems, data.stem_probabilities);
    renderProbBars(data.stem_probabilities);

    secDetect.classList.remove("hidden");
    secDetect.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
});


// ── Render Stem Grid ───────────────────────────────────────────────────────
function renderStemGrid(detected, probs) {
  stemsGrid.innerHTML = "";
  selectedStems.clear();
  separateBtn.disabled = true;
  updateSelectionHint();

  STEM_ORDER.forEach(stem => {
    const meta    = STEM_META[stem] || { emoji: "🎵", label: stem };
    const isFound = detected.includes(stem);
    const prob    = probs[stem] || 0;

    const card = document.createElement("div");
    card.className = "stem-card" + (isFound ? " detected" : " not-detected");
    card.dataset.stem = stem;

    card.innerHTML = `
      <span class="stem-badge ${isFound ? "present" : "absent"}">
        ${isFound ? "✓ Found" : "✗ Absent"}
      </span>
      <span class="stem-emoji">${meta.emoji}</span>
      <span class="stem-name">${meta.label}</span>
    `;

    if (isFound) {
      card.addEventListener("click", () => toggleStem(stem, card));
    }
    stemsGrid.appendChild(card);
  });
}

function toggleStem(stem, card) {
  if (selectedStems.has(stem)) {
    selectedStems.delete(stem);
    card.classList.remove("selected");
  } else {
    selectedStems.add(stem);
    card.classList.add("selected");
  }
  separateBtn.disabled = selectedStems.size === 0;
  updateSelectionHint();
}

function updateSelectionHint() {
  if (selectedStems.size === 0) {
    selHint.textContent = "Select at least one stem above to continue.";
    selHint.classList.remove("active");
  } else {
    const names = [...selectedStems]
      .map(s => STEM_META[s]?.label || s)
      .join(" + ");
    selHint.textContent = `Will extract: ${names}`;
    selHint.classList.add("active");
  }
}


// ── Render Probability Bars ────────────────────────────────────────────────
function renderProbBars(probs) {
  probBars.innerHTML = "";
  STEM_ORDER.forEach(stem => {
    const p    = probs[stem] || 0;
    const meta = STEM_META[stem] || { label: stem };
    const row  = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-label">${meta.label}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width: 0%" data-target="${(p * 100).toFixed(1)}%"></div>
      </div>
      <span class="prob-val">${(p * 100).toFixed(0)}%</span>
    `;
    probBars.appendChild(row);
  });

  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-fill").forEach(fill => {
      fill.style.width = fill.dataset.target;
    });
  });
}


// ── Time Formatter ─────────────────────────────────────────────────────────
function formatTime(sec) {
  if (isNaN(sec) || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}


// ── Synchronized A/B WaveSurfer Player ──────────────────────────────────────
function cleanupWaveSurfers() {
  if (wsOrig) {
    try { wsOrig.destroy(); } catch (e) {}
    wsOrig = null;
  }
  if (wsSep) {
    try { wsSep.destroy(); } catch (e) {}
    wsSep = null;
  }
  isPlaying = false;
  activeTrack = "sep";
  if (playIcon) playIcon.textContent = "▶";
}

function syncVolume() {
  if (!wsSep || !wsOrig) return;
  const vol = isMuted ? 0 : currentVolume;
  if (activeTrack === "sep") {
    wsSep.setVolume(vol);
    wsOrig.setVolume(0);
  } else {
    wsOrig.setVolume(vol);
    wsSep.setVolume(0);
  }
}

function updateTrackVisuals() {
  if (activeTrack === "sep") {
    trackBlockSep.classList.add("active");
    trackBlockOrig.classList.remove("active");
    soloSepBtn.classList.add("active");
    soloOrigBtn.classList.remove("active");
    toggleAbSep.classList.add("active");
    toggleAbOrig.classList.remove("active");
    abStatusPill.classList.remove("orig-active");
    abStatusText.textContent = "Playing: Separated Stems";
  } else {
    trackBlockOrig.classList.add("active");
    trackBlockSep.classList.remove("active");
    soloOrigBtn.classList.add("active");
    soloSepBtn.classList.remove("active");
    toggleAbOrig.classList.add("active");
    toggleAbSep.classList.remove("active");
    abStatusPill.classList.add("orig-active");
    abStatusText.textContent = "Playing: Original Mixture";
  }
}

function setTrackMode(mode) {
  activeTrack = mode;
  syncVolume();
  updateTrackVisuals();
}

function toggleAB() {
  setTrackMode(activeTrack === "sep" ? "orig" : "sep");
}

async function togglePlay() {
  if (!wsSep || !wsOrig) return;
  if (isPlaying) {
    wsSep.pause();
    wsOrig.pause();
    isPlaying = false;
    playIcon.textContent = "▶";
  } else {
    // Re-synchronize exact playback head before playing
    const curTime = wsSep.getCurrentTime();
    wsOrig.setTime(curTime);
    syncVolume();
    try {
      await Promise.all([wsSep.play(), wsOrig.play()]);
      isPlaying = true;
      playIcon.textContent = "❚❚";
    } catch (err) {
      console.warn("Autoplay blocked or playback error:", err);
    }
  }
}

function initWaveformPlayer(origUrl, sepUrl, stemNames) {
  cleanupWaveSurfers();

  if (sepTrackTitle) {
    sepTrackTitle.textContent = `Isolated: ${stemNames}`;
  }

  // 1. Initialize Original Track WaveSurfer
  wsOrig = WaveSurfer.create({
    container: "#waveform-orig",
    waveColor: "rgba(6, 182, 212, 0.35)",
    progressColor: "#06b6d4",
    cursorColor: "#ffffff",
    cursorWidth: 2,
    height: 70,
    barWidth: 2,
    barGap: 2,
    barRadius: 2,
    url: origUrl,
  });

  // 2. Initialize Separated Track WaveSurfer
  wsSep = WaveSurfer.create({
    container: "#waveform-sep",
    waveColor: "rgba(139, 92, 246, 0.35)",
    progressColor: "#8b5cf6",
    cursorColor: "#ffffff",
    cursorWidth: 2,
    height: 70,
    barWidth: 2,
    barGap: 2,
    barRadius: 2,
    url: sepUrl,
  });

  // Master time update & drift prevention
  wsSep.on("timeupdate", (t) => {
    timeCurrentEl.textContent = formatTime(t);
    const dur = wsSep.getDuration();
    if (dur && dur > 0) timeTotalEl.textContent = formatTime(dur);

    // Lock-step drift correction (> 50ms)
    if (wsOrig && isPlaying) {
      const origT = wsOrig.getCurrentTime();
      if (Math.abs(origT - t) > 0.05) {
        wsOrig.setTime(t);
      }
    }
  });

  wsSep.on("ready", () => {
    const dur = wsSep.getDuration();
    timeTotalEl.textContent = formatTime(dur);
    syncVolume();
  });

  wsOrig.on("ready", () => {
    syncVolume();
  });

  // Synchronized Interactive Seeking
  wsSep.on("interaction", (newTime) => {
    if (wsOrig) wsOrig.setTime(newTime);
    timeCurrentEl.textContent = formatTime(newTime);
  });

  wsOrig.on("interaction", (newTime) => {
    if (wsSep) wsSep.setTime(newTime);
    timeCurrentEl.textContent = formatTime(newTime);
  });

  // Track finished
  wsSep.on("finish", () => {
    isPlaying = false;
    playIcon.textContent = "▶";
    if (wsOrig) {
      wsOrig.pause();
      wsOrig.setTime(0);
    }
    wsSep.setTime(0);
  });

  // Set default active track (Separated)
  setTrackMode("sep");
}


// ── Player Controls Event Listeners ────────────────────────────────────────
btnPlayPause.addEventListener("click", togglePlay);

btnRestart.addEventListener("click", () => {
  if (wsSep && wsOrig) {
    wsSep.setTime(0);
    wsOrig.setTime(0);
    timeCurrentEl.textContent = "0:00";
  }
});

// A/B Solo & Toggle Buttons
soloOrigBtn.addEventListener("click", () => setTrackMode("orig"));
soloSepBtn.addEventListener("click",  () => setTrackMode("sep"));
toggleAbOrig.addEventListener("click", () => setTrackMode("orig"));
toggleAbSep.addEventListener("click",  () => setTrackMode("sep"));
toggleAbFlip.addEventListener("click", toggleAB);

// Volume & Mute
volumeSlider.addEventListener("input", (e) => {
  currentVolume = parseFloat(e.target.value);
  if (isMuted && currentVolume > 0) {
    isMuted = false;
    btnVolume.textContent = "🔊";
  }
  syncVolume();
});

btnVolume.addEventListener("click", () => {
  isMuted = !isMuted;
  btnVolume.textContent = isMuted ? "🔇" : "🔊";
  syncVolume();
});

// Keyboard Shortcuts: Space (Play/Pause), S (A/B Swap)
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (secOutput.classList.contains("hidden")) return;

  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  } else if (e.code === "KeyS") {
    e.preventDefault();
    toggleAB();
  }
});


// ── Step 2 → Separate ──────────────────────────────────────────────────────
separateBtn.addEventListener("click", async () => {
  if (!uploadId || selectedStems.size === 0) return;

  const stemList = [...selectedStems];
  const stemNames = stemList.map(s => STEM_META[s]?.label || s).join(" + ");
  showLoading(`Separating: ${stemNames}`,
    "Running query-conditioned separation. This may take a moment…");

  try {
    const res  = await fetch("/separate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        upload_id:      uploadId,
        upload_suffix:  uploadSuffix,
        selected_stems: stemList,
      }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || "Separation failed.");
    }

    const origAudioUrl = `/original/${uploadId}`;
    const sepAudioUrl  = `/audio/${data.output_id}`;

    lastOutputId = data.output_id;
    downloadVideoBtn.classList.toggle("hidden", !hasVideo);
    downloadBtn.href = sepAudioUrl;
    downloadBtn.download = `${stemNames.replace(/ \+ /g, "_")}_separated.wav`;
    outputDesc.textContent = `A/B synchronized comparison for: ${stemNames}`;

    secOutput.classList.remove("hidden");
    secOutput.scrollIntoView({ behavior: "smooth", block: "start" });

    // Initialize WaveSurfer dual player
    initWaveformPlayer(origAudioUrl, sepAudioUrl, stemNames);

  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
});


// ── Navigation ─────────────────────────────────────────────────────────────
newSepBtn.addEventListener("click", () => {
  cleanupWaveSurfers();
  secOutput.classList.add("hidden");
  secDetect.scrollIntoView({ behavior: "smooth", block: "start" });
});

newUploadBtn.addEventListener("click", () => {
  hasVideo = false;
  lastOutputId = null;
  cleanupWaveSurfers();
  secOutput.classList.add("hidden");
  secDetect.classList.add("hidden");
  clearFile();
  selectedStems.clear();
  uploadId = null;
  stemsGrid.innerHTML = "";
  probBars.innerHTML  = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
});


// ── Video files ──────────────────────────────────────────────────
// Video upload needs nothing here: the server extracts the audio track and the
// flow above continues unchanged. This block adds a button that puts the
// separated audio back into the video.
let hasVideo     = false;
let lastOutputId = null;

const downloadVideoBtn = document.getElementById("download-video-btn");

downloadVideoBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  if (!uploadId || !lastOutputId) return;
  showLoading("Building video…", "Putting the separated audio into the original video.");
  try {
    const res  = await fetch("/remux", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: uploadId, output_id: lastOutputId }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "Could not build the video.");
    window.location.href = `/video/${data.video_id}`;
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
});
