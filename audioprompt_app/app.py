import os
import sys
import gc
from pathlib import Path

import numpy as np
import streamlit as st
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


def _rms_dbfs(y: np.ndarray) -> float:
    y = y.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(y * y)) + 1e-12)
    return 20.0 * np.log10(rms)

# Target RMS level used when loudness-matching layers before blending.
# ~-20 dBFS. Adjust here to shift all layer blend levels up or down uniformly.
TARGET_BLEND_RMS = 0.1

def _normalize_to_rms(y: np.ndarray, target_rms: float = TARGET_BLEND_RMS) -> np.ndarray:
    """Scale y so its RMS equals target_rms before blend-gain is applied.

    Leaves near-silent arrays unscaled to avoid amplifying noise floor.
    Caps the scale factor at 10x to prevent extreme gain on very sparse layers.
    """
    y = y.astype(np.float32, copy=False)
    current_rms = float(np.sqrt(np.mean(y * y)) + 1e-12)
    if current_rms < 1e-6:
        return y
    scale = min(target_rms / current_rms, 10.0)
    return (y * scale).astype(np.float32)

# Import core from ./src (ensure our local package takes precedence)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# Import directly from submodules to avoid package-level side effects
from audioprompt_core.audio import load_audio_mono, apply_fades, wav_bytes, wav_bytes_concat_segments
from audioprompt_core.prompt import (
    pink_noise,
    imprint_melody_focus,
    rhythmic_gate_from_events,
    apply_hpf,
    apply_mono_lows,
)
from audioprompt_core.melody import (
    SCALES,
    NOTE_TO_MIDI,
    generate_random_melody,
    events_to_f0,
)
from audioprompt_core.mididrums import (
    inspect_midi_timing,
    parse_midi_drum_events,
    scale_lane_events,
    loop_lane_events,
    LANES,
)
from audioprompt_core.formants import build_vowel_plan
from audioprompt_core.midibass import (
    inspect_bass_midi_timing,
    parse_midi_bass_events,
    scale_bass_events,
    loop_bass_events,
    bass_events_to_f0,
)
from audioprompt_core.drumnoise import (
    build_drum_tone_params,
    synthesize_drum_layer,
)


st.set_page_config(page_title="AudioPrompt", page_icon="🎛️", layout="wide")

# Optional render cap for public hosting. Unset/empty/invalid = no cap, so
# local runs (and forks) behave exactly as before; the hosted demo sets e.g.
# AUDIOPROMPT_MAX_SECONDS=60. All cap-related UI is hidden when no cap is set.
def _max_prompt_seconds() -> float | None:
    raw = os.environ.get("AUDIOPROMPT_MAX_SECONDS", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None

MAX_PROMPT_SECONDS = _max_prompt_seconds()

# Optional GoatCounter analytics endpoint. Unset (the default for local dev and
# anyone self-hosting a clone) = no analytics, no network calls, nothing phones
# home. The hosted demo sets e.g.
# GOATCOUNTER_URL=https://audioprompt.goatcounter.com/count; forks can point it
# at their own dashboard.
GOATCOUNTER_URL = os.environ.get("GOATCOUNTER_URL", "").strip() or None

# UI language + accessibility layer. Importing i18n wraps the common st.*
# widgets so labels, tooltips, and option displays translate via the strings
# table, and so tooltips can render as visible captions ("help as text" mode)
# for screen readers. Rendered first so the controls are the first elements a
# screen reader reaches.
import i18n
from i18n import t

i18n.init()
i18n.render_controls()

# ------------------------------ Debug helpers ------------------------------ #
def _mem_usage_mb() -> float | None:
    """Best-effort current RSS in MB.

    Prefers psutil. On Linux without psutil, reads /proc/self/status (VmRSS).
    Falls back to resource.getrusage only as a last resort (note: that's peak).
    """
    # psutil preferred (current RSS)
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        pass
    # Linux /proc fallback
    try:
        if sys.platform.startswith("linux") and os.path.exists("/proc/self/status"):
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = float(parts[1])  # value in kB
                            return kb / 1024.0
    except Exception:
        pass
    # Last resort: peak (not current)
    try:
        import resource  # type: ignore
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("darwin"):
            return float(rss) / (1024 ** 2)  # bytes -> MB
        return float(rss) / 1024.0  # kB -> MB
    except Exception:
        return None

def _log_debug(msg: str):
    """Log diagnostics to server logs (stdout). Keeps UI clean.

    On Streamlit Community Cloud, these appear under app logs. Locally,
    they print to the terminal running `streamlit run`.
    """
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _clear_outputs():
    """Drop cached output bytes and related keys to free memory."""
    for k in (
        "prompt_bytes",
        "prompt_key",
        "combined_bytes",
        "combined_key",
        "combined_name",
        "drum_prompt_bytes",
        "drum_prompt_key",
        "bass_prompt_bytes",
        "bass_prompt_key",
    ):
        try:
            st.session_state.pop(k, None)
        except Exception:
            pass
    gc.collect()
    _log_debug(f"[clear] outputs cleared; mem={_mem_usage_mb()} MB")


def _upload_signature(uploaded_file):
    """Return a simple signature for the uploaded file to detect changes."""
    if uploaded_file is None:
        return None
    return (getattr(uploaded_file, "name", None), getattr(uploaded_file, "size", None))


def _inspect_uploaded_drum_midi(uploaded_file) -> dict | None:
    if uploaded_file is None:
        return None
    sig = _upload_signature(uploaded_file)
    if st.session_state.get("_drum_bpm_sig") == sig:
        return st.session_state.get("drum_timing_preview")
    try:
        timing = inspect_midi_timing(uploaded_file.getvalue())
    except Exception as e:
        _log_debug(f"[drum] BPM preview failed: {e}")
        timing = None
    st.session_state["_drum_bpm_sig"] = sig
    st.session_state["drum_timing_preview"] = timing
    st.session_state["detected_drum_bpm_preview"] = timing["bpm"] if timing else None
    if timing is not None:
        st.session_state["drum_bpm_value"] = int(round(float(timing["bpm"])))
    return timing

def _inspect_uploaded_bass_midi(uploaded_file) -> dict | None:
    if uploaded_file is None:
        return None
    sig = _upload_signature(uploaded_file)
    if st.session_state.get("_bass_bpm_sig") == sig:
        return st.session_state.get("bass_timing_preview")
    try:
        timing = inspect_bass_midi_timing(uploaded_file.getvalue())
    except Exception as e:
        _log_debug(f"[bass] BPM preview failed: {e}")
        timing = None
    st.session_state["_bass_bpm_sig"] = sig
    st.session_state["bass_timing_preview"] = timing
    st.session_state["detected_bass_bpm_preview"] = timing["bpm"] if timing else None
    if timing is not None:
        st.session_state["bass_bpm_value"] = int(round(float(timing["bpm"])))
    return timing


def _matched_drum_length_seconds(timing: dict | None, target_bpm: float) -> float | None:
    if not timing or not timing.get("length_s"):
        return None
    detected_bpm = max(float(timing.get("bpm", 0.0)), 1e-6)
    return float(timing["length_s"]) * detected_bpm / max(float(target_bpm), 1e-6)


def _slug(value: object) -> str:
    text = str(value).lower().replace("_", "-").replace(" ", "-")
    keep = []
    last_dash = False
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
            last_dash = False
        elif ch == "-" and keep and not last_dash:
            keep.append(ch)
            last_dash = True
    return "".join(keep).strip("-") or "none"


def _root_tag(root: object) -> str:
    text = str(root)
    return text.lower().replace("#", "s").replace("b", "b")


def _scale_tag(scale: str, enabled: bool) -> str:
    if not enabled:
        return "none"
    aliases = {
        "natural_minor": "natmin",
        "harmonic_minor": "harmmin",
        "melodic_minor": "melmin",
        "minor_pentatonic": "minpent",
        "major_pentatonic": "majpent",
        "minor_blues": "minblues",
        "major_blues": "majblues",
        "octatonic_whole_half": "octwh",
        "octatonic_half_whole": "octhw",
        "whole_tone": "whole",
        "double_harmonic": "dblharm",
        "harmonic_major": "harmaj",
    }
    return aliases.get(scale, _slug(scale))


def _focus_tag(enabled: bool, preset: str | None, focus_band) -> str:
    if not enabled:
        return "none"
    aliases = {"vocal": "voc", "guitar": "gtr", "bass": "bass"}
    if preset:
        return aliases.get(str(preset), _slug(preset))
    if focus_band is not None:
        return f"b{int(focus_band[0])}-{int(focus_band[1])}"
    return "custom"


def _vowel_tag(language: str, strength: float, accent_period: int) -> str:
    codes = {"english": "EN", "japanese": "JA", "spanish": "ES"}
    code = codes.get(language, language[:2].upper())
    tag = f"{code}-s{int(round(float(strength) * 100))}"
    if language == "english":
        # Accent period only shapes the plan for stress-timed English
        tag += f"-a{int(accent_period)}"
    return tag


def _download_name(kind: str, base_stem: str | None, meta: dict) -> str:
    prefix = _slug(base_stem) if base_stem else "ap"
    parts = [
        prefix,
        kind,
        f"r-{_root_tag(meta.get('root', 'none'))}",
        f"s-{meta.get('scale', 'none')}",
        f"mb-{int(round(float(meta.get('melody_bpm', 0))))}",
        f"ms-{meta.get('melody_seed', 'none')}",
    ]
    drum_bpm = meta.get("drum_bpm")
    if drum_bpm is not None:
        parts.append(f"db-{int(round(float(drum_bpm)))}")
    drum_seed = meta.get("drum_seed")
    if drum_seed is not None:
        parts.append(f"ds-{drum_seed}")
    length_s = meta.get("length_s")
    if length_s is not None:
        parts.append(f"len-{float(length_s):.1f}s")
    focus = meta.get("focus", "none")
    if focus != "none":
        parts.append(f"f-{focus}")
    character = meta.get("character", "neutral")
    if character != "neutral":
        parts.append(f"ch-{_slug(character)}")
    vowel = meta.get("vowel")
    if vowel:
        parts.append(f"vow-{vowel}")
    bw = meta.get("bw_frac")
    if bw is not None and abs(float(bw) - 0.01) > 1e-4:
        parts.append(f"bw-{int(round(float(bw) * 1000))}")
    return "_".join(parts) + ".wav"
st.markdown(
    """
    <style>
    /* Hide Streamlit's fixed header/toolbar entirely: it holds nothing
       visitors need (just the ⋮ menu; Deploy is dev-only) and its 60px
       pushed all content down. Trade-off: the tiny top-right "Running…"
       indicator goes too — the app shows its own spinner during renders. */
    header[data-testid="stHeader"] { display: none !important; }
    .block-container{padding-top:1.25rem;padding-bottom:0.75rem; max-width:80vw; margin-left:auto; margin-right:auto;}
    /* Tuck the page title closer to the language/help controls row */
    .block-container h1 { padding-top: 0.25rem; }
    /* Keep Streamlit's primary theme color for the button; minimal tweaks only */
    div.stButton > button[kind="primary"] { padding: 0.9rem 1.25rem; font-size: 1.05rem; border-radius: 10px; }
    /* Prevent download labels from wrapping */
    div.stButton > button { white-space: nowrap; }
    /* Style the file uploader like a drop zone */
    div[data-testid="stFileUploader"] > section {
        border: 2px dashed rgba(255,255,255,0.2);
        padding: 1rem; border-radius: 10px; transition: border-color .2s, background-color .2s;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: var(--primary-color, #FF6B6B); background: rgba(255,107,107,0.06);
    }
    /* Highlight when dragging files over (JS adds .dragging) */
    div[data-testid="stFileUploader"] > section.dragging {
        border-color: var(--primary-color, #FF6B6B) !important;
        background: rgba(255,107,107,0.12) !important;
    }
    /* Fallback: focus within */
    div[data-testid="stFileUploader"]:focus-within > section {
        border-color: var(--primary-color, #FF6B6B);
    }
    /* Align Generate row (button + seed) to bottom of the row container.
       We insert a marker .gen-row before the st.columns; the next stHorizontalBlock is the row. */
    .gen-row + div[data-testid="stHorizontalBlock"] { align-items: flex-end; gap: 32px; }
    /* Inline audio + download with top alignment */
    .dl-row + div[data-testid="stHorizontalBlock"] { align-items: flex-start; gap: 0.75rem; }
    .dl-row + div[data-testid="stHorizontalBlock"] > div { width: auto; }
    .dl-row + div[data-testid="stHorizontalBlock"] .stButton button { width: auto; }
    /* Add horizontal padding to main two columns */
    /* Use Streamlit columns(gap=...) for spacing; no extra CSS gap/padding needed here. */
    .footer-note { font-size: 0.85rem; opacity: 0.75; margin: 0; }
    /* Hide the tiny iframe container used for JS injection at footer */
    .js-hook + div[data-testid="stElementContainer"] { display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }
    .js-hook + div[data-testid="stElementContainer"] iframe { display: none !important; height: 0 !important; }
    /* Robustly hide the injected iframe (match by srcdoc content) and its container */
    iframe[data-testid="stIFrame"][srcdoc*="attachDragHighlight"] { display: none !important; height: 0 !important; visibility: hidden !important; }
    div[data-testid="stElementContainer"] > iframe[data-testid="stIFrame"][srcdoc*="attachDragHighlight"] { display: none !important; height: 0 !important; }
    /* Hide the wrapper itself if it contains that iframe (requires :has support, works in modern browsers) */
    div[data-testid="stElementContainer"]:has(> iframe[data-testid="stIFrame"][srcdoc*="attachDragHighlight"]) {
        display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important;
    }
    /* Hide the GoatCounter injector iframe the same way */
    iframe[data-testid="stIFrame"][srcdoc*="injectGoatCounter"] { display: none !important; height: 0 !important; visibility: hidden !important; }
    div[data-testid="stElementContainer"]:has(> iframe[data-testid="stIFrame"][srcdoc*="injectGoatCounter"]) {
        display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important;
    }
    /* Attractive pill-styled placeholder blocks (blue) */
    .spec-placeholder {
        width: 100%;
        background: #16384b;
        border-radius: 14px;
        padding: 14px 18px;
        color: rgba(255,255,255,0.95);
        font-weight: 400;
        margin: 0 0 14px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Seed label is painted via CSS ::before in the default (compact) layout.
# In help-as-text mode the widget renders its real, screen-reader-visible
# label instead, so the overlay is skipped to avoid a double label.
if not i18n.help_as_text():
    st.markdown(
        f"""
        <style>
        .st-key-seed {{ position: relative; }}
        .st-key-seed::before {{ content: '{t("Seed")}'; position: absolute; top: -18px; left: 2px; font-weight: 600; font-size: 0.85rem; line-height: 1; opacity: 0.9; }}
        .st-key-seed [data-testid="stNumberInputContainer"] {{ height: 56px; }}
        .st-key-seed input[data-testid="stNumberInputField"] {{ height: 56px; padding-top: 0; padding-bottom: 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# JS helper: add/remove .dragging class on the file-uploader drop zone during drag events
_DRAG_JS = """
<script>
  (function attachDragHighlight(){
    function wire(el){
      if(el.__dragWired) return; el.__dragWired = true;
      const sec = el.querySelector('section');
      if(!sec) return;
      let counter = 0;
      function onEnter(e){ e.preventDefault(); counter++; sec.classList.add('dragging'); }
      function onOver(e){ e.preventDefault(); sec.classList.add('dragging'); }
      function onLeave(e){ counter--; if(counter<=0){ sec.classList.remove('dragging'); counter=0; } }
      function onDrop(e){ sec.classList.remove('dragging'); counter=0; }
      sec.addEventListener('dragenter', onEnter);
      sec.addEventListener('dragover', onOver);
      sec.addEventListener('dragleave', onLeave);
      sec.addEventListener('drop', onDrop);
    }
    function scan(){
      document.querySelectorAll('div[data-testid="stFileUploader"]').forEach(wire);
    }
    scan(); setInterval(scan, 1000);
  })();
</script>
"""

# GoatCounter analytics injector (only used when GOATCOUNTER_URL is set).
# Approach: st.markdown(unsafe_allow_html=True) can't be used — browsers never
# execute <script> tags inserted via innerHTML — and loading count.js directly
# inside components.html would run it in the sandboxed iframe, reporting the
# iframe's srcdoc location and losing the real page referrer (e.g. suno.com).
# Streamlit component iframes are same-origin, though, so this snippet creates
# the <script data-goatcounter="..." async src="//gc.zgo.at/count.js"></script>
# element in the PARENT (top-level) document, where count.js sees the true URL
# and referrer. count.js also ignores localhost by default.
def _goatcounter_js(endpoint: str) -> str:
    import json
    return """
<script>
  (function injectGoatCounter(){
    try {
      var doc = window.parent.document;
      if (doc.querySelector('script[data-goatcounter]')) return; // once per page
      var s = doc.createElement('script');
      s.setAttribute('data-goatcounter', %s);
      s.async = true;
      s.src = '//gc.zgo.at/count.js';
      doc.head.appendChild(s);
    } catch (e) { /* parent inaccessible; skip silently */ }
  })();
</script>
""" % json.dumps(endpoint)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_prompt(
    sr: int,
    prompt_seconds: float,
    seed: int,
    enable_melody: bool,
    melody_params: dict,
    enable_focus: bool,
    focus_params: dict,
    enable_gate: bool,
    imprint_params: dict,
    lowend_cfg: dict | None = None,
    character_params: dict | None = None,
):
    n = int(sr * prompt_seconds)
    x_pink = pink_noise(n, sr, seed=seed)
    # Low-end processing before STFT (optional)
    if lowend_cfg:
        if lowend_cfg.get("tame_low_end"):
            x_pink = apply_hpf(x_pink, sr, cutoff_hz=float(lowend_cfg.get("hpf_cutoff_hz", 25.0)), taps=int(lowend_cfg.get("hpf_taps", 2049)))
        if lowend_cfg.get("mono_lows"):
            x_pink = apply_mono_lows(x_pink, sr, cutoff_hz=float(lowend_cfg.get("mono_cutoff_hz", 120.0)), taps=int(lowend_cfg.get("hpf_taps", 2049)))
    events = None
    focus_arg = None
    if enable_focus:
        if focus_params.get("preset") and focus_params["preset"] != "custom":
            focus_arg = focus_params["preset"]
        elif focus_params.get("band") is not None:
            focus_arg = (int(focus_params["band"][0]), int(focus_params["band"][1]))
        else:
            focus_arg = None

    if enable_melody:
        events = generate_random_melody(
            duration_s=prompt_seconds,
            bpm=melody_params["bpm"],
            root=melody_params["root"],
            octave=4,
            scale=melody_params["scale"],
            low_midi=int(melody_params["low_midi"]),
            high_midi=int(melody_params["high_midi"]),
            step_bias=melody_params["step_bias"],
            leap_max_scale_steps=int(melody_params["leap_steps"]),
            rest_prob=melody_params["rest_prob"],
            durations_beats=melody_params["durations"],
            duration_probs=melody_params["duration_probs"],
            seed=seed,
        )
        f0 = events_to_f0(
            events,
            sr,
            n_samples=n,
            glide_prob=melody_params["glide_prob"],
            glide_frac=melody_params["glide_frac"],
            vibrato_hz=melody_params["vib_hz"],
            vibrato_depth=melody_params["vib_depth"],
            seed=seed,
        )
        cp = character_params or {}
        vowel_plan = (
            build_vowel_plan(
                events,
                language=str(cp.get("vowel_language", "english")),
                accent_period=int(cp.get("accent_period", 2)),
                seed=seed,
            )
            if cp.get("enable_formants")
            else None
        )
        y_prompt = imprint_melody_focus(
            x_pink,
            sr,
            f0_hz=f0,
            gain=imprint_params["gain"],
            harmonics=int(imprint_params["harmonics"]),
            bw_frac=imprint_params["bw_frac"],
            focus=focus_arg,
            band_floor_db=imprint_params["floor_db"],
            sharpness=imprint_params["sharpness"],
            n_fft=int(imprint_params["n_fft"]),
            character=cp.get("character", "neutral"),
            note_shape=cp.get("note_shape", "natural"),
            detune_spread_cents=float(cp.get("detune_spread_cents", 0.0)),
            vowel_plan=vowel_plan,
            formant_strength=float(cp.get("formant_strength", 1.0)),
            melody_noise_floor_db=float(imprint_params.get("melody_noise_floor_db", 0.0)),
        )
    elif enable_focus:
        y_prompt = imprint_melody_focus(
            x_pink,
            sr,
            f0_hz=0.0,
            gain=0.0,
            harmonics=0,
            bw_frac=imprint_params["bw_frac"],
            focus=focus_arg,
            band_floor_db=imprint_params["floor_db"],
            sharpness=imprint_params["sharpness"],
            n_fft=int(imprint_params["n_fft"]),
        )
    else:
        y_prompt = x_pink.astype(np.float32)

    if enable_gate and events is not None:
        shape = (character_params or {}).get("note_shape", "natural")
        decay_mult = float((character_params or {}).get("note_decay", 1.0))
        gate = rhythmic_gate_from_events(events, sr, n_samples=n, shape=shape, decay_mult=decay_mult)
        y_prompt = (y_prompt * (0.15 + 0.85 * gate)).astype(np.float32)

    if enable_melody and melody_params.get("drone_level", 0.0) > 0.0:
        from audioprompt_core.melody import midi_to_hz
        root_midi = NOTE_TO_MIDI.get(melody_params["root"], 60)
        drone_hz = midi_to_hz(root_midi - 24) # Two octaves down for bass drone
        t_arr = np.arange(n) / sr
        # Blend sine and pseudo-triangle for a warm tone
        drone_wave = 0.5 * np.sin(2 * np.pi * drone_hz * t_arr)
        drone_wave += 0.25 * np.arcsin(np.sin(2 * np.pi * drone_hz * t_arr))
        # Simple fade to avoid clicks
        fade_len = min(int(sr * 0.1), n // 2)
        if fade_len > 0:
            fade_env = np.ones(n, dtype=np.float32)
            fade_env[:fade_len] = np.linspace(0, 1, fade_len)
            fade_env[-fade_len:] = np.linspace(1, 0, fade_len)
            drone_wave *= fade_env
        y_prompt = y_prompt + (drone_wave * float(melody_params["drone_level"]) * 0.5).astype(np.float32)

    # Normalize
    peak = float(np.max(np.abs(y_prompt)) + 1e-12)
    if peak > 0:
        y_prompt = (y_prompt / peak).astype(np.float32)
    return y_prompt, events


st.title("🎛️ AudioPrompt")
top_left, top_right = st.columns([1, 1], gap="large")
with top_left:
    st.subheader("Quick Start")
    st.markdown(i18n.block("quick_start"))
with top_right:
    st.subheader("Input Audio")
    uploaded = st.file_uploader(
        "Input audio (optional)",
        type=["wav", "flac", "ogg", "aiff", "aif", "mp3"],
        accept_multiple_files=False,
        help=(
            "Drag & drop a file. If provided, the prompt will be prepended to create a combined output. "
            "MP3 support depends on your libsndfile build."
        ),
        key="uploaded_file",
    )
    sr = st.number_input(
        "Sample rate (Hz)",
        min_value=8000,
        max_value=96000,
        value=48000,
        step=1000,
        help="Processing rate; inputs are resampled. Higher SR costs more CPU.",
    )

    # Auto-clear outputs if the uploaded file changes (or is cleared)
    try:
        prev_sig = st.session_state.get("_prev_upload_sig", "__unset__")
        curr_sig = _upload_signature(st.session_state.get("uploaded_file"))
        if prev_sig != curr_sig:
            if prev_sig != "__unset__":
                _clear_outputs()
                _log_debug("[upload] input changed; outputs cleared")
            st.session_state["_prev_upload_sig"] = curr_sig
    except Exception:
        pass

    # Toggles moved next to relevant sections below
    # Prompt seconds moved to Output & Seed section for better workflow alignment

# Single divider spanning both columns to separate top row from main content
st.divider()

# Begin main columns
st.markdown("<div class='main-cols'>", unsafe_allow_html=True)
left, right = st.columns(2, gap="large")

with left:

    st.subheader("Melody")
    mcol1, mcol2 = st.columns([1,1])
    with mcol1:
        enable_melody = st.checkbox(
            "Enable melody",
            value=True,
            help="Imprint a randomized melody (scale‑constrained) onto pink noise.",
        )
    with mcol2:
        enable_gate = st.checkbox(
            "Rhythmic gate",
            value=True,
            help="Apply a note‑shaped amplitude envelope for phrasing.",
        )
    if enable_melody:
        roots = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
        scales = sorted(list(SCALES.keys()))
        # Root and Scale side-by-side
        msel1, msel2 = st.columns(2, gap="small")
        with msel1:
            melody_root = st.selectbox(
                "Root",
                roots,
                index=roots.index("E"),
                help="Root note for the scale (C4=60).",
            )
        with msel2:
            melody_scale = st.selectbox(
                "Scale",
                options=scales,
                index=scales.index("minor_pentatonic") if "minor_pentatonic" in scales else 0,
                help="Choose from major/modes, pentatonics, blues, etc.",
            )

        col1, col2 = st.columns(2, gap="small")
        with col1:
            bpm = st.slider("BPM", 20, 220, 96, 1, help="Tempo driving randomized note durations. Go as low as 20 for halftime vocals over 40 BPM drum tracks.")
        with col2:
            vib_hz = st.slider("Vibrato Hz", 3.0, 9.0, 5.5, 0.1, help="Rate of pitch modulation.")
        vib_depth = st.slider("Vibrato depth", 0.0, 0.05, 0.02, 0.001, help="Depth of pitch modulation (fraction).")
        ccol1, ccol2, ccol3 = st.columns(3, gap="small")
        with ccol1:
            melody_character = st.selectbox(
                "Melody character",
                options=["neutral", "warm", "voice", "reed", "bell", "pluck", "bright", "wide"],
                index=0,
                help=(
                    "Harmonic color of the melody imprint. "
                    "Neutral: flat harmonic stack. Warm: rolls off high harmonics for a mellow tone. "
                    "Voice: emphasizes low harmonics (1–5) for a vocal buzz. Reed: odd harmonics only (clarinet-like). "
                    "Bell: peaks around the 3rd harmonic for a metallic shimmer. Pluck: fast harmonic decay. "
                    "Bright: boosts higher harmonics for more edge. Wide: neutral harmonics + 7-cent detune spread for subtle thickness."
                ),
            )
        with ccol2:
            note_shape = st.selectbox(
                "Note shape",
                options=["natural", "tight", "pluck", "smooth"],
                index=0,
                help=(
                    "Shapes the base envelope for the melody notes. "
                    "Tight and pluck have exponential decays; smooth has a softer attack."
                ),
            )
        with ccol3:
            note_decay = st.slider(
                "Note decay",
                0.1, 3.0, 1.0, 0.1,
                help="Scales the length of the note envelope decay. Lower = shorter, staccato notes."
            )

        with st.expander("Melody – Advanced", expanded=False):
            # Auto-set Low/High MIDI to root note (octave 3) + 1.5 octaves when root changes.
            _root_midi_base = NOTE_TO_MIDI.get(melody_root, 60) - 12
            if st.session_state.get("_melody_root_for_range") != melody_root:
                st.session_state["low_midi_val"] = _root_midi_base
                st.session_state["high_midi_val"] = _root_midi_base + 18
                st.session_state["_melody_root_for_range"] = melody_root
            colA1, colA2 = st.columns(2, gap="small")
            with colA1:
                low_midi = st.slider("Low MIDI", 24, 84, _root_midi_base, 1,
                                     help="Register floor. Defaults to the root note in octave 3; adjust to taste.",
                                     key="low_midi_val")
                step_bias = st.slider("Step bias", 0.0, 1.0, 0.5, 0.01, help="Probability of moving to a neighboring scale degree.")
                glide_prob = st.slider("Glide prob", 0.0, 1.0, 0.05, 0.01, help="Probability of sliding into the next note.")
            with colA2:
                high_midi = st.slider("High MIDI", 36, 96, _root_midi_base + 18, 1,
                                      help="Register ceiling. Defaults to 1.5 octaves above Low MIDI.",
                                      key="high_midi_val")
                leap_steps = st.slider("Max leap (scale steps)", 1, 8, 7, 1, help="Largest jump when not stepping.")
                glide_frac = st.slider("Glide frac", 0.0, 0.9, 0.35, 0.01, help="Portion of the note duration spent gliding.")
            rdcol1, rdcol2 = st.columns(2, gap="small")
            with rdcol1:
                rest_prob = st.slider("Rest prob", 0.0, 0.5, 0.10, 0.01, help="Chance of rests vs notes.")
            with rdcol2:
                drone_level = st.slider("Root drone level", 0.0, 1.0, 0.0, 0.05, help="Subtle sustained drone on the root note beneath the melody.")
            icol1, icol2 = st.columns(2, gap="small")
            with icol1:
                imprint_gain = st.slider("Imprint gain", 0.0, 16.0, 8.0, 0.5,
                    help="Strength of harmonic emphasis on the pink noise.")
            with icol2:
                harmonics = st.slider("Harmonics", 0, 16, 10, 1,
                    help="Number of harmonic peaks in the pitch mask.")
            bw_frac = st.slider("BW frac", 0.002, 0.1, 0.01, 0.001,
                format="%.3f",
                help="Primary texture control. Narrow (0.005–0.01): tight pitch instruction, "
                     "can sound synthetic. Widen (0.02–0.05) to make the melody feel less "
                     "robotic without losing the harmonic steer. Go higher (0.05–0.1) for a "
                     "very diffuse, atmospheric texture.")
            melody_noise_floor_db = st.slider("Noise floor (dB)", -60.0, 0.0, 0.0, 1.0,
                help="Attenuation of non-harmonic pink noise. 0 dB = full noise floor (current "
                     "behavior). Lower values reduce broadband static while keeping harmonic "
                     "emphasis. Try –18 to –30 dB if the prompt is introducing extra noise into "
                     "your AI music output.")

        with st.expander("Vowel character", expanded=False):
            enable_formants = st.checkbox(
                "Imprint vowels",
                value=False,
                help=(
                    "Shape the melody imprint toward sung vowel sounds in the "
                    "selected language. Off (default) = a vowel-neutral buzz, "
                    "which gives the AI more freedom and often hallucinates "
                    "instruments better. On = biases the output toward "
                    "voice-like vowel color. This adds vowel resonances only — "
                    "no consonants, no words; it cannot make the AI sing "
                    "specific lyrics or guarantee a language."
                ),
            )
            _vowel_lang_options = ["English", "Japanese", "Spanish"]
            vowel_language_label = st.selectbox(
                "Language",
                options=_vowel_lang_options,
                index=0,
                help=(
                    "Vowel system used for the imprint. English: full vowels on "
                    "stressed notes, weak notes reduce to a neutral 'schwa' — the "
                    "rhythm signature of stress-timed English. Japanese: the five "
                    "vowels a-i-u-e-o on every note with no reduction (Japanese is "
                    "mora-timed and keeps every vowel full; its 'u' is the brighter "
                    "unrounded Japanese u). Spanish: the five Spanish vowels on "
                    "every note, no reduction (syllable-timed)."
                ),
            )
            vowel_language = vowel_language_label.lower()
            vfcol1, vfcol2 = st.columns(2, gap="small")
            with vfcol1:
                formant_strength = st.slider(
                    "Vowel strength", 0.0, 1.0, 0.6, 0.05,
                    help=(
                        "How hard the vowels are imprinted. Low = a hint the "
                        "model can override; high = strong vowel character that "
                        "can sound robotic if overdone. Start ~0.6 and tune by ear."
                    ),
                )
            with vfcol2:
                if vowel_language == "english":
                    accent_period = st.selectbox(
                        "Stress pattern",
                        options=[2, 3, 4],
                        index=0,
                        help=(
                            "English only. Accented notes get full vowels; the rest "
                            "reduce to a neutral 'schwa'. 2 = strong-weak (most "
                            "English-like), 3 = strong-weak-weak (more lilting). "
                            "Vowel reduction is the acoustic signature of "
                            "stress-timed English — Japanese and Spanish keep every "
                            "vowel full, so this control is hidden for them."
                        ),
                    )
                else:
                    accent_period = 2  # unused: these languages keep every vowel full
            st.caption(
                "A gentle acoustic bias, not a lyric engine — it colors the noise "
                "toward the language's vowel sounds, but the AI tool's own text "
                "prompt still has the biggest influence on language and words."
            )
    else:
        # Provide defaults when melody disabled to keep variables defined
        melody_root = "E"
        melody_scale = "minor_blues"
        bpm = 96
        _root_midi_base = NOTE_TO_MIDI.get(melody_root, 60) - 12
        low_midi, high_midi = _root_midi_base, _root_midi_base + 18
        step_bias, leap_steps, rest_prob = 0.5, 7, 0.10
        drone_level = 0.0
        glide_prob, glide_frac = 0.05, 0.35
        vib_hz, vib_depth = 5.5, 0.02
        melody_character, note_shape = "neutral", "natural"
        note_decay = 1.0
        enable_formants = False
        formant_strength = 0.6
        accent_period = 2
        vowel_language = "english"
        imprint_gain, harmonics, bw_frac = 8.0, 10, 0.01
        melody_noise_floor_db = 0.0

    # Add ~36px top margin before Drum header for clearer separation
    st.markdown("<div style='height: 36px'></div>", unsafe_allow_html=True)
    # Drum MIDI Imprint (left) — completely separate from melody module
    st.subheader("Drum MIDI Imprint")
    enable_drum = st.checkbox(
        "Enable drum MIDI imprint",
        value=False,
        help="Upload a MIDI drum track to generate a rhythm layer from band-limited pink noise.",
    )
    drum_midi_file = None
    drum_lane_gains = {}
    drum_bpm = int(bpm)
    match_melody_bpm = True
    loop_drums = True
    drum_trim_silence = True
    drum_timing_preview = None
    st.session_state.setdefault("drum_bpm_value", int(bpm))
    if enable_drum:
        drum_midi_file = st.file_uploader(
            "Drum MIDI (.mid / .midi)",
            type=["mid", "midi"],
            accept_multiple_files=False,
            help=(
                "Upload a General MIDI drum track exported from your DAW or drum sequencer. "
                "Use MIDI note numbers as the source of truth: kick 36, snare 38, closed hat 42. "
                "Your DAW may label those as C1/D1/F#1 or C2/D2/F#2 depending on octave numbering."
            ),
            key="drum_midi_file",
        )
        if drum_midi_file is not None:
            st.caption(t("Uploaded: {name}", name=drum_midi_file.name))
            drum_timing_preview = _inspect_uploaded_drum_midi(drum_midi_file)
            if drum_timing_preview is not None:
                st.caption(t(
                    "Detected drum MIDI: {bpm} BPM, {length} sec",
                    bpm=f"{drum_timing_preview['bpm']:g}",
                    length=f"{drum_timing_preview['length_s']:.2f}",
                ))
        else:
            st.session_state.pop("y_drum", None)
            st.session_state.pop("drum_bpm", None)
            st.session_state.pop("detected_drum_bpm_preview", None)
            st.session_state.pop("drum_timing_preview", None)
            st.session_state.pop("_drum_bpm_sig", None)

        dr1, dr2 = st.columns(2, gap="small")
        with dr1:
            kick_gain = st.slider(
                "Kick amount",
                0.0,
                2.0,
                1.0,
                0.05,
                help=(
                    "Gain for General MIDI kick notes 35/36. Your DAW may display MIDI 36 as "
                    "C1 or C2 depending on octave numbering."
                ),
            )
            snare_gain = st.slider(
                "Snare amount",
                0.0,
                2.0,
                0.9,
                0.05,
                help="Gain for General MIDI snare, clap, side stick, and tom notes routed to the snare noise lane.",
            )
        with dr2:
            hat_gain = st.slider(
                "Hat amount",
                0.0,
                2.0,
                0.7,
                0.05,
                help="Gain for General MIDI hi-hat and cymbal notes routed to the hat noise lane.",
            )
            perc_gain = st.slider(
                "Perc amount",
                0.0,
                2.0,
                0.5,
                0.05,
                help=(
                    "Gain for miscellaneous General MIDI percussion and any unmapped notes routed "
                    "to the perc noise lane."
                ),
            )
        drum_lane_gains = {
            "kick": kick_gain,
            "snare": snare_gain,
            "hat": hat_gain,
            "perc": perc_gain,
        }
        dt1, dt2 = st.columns(2, gap="small")
        with dt1:
            drum_character = st.selectbox(
                "Drum character",
                options=["clean", "tight", "deep", "bright", "breakbeat"],
                index=0,
                help=(
                    "Tonal preset for the drum layer. "
                    "Clean: balanced default. Tight: shorter envelopes and snappier transients. "
                    "Deep: bass-heavy low end with extra kick weight. Bright: more hat and upper-mid presence. "
                    "Breakbeat: pumped body, snappy snare, and light drive — groove timing stays intact."
                ),
            )
            snare_tune = st.slider(
                "Snare tune",
                -12,
                12,
                0,
                1,
                help="Shifts the snare lane's noise band in semitones before the drum layer is synthesized.",
            )
        with dt2:
            drum_decay = st.slider(
                "Drum decay",
                0.5,
                1.8,
                1.0,
                0.05,
                help="Shortens or lengthens all drum lane envelopes. Lower values are tighter; higher values ring longer.",
            )
        tempo1, tempo2 = st.columns(2, gap="small")
        with tempo1:
            match_melody_bpm = st.checkbox(
                "Match melody BPM",
                value=True,
                help="Use the melody BPM for the drum layer. Turn off to set an independent drum BPM.",
            )
            detected_bpm_for_reset = st.session_state.get("detected_drum_bpm_preview")
            if st.button(
                "Use detected BPM",
                disabled=detected_bpm_for_reset is None,
                help="Reset Independent drum BPM to the tempo detected in the uploaded MIDI file.",
            ):
                st.session_state["drum_bpm_value"] = int(round(float(detected_bpm_for_reset)))
        with tempo2:
            drum_bpm = st.slider(
                "Independent drum BPM",
                20,
                220,
                value=int(st.session_state.get("drum_bpm_value", bpm)),
                step=1,
                key="drum_bpm_value",
                disabled=match_melody_bpm,
                help=(
                    "Target BPM used only when Match melody BPM is off. "
                    "New uploads start at their detected MIDI BPM so you can quickly return to the original groove tempo."
                ),
            )
            drum_bpm = int(st.session_state.get("drum_bpm_value", drum_bpm))
        target_drum_bpm_preview = float(bpm) if match_melody_bpm else float(drum_bpm)
        matched_drum_seconds = _matched_drum_length_seconds(drum_timing_preview, target_drum_bpm_preview)
        if matched_drum_seconds is not None:
            st.caption(t(
                "Matched drum length: {length} sec at {bpm} BPM",
                length=f"{matched_drum_seconds:.2f}", bpm=f"{target_drum_bpm_preview:g}",
            ))
        loop_drums = st.checkbox(
            "Loop drums to prompt length",
            value=True,
            help=(
                "Repeats the drum MIDI when the prompt is longer than the MIDI region. "
                "Turn off to let the drums stop after the uploaded MIDI ends."
            ),
        )
        drum_trim_silence = st.checkbox(
            "Trim silence before first note",
            value=True,
            # Explicit key: the bass section has an identically labeled
            # checkbox, and in help-as-text mode (help stripped from the
            # widget) the auto-generated IDs would collide.
            key="drum_trim_silence_cb",
            help=(
                "Removes empty lead-in before the first drum hit "
                "(e.g. Logic session-player exports add a phantom bar). "
                "Turn off if your drums genuinely start with deliberate leading silence."
            ),
        )
    else:
        drum_lane_gains = {"kick": 1.0, "snare": 0.9, "hat": 0.7, "perc": 0.5}
        drum_character, snare_tune, drum_decay = "clean", 0, 1.0
        drum_trim_silence = True

    st.markdown("<div style='height: 36px'></div>", unsafe_allow_html=True)
    st.subheader("Bass MIDI Imprint")
    enable_bass = st.checkbox(
        "Enable bass MIDI imprint",
        value=False,
        help="Upload a MIDI bassline to generate a nuanced, pitch-sliding bass layer.",
    )
    bass_midi_file = None
    bass_timing_preview = None
    if enable_bass:
        bass_midi_file = st.file_uploader(
            "Bass MIDI (.mid / .midi)",
            type=["mid", "midi"],
            accept_multiple_files=False,
            help=(
                "Upload a bass track. Pitch bend and note velocity will be preserved and imprinted."
            ),
            key="bass_midi_file",
        )
        if bass_midi_file is not None:
            st.caption(t("Uploaded: {name}", name=bass_midi_file.name))
            bass_timing_preview = _inspect_uploaded_bass_midi(bass_midi_file)
            if bass_timing_preview is not None:
                st.caption(t(
                    "Detected bass MIDI: {bpm} BPM, {length} sec",
                    bpm=f"{bass_timing_preview['bpm']:g}",
                    length=f"{bass_timing_preview['length_s']:.2f}",
                ))
        else:
            st.session_state.pop("y_bass", None)
            st.session_state.pop("bass_timing_preview", None)
            st.session_state.pop("_bass_bpm_sig", None)

        bcol1, bcol2 = st.columns(2, gap="small")
        with bcol1:
            bass_character = st.selectbox(
                "Bass character",
                options=["Upright", "Fingerstyle", "Picked", "Synth", "Sub"],
                index=0,
                help=(
                    "Tonal preset for the bass imprint. "
                    "Upright: warm, pluck-like decay in the low-mid range (40–800 Hz). "
                    "Fingerstyle: balanced midrange tone (40–2000 Hz). "
                    "Picked: bright with lots of harmonics and strong attack transient (40–5000 Hz). "
                    "Sub: deep, smooth, low-frequency-only (30–200 Hz). "
                    "Synth: odd-harmonic reed-like character with smooth envelope (30–3000 Hz)."
                ),
            )
        with bcol2:
            bass_note_shape = st.selectbox(
                "Note shape base",
                options=["natural", "tight", "pluck", "smooth"],
                index=0,
                help="Base envelope shape before Decay offset and velocity are applied.",
            )
            
        bcol4, bcol5 = st.columns(2, gap="small")
        with bcol4:
            bass_decay_offset = st.slider("Decay offset", 0.1, 3.0, 1.0, 0.1, help="Scales the length of the bass note envelope decay.")
        with bcol5:
            bass_pb_range = st.slider("Pitch bend range", 1, 48, 12, 1, help="Matches the pitch wheel range of the virtual instrument that generated the MIDI (Logic slides often use 12, 24, or 48 semitones).")
            
        btempo1, btempo2 = st.columns(2, gap="small")
        with btempo1:
            match_melody_bpm_bass = st.checkbox(
                "Match melody BPM (Bass)",
                value=True,
                help="Use the melody BPM for the bass layer.",
            )
            detected_bpm_for_reset_bass = st.session_state.get("detected_bass_bpm_preview")
            if st.button(
                "Use detected BPM",
                disabled=detected_bpm_for_reset_bass is None,
                help="Reset Independent bass BPM to the tempo detected in the uploaded MIDI file.",
                key="btn_bass_bpm_reset"
            ):
                st.session_state["bass_bpm_value"] = int(round(float(detected_bpm_for_reset_bass)))
        with btempo2:
            bass_bpm = st.slider(
                "Independent bass BPM",
                20,
                220,
                value=int(st.session_state.get("bass_bpm_value", bpm)),
                step=1,
                key="bass_bpm_value",
                disabled=match_melody_bpm_bass,
                help=(
                    "Target BPM used only when Match melody BPM is off. "
                    "New uploads start at their detected MIDI BPM so you can quickly return to the original groove tempo."
                ),
            )
            bass_bpm = int(st.session_state.get("bass_bpm_value", bass_bpm))
            
        target_bass_bpm_preview = float(bpm) if match_melody_bpm_bass else float(bass_bpm)
        matched_bass_seconds = _matched_drum_length_seconds(bass_timing_preview, target_bass_bpm_preview)
        if matched_bass_seconds is not None:
            st.caption(t(
                "Matched bass length: {length} sec at {bpm} BPM",
                length=f"{matched_bass_seconds:.2f}", bpm=f"{target_bass_bpm_preview:g}",
            ))
            
        loop_bass = st.checkbox(
            "Loop bass to prompt length",
            value=True,
            help="Repeats the bass MIDI when the prompt is longer than the MIDI region.",
        )
        bass_trim_silence = st.checkbox(
            "Trim silence before first note",
            value=True,
            key="bass_trim_silence_cb",  # see drum twin: avoids duplicate auto-ID
            help=(
                "Removes empty lead-in before the first note "
                "(e.g. Logic session-player exports add a phantom bar). "
                "Preserves an intentional pickup or rest that's written on "
                "the note itself. Turn off if your bass should start with "
                "deliberate leading silence."
            ),
        )
        with st.expander("Bass – Advanced", expanded=False):
            bacol1, bacol2 = st.columns(2, gap="small")
            with bacol1:
                bass_imprint_gain = st.slider("Bass imprint gain", 0.0, 16.0, 8.0, 0.5,
                    help="Strength of harmonic emphasis on the bass pitch mask.")
            with bacol2:
                bass_bw_frac = st.slider("Bass BW frac", 0.002, 0.05, 0.01, 0.001,
                    format="%.3f",
                    help="Relative bandwidth around each harmonic. Wider = looser, "
                         "less synthetic-sounding pitch lock.")
            bass_noise_floor_db = st.slider("Bass noise floor (dB)", -60.0, 0.0, 0.0, 1.0,
                help="Attenuation of non-harmonic noise within the bass focus band. "
                     "0 dB = no attenuation (current behavior). Lower values reduce "
                     "residual static inside the bass frequency window. Less impactful "
                     "than melody noise floor since the bass focus band is already narrow.")
    else:
        bass_character, bass_note_shape = "Upright", "natural"
        bass_decay_offset, bass_pb_range = 1.0, 12
        match_melody_bpm_bass, loop_bass = True, True
        bass_trim_silence = True
        bass_bpm = int(bpm)
        bass_imprint_gain, bass_bw_frac = 8.0, 0.01
        bass_noise_floor_db = 0.0

    # Add ~36px top margin before Focus header for clearer separation
    st.markdown("<div style='height: 36px'></div>", unsafe_allow_html=True)
    # Focus (left) — for future tabs, keep flat containers
    st.subheader("Focus")
    fcol1, fcol2 = st.columns([1,1])
    with fcol1:
        enable_focus = st.checkbox(
            "Enable focus band",
            value=False,
            help="Emphasize energy in a vocal/guitar/bass band or a custom Hz range.",
        )
    with fcol2:
        tame_low_end = st.checkbox(
            "Bass Roll-Off",
            value=True,
            help=(
                "Applies a 25 Hz high-pass filter to the generated prompt noise before melody/focus imprinting. "
                "This removes sub-bass rumble from the prompt layer only; it does not EQ your uploaded audio."
            ),
        )
    if enable_focus:
        focus_preset = st.radio(
            "Preset",
            options=["vocal", "guitar", "bass", "custom"],
            index=0,
            horizontal=True,
            help="Choose a preset band or select ‘custom’ to set your own Hz range.",
        )
    else:
        focus_preset = "none"
    # Only show Advanced controls when focus is enabled; otherwise use defaults
    if enable_focus:
        with st.expander("Focus Band – Advanced", expanded=(focus_preset == "custom")):
            if focus_preset == "custom":
                band = st.slider("Focus Hz band", 20, 20000, (120, 3200), step=10, help="Twin‑handle slider: low/high cutoff in Hz.")
            else:
                band = None
            colf4, colf5 = st.columns(2)
            with colf4:
                floor_db = st.slider("Band floor (dB)", -36, 0, -18, 1, help="Attenuation outside the focus band.")
            with colf5:
                sharpness = st.slider("Band edge sharpness", 6, 24, 12, 1, help="Sigmoid steepness of the band rolloff. 6 = gentle slope; 12 = moderate; 24 ≈ near-brick-wall. Dimensionless — not dB/Hz.")
    else:
        band = None
        floor_db, sharpness = -18.0, 12


with right:
    # Reserve a container at the top for Generate so it's visually above
    gen_top = st.container()

    # Collect params for generation
    melody_params = dict(
        bpm=bpm,
        root=melody_root,
        scale=melody_scale,
        low_midi=low_midi,
        high_midi=high_midi,
        step_bias=step_bias,
        leap_steps=leap_steps,
        rest_prob=rest_prob,
        drone_level=float(drone_level),
        durations=(0.25, 0.5, 1.0),
        duration_probs=(0.25, 0.5, 0.25),
        glide_prob=glide_prob,
        glide_frac=glide_frac,
        vib_hz=float(vib_hz),
        vib_depth=float(vib_depth),
    )
    character_params = dict(
        character="neutral" if melody_character == "wide" else melody_character,
        note_shape=note_shape,
        note_decay=float(note_decay),
        detune_spread_cents=7.0 if melody_character == "wide" else 0.0,
        enable_formants=bool(enable_formants),
        formant_strength=float(formant_strength),
        accent_period=int(accent_period),
        vowel_language=str(vowel_language),
    )
    focus_params = dict(preset=focus_preset if focus_preset != "none" else None, band=band)
    imprint_params = dict(gain=imprint_gain, harmonics=harmonics, bw_frac=bw_frac,
                          floor_db=floor_db, sharpness=sharpness, n_fft=2048,
                          melody_noise_floor_db=melody_noise_floor_db)
    # Low-end config dict for core processing
    hpf_taps = 2049  # Steep by default
    lowend_cfg = dict(
        tame_low_end=bool(tame_low_end),
        hpf_cutoff_hz=25,
        hpf_taps=int(hpf_taps),
        mono_lows=True,
        mono_cutoff_hz=120,
    )

    # Render Generate controls at the top of the right column
    with gen_top:
        st.subheader("Generate")
        st.markdown("<div class='gen-row'>", unsafe_allow_html=True)
        btn_col, seed_col = st.columns([3,1], gap="medium")
        with btn_col:
            # Placeholder: the button is rendered further down, once the
            # resolved prompt length is known, so it can be disabled when an
            # active AUDIOPROMPT_MAX_SECONDS cap is exceeded. Visual position
            # in the page is unchanged.
            gen_btn_slot = st.empty()
        with seed_col:
            seed = st.number_input(
                "Seed",
                min_value=-1,
                max_value=10_000_000,
                value=-1,
                step=1,
                # Compact layout paints the label via CSS; help-as-text mode
                # shows the real label so screen readers get an accessible name.
                label_visibility="visible" if i18n.help_as_text() else "collapsed",
                help="Controls randomness for pink noise and the melody (notes, glides, etc.). Set to -1 to use a new random seed each generation.",
                key="seed",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        # Debug logs are always sent to server stdout via _log_debug();
        # no UI toggle needed to keep the interface clean.

        # Output & Seed just beneath Generate (no separate heading to reduce clutter)
        can_match_drum_length = enable_drum and drum_midi_file is not None and matched_drum_seconds is not None
        can_match_bass_length = enable_bass and bass_midi_file is not None and matched_bass_seconds is not None
        
        prompt_len_opts = ["Manual seconds"]
        if can_match_drum_length:
            prompt_len_opts.insert(0, "Match drum MIDI")
        if can_match_bass_length:
            prompt_len_opts.insert(0, "Match bass MIDI")
            
        _cap_note = (
            t(" Note: the public demo caps prompt length at {cap} s.", cap=f"{MAX_PROMPT_SECONDS:g}")
            if MAX_PROMPT_SECONDS is not None
            else ""
        )
        if len(prompt_len_opts) > 1:
            prompt_length_mode = st.radio(
                "Prompt length",
                options=prompt_len_opts,
                index=0,
                horizontal=True,
                help=t("Match length to uploaded MIDI regions (adjusted to target BPM) or use manual seconds.") + _cap_note,
            )
        else:
            prompt_length_mode = "Manual seconds"

        manual_prompt_seconds = st.slider(
            "Manual seconds" if len(prompt_len_opts) > 1 else "Prompt seconds",
            1.0,
            20.0,
            4.0,
            0.5,
            disabled=prompt_length_mode != "Manual seconds",
            help=t(
                "Length of the generated prompt when Prompt length is set to Manual seconds."
                if len(prompt_len_opts) > 1
                else "Length of the generated prompt."
            ) + _cap_note,
        )
        if prompt_length_mode == "Match drum MIDI" and matched_drum_seconds is not None:
            prompt_seconds = float(matched_drum_seconds)
            st.caption(t("Prompt length matched to drum MIDI: {length} sec", length=f"{prompt_seconds:.2f}"))
        elif prompt_length_mode == "Match bass MIDI" and matched_bass_seconds is not None:
            prompt_seconds = float(matched_bass_seconds)
            st.caption(t("Prompt length matched to bass MIDI: {length} sec", length=f"{prompt_seconds:.2f}"))
        else:
            prompt_seconds = float(manual_prompt_seconds)

        # Enforce the optional public-demo cap against the resolved prompt
        # length for whichever length mode is selected — checked before any
        # DSP runs, using the already-detected MIDI lengths. No truncation:
        # the Generate button is disabled so an over-cap render can't start.
        cap_exceeded = (
            MAX_PROMPT_SECONDS is not None
            and float(prompt_seconds) > float(MAX_PROMPT_SECONDS) + 1e-6
        )
        if cap_exceeded:
            st.warning(t(
                "This prompt would be {length}s — the public demo caps prompts at "
                "{cap}s (plenty for a 16-bar loop even at 70 BPM). Trim your MIDI to "
                "{cap}s or shorter, or run the app yourself for unlimited length: "
                "https://github.com/rwolt/audioprompt",
                length=f"{prompt_seconds:.1f}", cap=f"{MAX_PROMPT_SECONDS:g}",
            ))
        with gen_btn_slot:
            pressed = st.button(
                "Generate Prompt", type="primary", width="stretch", disabled=cap_exceeded
            )
        if pressed and cap_exceeded:
            pressed = False  # belt-and-braces: never start an over-cap render
        st.markdown(t("**Prompt Level**"))
        ag_col1, ag_col2 = st.columns([1,1])
        with ag_col1:
            auto_gain = st.checkbox(
                "Auto gain (match input audio)",
                value=False,
                help="Detect input audio loudness and set the prompt level automatically (RMS in dBFS)."
            )
        with ag_col2:
            auto_gain_offset_db = st.slider(
                "Prompt relative to input (dB)",
                -12.0, 6.0, -3.0, 0.5,
                help="Target prompt loudness relative to input RMS (e.g., −3 dB makes the prompt slightly quieter).",
                disabled=not auto_gain,
            )
        # Manual gain and fades
        colo1, colo2, colo3 = st.columns(3)
        with colo1:
            prompt_gain_db = st.slider(
                "Prompt gain (dB)",
                -24.0, 6.0, -3.0, 0.5,
                help="Level for the prepended prompt.",
                disabled=auto_gain,
            )
        with colo2:
            fade_in_ms = st.slider("Fade-in (ms)", 0, 200, 10, 1, help="Smooth ramp at the start.")
        with colo3:
            fade_out_ms = st.slider("Fade-out (ms)", 0, 500, 50, 1, help="Smooth ramp at the end.")

        # Blend controls — always visible so the user can tune the mix
        st.markdown(t("**Layer Blend**"))
        bl1, bl2, bl3 = st.columns(3, gap="small")
        with bl1:
            melody_blend_gain = st.slider(
                "Melody level",
                0.0, 2.0, 1.0, 0.05,
                help="Gain for the melody / focus / pink-noise layer, applied after loudness-matching all layers to the same RMS level.",
            )
        with bl2:
            drum_blend_gain = st.slider(
                "Drum level",
                0.0, 2.0, 1.0, 0.05,
                help="Gain for the drum MIDI layer, applied after loudness-matching all layers to the same RMS level.",
            )
        with bl3:
            bass_blend_gain = st.slider(
                "Bass level",
                0.0, 2.0, 1.0, 0.05,
                help="Gain for the bass MIDI layer, applied after loudness-matching all layers to the same RMS level.",
            )

        # (Outputs header and preview toggle appear below, just before outputs)

        # Generate immediately after controls so results are available below in this same run
        # Only generate when the button is pressed (no auto-generate on first load)
        if pressed:
            with st.spinner("Generating prompt..."):
                # Clear old outputs first to reduce memory before regenerating
                _clear_outputs()
                # Resolve seed: -1 means random each generation
                seed_input = int(st.session_state.get("seed", 7))
                seed_to_use = int(np.random.randint(0, 10_000_000)) if seed_input == -1 else seed_input
                _log_debug(f"[gen] seed={seed_to_use}; mem_before={_mem_usage_mb()} MB")
                # Melody prompt
                y_prompt, events = build_prompt(
                    sr=int(sr),
                    prompt_seconds=float(prompt_seconds),
                    seed=seed_to_use,
                    enable_melody=bool(enable_melody),
                    melody_params=melody_params,
                    enable_focus=bool(enable_focus),
                    focus_params=focus_params,
                    enable_gate=bool(enable_gate),
                    imprint_params=imprint_params,
                    lowend_cfg=lowend_cfg,
                    character_params=character_params,
                )
                # If neither melody nor focus is on, silence the melody layer
                # so drum-only output is truly just the drum layer.
                if not enable_melody and not enable_focus and enable_drum:
                    y_prompt = np.zeros_like(y_prompt)
                    events = None
                st.session_state["y_prompt"], st.session_state["events"] = y_prompt, events
                st.session_state["seed_used"] = seed_to_use
                st.session_state["prompt_sr"] = int(sr)
                _log_debug(f"[gen] melody prompt_len={len(y_prompt)} samples @ {int(sr)} Hz; mem_after={_mem_usage_mb()} MB")

                # Drum MIDI prompt
                drum_seed_used = None
                if enable_drum and drum_midi_file is not None:
                    try:
                        drum_seed_used = seed_to_use + 1
                        drum_lanes, detected_drum_bpm = parse_midi_drum_events(drum_midi_file.getvalue(), trim_leading_silence=bool(drum_trim_silence))
                        target_drum_bpm = float(bpm) if match_melody_bpm else float(drum_bpm)
                        drum_speed_mult = target_drum_bpm / max(float(detected_drum_bpm), 1e-6)
                        _log_debug(
                            f"[gen] drum MIDI parsed: {detected_drum_bpm} BPM; target={target_drum_bpm:.1f} BPM"
                        )
                        if abs(float(drum_speed_mult) - 1.0) > 1e-6:
                            drum_lanes = scale_lane_events(drum_lanes, float(drum_speed_mult))
                            _log_debug(f"[gen] drum BPM scaled by {drum_speed_mult:.3f}x")
                        if loop_drums:
                            drum_lanes = loop_lane_events(drum_lanes, float(prompt_seconds))
                            _log_debug("[gen] drum events looped to prompt length")
                        drum_lane_params, drum_drive = build_drum_tone_params(
                            drum_character,
                            sr=int(sr),
                            snare_tune_semitones=float(snare_tune),
                            decay_mult=float(drum_decay),
                        )
                        y_drum = synthesize_drum_layer(
                            drum_lanes,
                            sr=int(sr),
                            prompt_seconds=float(prompt_seconds),
                            seed=drum_seed_used,  # correlated but not identical
                            lane_gains=drum_lane_gains,
                            lane_params=drum_lane_params,
                            master_gain=1.0,  # full-level; blend gain applied at render time
                            drive=drum_drive,
                        )
                        st.session_state["y_drum"] = y_drum
                        st.session_state["drum_bpm"] = round(float(target_drum_bpm), 1)
                        st.session_state["detected_drum_bpm"] = round(float(detected_drum_bpm), 1)
                        _log_debug(f"[gen] drum layer: len={len(y_drum)}, peak={np.max(np.abs(y_drum)):.4f}")
                    except Exception as e:
                        _log_debug(f"[gen] drum error: {e}")
                        st.error(t("Failed to parse drum MIDI. Make sure you uploaded a valid .mid/.midi file.\nError: {err}", err=e))
                        st.session_state.pop("y_drum", None)
                        drum_seed_used = None
                else:
                    st.session_state.pop("y_drum", None)
                    st.session_state.pop("drum_bpm", None)
                    st.session_state.pop("detected_drum_bpm", None)
                    
                # Bass MIDI prompt
                bass_seed_used = None
                if enable_bass and bass_midi_file is not None:
                    try:
                        bass_seed_used = seed_to_use + 2
                        bass_events, bass_bends, detected_bass_bpm = parse_midi_bass_events(
                            bass_midi_file.getvalue(),
                            pitch_bend_range=float(bass_pb_range),
                            trim_leading_silence=bool(bass_trim_silence),
                        )
                        target_bass_bpm = float(bpm) if match_melody_bpm_bass else float(bass_bpm)
                        bass_speed_mult = target_bass_bpm / max(float(detected_bass_bpm), 1e-6)
                        _log_debug(
                            f"[gen] bass MIDI parsed: {detected_bass_bpm} BPM; target={target_bass_bpm:.1f} BPM"
                        )
                        if abs(float(bass_speed_mult) - 1.0) > 1e-6:
                            bass_events, bass_bends = scale_bass_events(bass_events, bass_bends, float(bass_speed_mult))
                            _log_debug(f"[gen] bass BPM scaled by {bass_speed_mult:.3f}x")
                        if loop_bass:
                            bass_events, bass_bends = loop_bass_events(bass_events, bass_bends, float(prompt_seconds))
                            _log_debug("[gen] bass events looped to prompt length")
                            
                        # Generate pitch trajectory from bass events
                        n_samples = int(sr * prompt_seconds)
                        bass_f0 = bass_events_to_f0(bass_events, bass_bends, sr, n_samples)
                        
                        # Apply timbre based on preset
                        bass_char_params = {"character": "neutral", "note_shape": bass_note_shape, "note_decay": float(bass_decay_offset)}
                        bass_imprint_params = {"gain": bass_imprint_gain, "harmonics": 10, "bw_frac": bass_bw_frac, "floor_db": -24.0, "sharpness": 12, "n_fft": 2048}
                        bass_focus = (40, 800)
                        
                        if bass_character == "Upright":
                            bass_focus = (40, 800)
                            bass_char_params["character"] = "warm"
                            bass_char_params["note_shape"] = bass_note_shape if bass_note_shape != "natural" else "pluck"
                            bass_imprint_params["harmonics"] = 6
                        elif bass_character == "Picked":
                            bass_focus = (40, 5000)
                            bass_char_params["character"] = "bright"
                            bass_char_params["note_shape"] = bass_note_shape if bass_note_shape != "natural" else "tight"
                            bass_imprint_params["harmonics"] = 15
                        elif bass_character == "Fingerstyle":
                            bass_focus = (40, 2000)
                            bass_char_params["character"] = "neutral"
                            bass_char_params["note_shape"] = bass_note_shape if bass_note_shape != "natural" else "natural"
                            bass_imprint_params["harmonics"] = 8
                        elif bass_character == "Sub":
                            bass_focus = (30, 200)
                            bass_char_params["character"] = "warm"
                            bass_char_params["note_shape"] = bass_note_shape if bass_note_shape != "natural" else "smooth"
                            bass_imprint_params["harmonics"] = 3
                        elif bass_character == "Synth":
                            bass_focus = (30, 3000)
                            bass_char_params["character"] = "reed"
                            bass_char_params["note_shape"] = bass_note_shape if bass_note_shape != "natural" else "smooth"
                            bass_imprint_params["harmonics"] = 12
                            
                        x_bass = pink_noise(n_samples, sr, seed=bass_seed_used)
                        y_bass = imprint_melody_focus(
                            x_bass,
                            sr,
                            f0_hz=bass_f0,
                            gain=bass_imprint_params["gain"],
                            harmonics=bass_imprint_params["harmonics"],
                            bw_frac=bass_imprint_params["bw_frac"],
                            focus=bass_focus,
                            band_floor_db=bass_imprint_params["floor_db"],
                            sharpness=bass_imprint_params["sharpness"],
                            n_fft=bass_imprint_params["n_fft"],
                            character=bass_char_params["character"],
                            melody_noise_floor_db=float(bass_noise_floor_db),
                        )
                        
                        # Apply velocity-sensitive rhythmic gate
                        bass_gate = rhythmic_gate_from_events(
                            bass_events, sr, n_samples=n_samples, shape=bass_char_params["note_shape"], decay_mult=bass_char_params["note_decay"]
                        )
                        y_bass = (y_bass * (0.05 + 0.95 * bass_gate)).astype(np.float32)
                        
                        # Normalize bass layer
                        peak = float(np.max(np.abs(y_bass)) + 1e-12)
                        if peak > 0:
                            y_bass = (y_bass / peak).astype(np.float32)
                            
                        st.session_state["y_bass"] = y_bass
                        st.session_state["bass_bpm"] = round(float(target_bass_bpm), 1)
                        st.session_state["detected_bass_bpm"] = round(float(detected_bass_bpm), 1)
                        _log_debug(f"[gen] bass layer: len={len(y_bass)}, peak={np.max(np.abs(y_bass)):.4f}")
                        # Free full-length intermediates; y_bass lives on in session_state
                        del x_bass, bass_f0, bass_gate
                    except Exception as e:
                        _log_debug(f"[gen] bass error: {e}")
                        st.error(t("Failed to parse bass MIDI. Make sure you uploaded a valid .mid/.midi file.\nError: {err}", err=e))
                        st.session_state.pop("y_bass", None)
                        bass_seed_used = None
                else:
                    st.session_state.pop("y_bass", None)
                    st.session_state.pop("bass_bpm", None)
                    st.session_state.pop("detected_bass_bpm", None)

                st.session_state["download_meta"] = {
                    "root": melody_root if enable_melody else "none",
                    "scale": _scale_tag(melody_scale, bool(enable_melody)),
                    "melody_bpm": bpm,
                    "melody_seed": seed_to_use,
                    "drum_bpm": target_drum_bpm if drum_seed_used is not None else None,
                    "drum_seed": drum_seed_used,
                    "bass_bpm": target_bass_bpm if bass_seed_used is not None else None,
                    "bass_seed": bass_seed_used,
                    "length_s": prompt_seconds,
                    "focus": _focus_tag(
                        bool(enable_focus),
                        focus_params.get("preset"),
                        band if focus_params.get("preset") is None else None,
                    ),
                    "character": melody_character,
                    "vowel": (
                        _vowel_tag(vowel_language, formant_strength, accent_period)
                        if enable_melody and enable_formants else None
                    ),
                    "bw_frac": bw_frac if enable_melody else None,
                }
                gc.collect()
                _log_debug(f"[gen] done; mem={_mem_usage_mb()} MB")
            # Text status so completion is announced to screen readers too
            st.success(t("Prompt generated — {length} sec", length=f"{prompt_seconds:.2f}"))

        # Outputs header
        st.subheader("Outputs")

        # Audio and download buttons directly under Outputs
        if "y_prompt" in st.session_state:
            # Build prompt WAV using stored prompt SR
            y_prompt_local = st.session_state["y_prompt"]
            sr_prompt_local = int(st.session_state.get("prompt_sr", int(sr)))

            # RMS-normalize each layer to TARGET_BLEND_RMS before applying blend gains,
            # so gain sliders at equal values produce comparable perceived loudness
            # regardless of each layer's internal peak/density characteristics.
            y_render = (_normalize_to_rms(y_prompt_local) * float(melody_blend_gain)).astype(np.float32)

            if "y_drum" in st.session_state:
                y_drum_local = _normalize_to_rms(st.session_state["y_drum"])
                min_len = min(len(y_render), len(y_drum_local))
                y_render = y_render[:min_len] + (y_drum_local[:min_len] * float(drum_blend_gain))

            if "y_bass" in st.session_state:
                y_bass_local = _normalize_to_rms(st.session_state["y_bass"])
                min_len = min(len(y_render), len(y_bass_local))
                y_render = y_render[:min_len] + (y_bass_local[:min_len] * float(bass_blend_gain))
                
            peak = float(np.max(np.abs(y_render)) + 1e-12)
            if peak > 0:
                y_render = (y_render / peak).astype(np.float32)
            _log_debug(f"[render] final blend peak after norm={np.max(np.abs(y_render)):.4f}")

            # File naming for prompt/combined
            seed_val_local = int(st.session_state.get("seed_used", st.session_state.get("seed", 7)))
            base_stem_local = Path(uploaded.name).stem if uploaded is not None else None
            download_meta = st.session_state.get(
                "download_meta",
                {
                    "root": melody_root if enable_melody else "none",
                    "scale": _scale_tag(melody_scale, bool(enable_melody)),
                    "melody_bpm": bpm,
                    "melody_seed": seed_val_local,
                    "drum_bpm": st.session_state.get("drum_bpm") if "y_drum" in st.session_state else None,
                    "drum_seed": seed_val_local + 1 if "y_drum" in st.session_state else None,
                    "bass_bpm": st.session_state.get("bass_bpm") if "y_bass" in st.session_state else None,
                    "bass_seed": seed_val_local + 2 if "y_bass" in st.session_state else None,
                    "length_s": prompt_seconds,
                    "focus": _focus_tag(
                        bool(enable_focus),
                        focus_params.get("preset"),
                        band if focus_params.get("preset") is None else None,
                    ),
                    "character": melody_character,
                    "vowel": (
                        _vowel_tag(vowel_language, formant_strength, accent_period)
                        if enable_melody and enable_formants else None
                    ),
                    "bw_frac": bw_frac if enable_melody else None,
                },
            )
            prompt_only_name_local = _download_name("prompt", base_stem_local, download_meta)

            # Prompt audio + download (render below in this column)
            # Reuse cached prompt bytes if parameters unchanged
            st.caption(t("Seed used: {seed}", seed=seed_val_local))
            if "y_drum" in st.session_state:
                drum_bpm_display = st.session_state.get("drum_bpm", "—")
                detected_bpm_display = st.session_state.get("detected_drum_bpm")
                if detected_bpm_display is not None:
                    st.caption(t("Drum layer: {bpm} BPM (detected {detected} BPM)",
                                 bpm=drum_bpm_display, detected=detected_bpm_display))
                else:
                    st.caption(t("Drum layer: {bpm} BPM", bpm=drum_bpm_display))
            if "y_bass" in st.session_state:
                bass_bpm_display = st.session_state.get("bass_bpm", "—")
                detected_bass_bpm_display = st.session_state.get("detected_bass_bpm")
                if detected_bass_bpm_display is not None:
                    st.caption(t("Bass layer: {bpm} BPM (detected {detected} BPM)",
                                 bpm=bass_bpm_display, detected=detected_bass_bpm_display))
                else:
                    st.caption(t("Bass layer: {bpm} BPM", bpm=bass_bpm_display))
            st.markdown(t("**Prompt**"))
            prompt_key_now = (
                "prompt",
                int(sr_prompt_local),
                int(seed_val_local),
                bool(enable_drum),
                bool(enable_bass),
                round(float(melody_blend_gain), 3),
                round(float(drum_blend_gain), 3),
                round(float(bass_blend_gain), 3),
                round(float(st.session_state.get("drum_bpm", drum_bpm)), 3),
                round(float(st.session_state.get("bass_bpm", bass_bpm)), 3),
                bool(loop_drums),
                bool(loop_bass),
                round(float(prompt_seconds), 3),
            )
            if st.session_state.get("prompt_key") != prompt_key_now or "prompt_bytes" not in st.session_state:
                prompt_wav_local = wav_bytes(y_render, sr_prompt_local)
                st.session_state["prompt_bytes"] = prompt_wav_local
                st.session_state["prompt_key"] = prompt_key_now
                _log_debug(f"[prompt] wav_bytes={len(prompt_wav_local)/1e6:.2f} MB; mem={_mem_usage_mb()} MB (encoded)")
            else:
                prompt_wav_local = st.session_state["prompt_bytes"]
                _log_debug(f"[prompt] reused cache; mem={_mem_usage_mb()} MB")
            st.markdown("<div class='dl-row'>", unsafe_allow_html=True)
            ap_col_audio, ap_col_btn = st.columns([4,1], gap="small")
            with ap_col_audio:
                st.audio(prompt_wav_local, format="audio/wav")
            with ap_col_btn:
                st.download_button(
                    "Download Prompt",
                    data=prompt_wav_local,
                    file_name=prompt_only_name_local,
                    mime="audio/wav",
                )
            st.markdown("</div>", unsafe_allow_html=True)
            # Keep cached bytes; no local deletes here

            # Combined audio + download (if input provided)
            if uploaded is not None:
                try:
                    x_local, sr_in_local = load_audio_mono(uploaded, int(sr))
                except Exception as e:
                    st.error(t(
                        "Failed to read input audio. Prefer WAV/FLAC/OGG. "
                        "MP3 support depends on your libsndfile build.\n"
                        "Error: {err}", err=e,
                    ))
                    x_local = None
                if x_local is not None:
                    target_len_local = int(round(float(prompt_seconds) * int(sr)))
                    prompt_local = y_render
                    # Resample prompt if SR differs
                    if sr_prompt_local != int(sr):
                        g_local = np.gcd(sr_prompt_local, int(sr))
                        prompt_local = resample_poly(prompt_local, int(sr) // g_local, sr_prompt_local // g_local).astype(np.float32)
                    prompt_local = prompt_local[:target_len_local]
                    # Auto/Manual gain
                    if auto_gain:
                        try:
                            seed_rms_db_local = _rms_dbfs(x_local)
                            prompt_rms_db_local = _rms_dbfs(prompt_local[:target_len_local]) if target_len_local > 0 else _rms_dbfs(prompt_local)
                            desired_db_local = seed_rms_db_local + float(auto_gain_offset_db)
                            gain_db_local = desired_db_local - prompt_rms_db_local
                        except Exception:
                            gain_db_local = float(prompt_gain_db)
                    else:
                        gain_db_local = float(prompt_gain_db)
                    gain_local = 10 ** (gain_db_local / 20.0)
                    prompt_local = apply_fades(prompt_local * gain_local, int(sr), int(fade_in_ms), int(fade_out_ms))

                    # Build or reuse Combined WAV without allocating a large concatenated array
                    combined_key_now = (
                        "combined",
                        int(sr), float(prompt_seconds), int(seed_val_local),
                        bool(auto_gain), float(auto_gain_offset_db) if auto_gain else float(prompt_gain_db),
                        int(fade_in_ms), int(fade_out_ms),
                        getattr(uploaded, "name", None), getattr(uploaded, "size", None),
                        bool(enable_drum),
                        bool(enable_bass),
                        round(float(melody_blend_gain), 3),
                        round(float(drum_blend_gain), 3),
                        round(float(bass_blend_gain), 3),
                        round(float(st.session_state.get("drum_bpm", drum_bpm)), 3),
                        round(float(st.session_state.get("bass_bpm", bass_bpm)), 3),
                        bool(loop_drums),
                        bool(loop_bass),
                        round(float(prompt_seconds), 3),
                    )
                    if st.session_state.get("combined_key") != combined_key_now or "combined_bytes" not in st.session_state:
                        combined_wav_local = wav_bytes_concat_segments([prompt_local, x_local.astype(np.float32, copy=False)], int(sr))
                        st.session_state["combined_bytes"] = combined_wav_local
                        st.session_state["combined_key"] = combined_key_now
                        _log_debug(f"[combined] built; wav_bytes={len(combined_wav_local)/1e6:.2f} MB; mem={_mem_usage_mb()} MB")
                    else:
                        combined_wav_local = st.session_state["combined_bytes"]
                        _log_debug(f"[combined] reused cache; mem={_mem_usage_mb()} MB")
                    st.markdown(t("**Combined**"))
                    st.markdown("<div class='dl-row'>", unsafe_allow_html=True)
                    cap_col_audio, cap_col_btn = st.columns([4,1], gap="small")
                    with cap_col_audio:
                        st.audio(combined_wav_local, format="audio/wav")
                    with cap_col_btn:
                        combined_name_local = _download_name("combined", Path(uploaded.name).stem, download_meta)
                        st.download_button(
                            "Download Combined",
                            data=combined_wav_local,
                            file_name=combined_name_local,
                            mime="audio/wav",
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                    try:
                        del prompt_local
                        del x_local
                    except Exception:
                        pass
                    gc.collect()
                    _log_debug(f"[combined] locals cleared; mem={_mem_usage_mb()} MB")
            # Free per-render blend buffers: the encoded WAV bytes are cached in
            # session_state, and y_prompt/y_drum/y_bass stay there for re-blending,
            # so these full-length arrays are no longer needed this run.
            try:
                del y_render
            except Exception:
                pass
            try:
                del y_drum_local
            except Exception:
                pass
            try:
                del y_bass_local
            except Exception:
                pass
            gc.collect()
        # Blue placeholders when outputs are not ready (pill style)
        if "y_prompt" not in st.session_state:
            st.markdown(f"<div class='spec-placeholder'>{t('Set your parameters and press Generate Prompt.')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='spec-placeholder'>{t('No input file uploaded; only the prompt is generated.')}</div>", unsafe_allow_html=True)
        elif uploaded is None:
            st.markdown(f"<div class='spec-placeholder'>{t('No input file uploaded; only the prompt is generated.')}</div>", unsafe_allow_html=True)

# Footer: brief Terms & Privacy notice (public hosting)
st.markdown("---")
st.markdown(
    f"<div class='footer-note'>{i18n.block('footer_terms')}</div>",
    unsafe_allow_html=True,
)

# Inject drag-over highlight script at the end to avoid top spacer
st.markdown("<div class='js-hook'></div>", unsafe_allow_html=True)
components.html(_DRAG_JS, height=0)

# Analytics only when the operator has configured an endpoint (see GOATCOUNTER_URL above)
if GOATCOUNTER_URL:
    components.html(_goatcounter_js(GOATCOUNTER_URL), height=0)
