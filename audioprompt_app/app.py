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


# Import core from ./src (ensure our local package takes precedence)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# Import directly from submodules to avoid package-level side effects
from audioprompt_core.audio import (
    load_audio_mono,
    apply_fades,
    wav_bytes,
    wav_bytes_concat_segments,
    tag_suffix,
)
from audioprompt_core.prompt import (
    pink_noise,
    imprint_melody_focus,
    rhythmic_gate_from_events,
    apply_hpf,
    apply_mono_lows,
)
from audioprompt_core.melody import (
    SCALES,
    generate_random_melody,
    events_to_f0,
)


st.set_page_config(page_title="AudioPrompt", layout="wide")


# ------------------------------ Debug helpers ------------------------------ #
def _mem_usage_mb() -> float | None:
    """Best-effort current RSS in MB.

    Prefers psutil. On Linux without psutil, reads /proc/self/status (VmRSS).
    Falls back to resource.getrusage only as a last resort (note: that's peak).
    """
    # psutil preferred (current RSS)
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
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
            return float(rss) / (1024**2)  # bytes -> MB
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


st.markdown(
    """
    <style>
    .block-container{padding-top:1rem;padding-bottom:0.75rem; max-width:80vw; margin-left:auto; margin-right:auto;}
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
    /* Seed: keep row height equal to button; overlay label via ::before on the widget container */
    .st-key-seed { position: relative; }
    .st-key-seed::before { content: 'Seed'; position: absolute; top: -18px; left: 2px; font-weight: 600; font-size: 0.85rem; line-height: 1; opacity: 0.9; }
    .st-key-seed [data-testid="stNumberInputContainer"] { height: 56px; }
    .st-key-seed input[data-testid="stNumberInputField"] { height: 56px; padding-top: 0; padding-bottom: 0; }
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
    timbre_params: dict | None = None,
):
    n = int(sr * prompt_seconds)
    x_pink = pink_noise(n, sr, seed=seed)
    # Low-end processing before STFT (optional)
    if lowend_cfg:
        if lowend_cfg.get("tame_low_end"):
            x_pink = apply_hpf(
                x_pink,
                sr,
                cutoff_hz=float(lowend_cfg.get("hpf_cutoff_hz", 25.0)),
                taps=int(lowend_cfg.get("hpf_taps", 2049)),
            )
        if lowend_cfg.get("mono_lows"):
            x_pink = apply_mono_lows(
                x_pink,
                sr,
                cutoff_hz=float(lowend_cfg.get("mono_cutoff_hz", 120.0)),
                taps=int(lowend_cfg.get("hpf_taps", 2049)),
            )
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
            timbre=timbre_params.get("timbre", "neutral")
            if timbre_params
            else "neutral",
            mask_shape=timbre_params.get("mask_shape", "gaussian")
            if timbre_params
            else "gaussian",
            detune_cents=timbre_params.get("detune_cents", 0.0)
            if timbre_params
            else 0.0,
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
        gate = rhythmic_gate_from_events(
            events, sr, n_samples=n, attack=0.01, release=0.03
        )
        y_prompt = (y_prompt * (0.15 + 0.85 * gate)).astype(np.float32)

    # Normalize
    peak = float(np.max(np.abs(y_prompt)) + 1e-12)
    if peak > 0:
        y_prompt = (y_prompt / peak).astype(np.float32)
    return y_prompt, events


st.title("AudioPrompt")
top_left, top_right = st.columns([1, 1], gap="large")
with top_left:
    st.subheader("Quick Start")
    st.markdown(
        """
        AudioPrompt creates a short, steerable pink‑noise clip that can guide AI music models. It can imprint a scale‑based melody, emphasize a frequency band (vocal/guitar/bass/custom), and prepend the prompt to your input audio.

        1. Drag‑drop input audio to create a combined output, or leave empty to generate a prompt only (a simple 1–2 bar drum loop works great as a starting point).
        2. Set Prompt seconds and choose Melody settings (root/scale/BPM).
        3. Use Focus (or Custom band) and enable Tame Low End for cleaner results.
        4. Click Generate Prompt. Preview the Prompt, and if input audio is provided, the Combined result. Download the tagged WAVs.

        Tips: 3–6 s prompts give a clear steer without masking; “Vocal” focus often helps melody “speak”.
        """
    )
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
    mcol1, mcol2 = st.columns([1, 1])
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
        roots = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
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
                index=scales.index("minor_pentatonic")
                if "minor_pentatonic" in scales
                else 0,
                help="Choose from major/modes, pentatonics, blues, etc.",
            )

        col1, col2 = st.columns(2, gap="small")
        with col1:
            bpm = st.slider(
                "BPM", 40, 220, 96, 1, help="Tempo driving randomized note durations."
            )
        with col2:
            vib_hz = st.slider(
                "Vibrato Hz", 3.0, 9.0, 5.5, 0.1, help="Rate of pitch modulation."
            )
        vib_depth = st.slider(
            "Vibrato depth",
            0.0,
            0.05,
            0.02,
            0.001,
            help="Depth of pitch modulation (fraction).",
        )

        with st.expander("Melody – Advanced", expanded=False):
            colA1, colA2 = st.columns(2, gap="small")
            with colA1:
                low_midi = st.slider(
                    "Low MIDI", 24, 84, 55, 1, help="Register floor (C4=60)."
                )
                step_bias = st.slider(
                    "Step bias",
                    0.0,
                    1.0,
                    0.5,
                    0.01,
                    help="Probability of moving to a neighboring scale degree.",
                )
                glide_prob = st.slider(
                    "Glide prob",
                    0.0,
                    1.0,
                    0.25,
                    0.01,
                    help="Probability of sliding into the next note.",
                )
            with colA2:
                high_midi = st.slider(
                    "High MIDI",
                    36,
                    96,
                    79,
                    1,
                    help="Register ceiling. Keep Low < High.",
                )
                leap_steps = st.slider(
                    "Max leap (scale steps)",
                    1,
                    8,
                    7,
                    1,
                    help="Largest jump when not stepping.",
                )
                glide_frac = st.slider(
                    "Glide frac",
                    0.0,
                    0.9,
                    0.35,
                    0.01,
                    help="Portion of the note duration spent gliding.",
                )
            rest_prob = st.slider(
                "Rest prob", 0.0, 0.5, 0.12, 0.01, help="Chance of rests vs notes."
            )

            st.divider()
            st.markdown("**Timbre** — change the texture of the imprinted melody")
            timbre_preset = st.selectbox(
                "Timbre preset",
                options=[
                    "neutral",
                    "bright",
                    "dark",
                    "pluck",
                    "reece",
                    "vocal",
                    "reed",
                    "bell",
                    "metallic",
                ],
                index=0,
                help=(
                    "Controls harmonic amplitudes to change the timbral character of the imprinted melody. "
                    "'neutral' = flat (original synthetic sound). 'dark/reece' = deep bass sounds. "
                    "'bright/metallic' = harsh/brassy. 'pluck' = tight, short. 'vocal/reed/bell' = resonant tones."
                ),
            )
            mask_shape = st.selectbox(
                "Mask shape",
                options=["gaussian", "triangle", "cauchy", "square"],
                index=0,
                help=(
                    "Changes the filter shape around each harmonic. 'gaussian' = smooth and natural. "
                    "'triangle' = sharper, more defined. 'cauchy' = heavier tails, more smear/texture. "
                    "'square' = hard edges, more synthetic."
                ),
            )
            detune_cents = st.slider(
                "Detune (cents)",
                0.0,
                20.0,
                0.0,
                0.5,
                help=(
                    "Subtle pitch offset for thickness/chorus effect. Adds slight detune to harmonics "
                    "(like unison on a synth). 0 = no detune, 5-15 = subtle warmth/widening."
                ),
            )
    else:
        # Provide defaults when melody disabled to keep variables defined
        melody_root = "E"
        melody_scale = "minor_blues"
        bpm = 96
        low_midi, high_midi = 55, 79
        step_bias, leap_steps, rest_prob = 0.5, 7, 0.12
        glide_prob, glide_frac = 0.25, 0.35
        vib_hz, vib_depth = 5.5, 0.02
        timbre_preset, mask_shape, detune_cents = "neutral", "gaussian", 0.0

    # Add ~36px top margin before Focus header for clearer separation
    st.markdown("<div style='height: 36px'></div>", unsafe_allow_html=True)
    # Focus (left) — for future tabs, keep flat containers
    st.subheader("Focus")
    fcol1, fcol2 = st.columns([1, 1])
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
            help="Removes sub-bass rumble for cleaner starts and more headroom.",
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
                band = st.slider(
                    "Focus Hz band",
                    20,
                    20000,
                    (120, 3200),
                    step=10,
                    help="Twin‑handle slider: low/high cutoff in Hz.",
                )
            else:
                band = None
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                imprint_gain = st.slider(
                    "Imprint gain",
                    0.0,
                    16.0,
                    8.0,
                    0.5,
                    help="Strength of harmonic emphasis.",
                )
            with colf2:
                harmonics = st.slider(
                    "Harmonics", 0, 16, 10, 1, help="Number of harmonic peaks."
                )
            with colf3:
                bw_frac = st.slider(
                    "BW frac",
                    0.002,
                    0.05,
                    0.01,
                    0.001,
                    help="Relative bandwidth around each harmonic.",
                )
            colf4, colf5 = st.columns(2)
            with colf4:
                floor_db = st.slider(
                    "Band floor (dB)",
                    -36,
                    0,
                    -18,
                    1,
                    help="Attenuation outside the focus band.",
                )
            with colf5:
                sharpness = st.slider(
                    "Band edge sharpness",
                    6,
                    24,
                    12,
                    1,
                    help="Steepness of the band edges.",
                )
            # Low‑end advanced controls hidden for now; using sensible defaults
    else:
        band = None
        imprint_gain, harmonics, bw_frac, floor_db, sharpness = 8.0, 10, 0.01, -18.0, 12

    # Build Focus/imprint/low‑end configs for generation
    focus_params = dict(
        preset=focus_preset if focus_preset != "none" else None, band=band
    )
    imprint_params = dict(
        gain=locals().get("imprint_gain", 8.0),
        harmonics=locals().get("harmonics", 10),
        bw_frac=locals().get("bw_frac", 0.01),
        floor_db=locals().get("floor_db", -18.0),
        sharpness=locals().get("sharpness", 12),
        n_fft=2048,
    )
    # Defaults while advanced controls are hidden
    hpf_taps = 2049  # Steep by default
    lowend_cfg = dict(
        tame_low_end=bool(tame_low_end),
        hpf_cutoff_hz=25,
        hpf_taps=int(hpf_taps),
        mono_lows=True,
        mono_cutoff_hz=120,
    )

with right:
    # Reserve a container at the top for Generate so it's visually above
    gen_top = st.container()

    # Seed control moved next to Generate button
    output_suffix = "_with_prompt"
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
        durations=(0.25, 0.5, 1.0),
        duration_probs=(0.25, 0.5, 0.25),
        glide_prob=glide_prob,
        glide_frac=glide_frac,
        vib_hz=float(vib_hz),
        vib_depth=float(vib_depth),
    )
    timbre_params = dict(
        timbre=timbre_preset,
        mask_shape=mask_shape,
        detune_cents=float(detune_cents),
    )
    focus_params = dict(
        preset=focus_preset if focus_preset != "none" else None, band=band
    )
    imprint_params = dict(
        gain=locals().get("imprint_gain", 8.0),
        harmonics=locals().get("harmonics", 10),
        bw_frac=locals().get("bw_frac", 0.01),
        floor_db=locals().get("floor_db", -18.0),
        sharpness=locals().get("sharpness", 12),
        n_fft=2048,
    )
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
        btn_col, seed_col = st.columns([3, 1], gap="medium")
        with btn_col:
            pressed = st.button("Generate Prompt", type="primary", width="stretch")
        with seed_col:
            seed = st.number_input(
                "Seed",
                min_value=-1,
                max_value=10_000_000,
                value=-1,
                step=1,
                label_visibility="collapsed",
                help="Controls randomness for pink noise and the melody (notes, glides, etc.). Set to -1 to use a new random seed each generation.",
                key="seed",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        # Debug logs are always sent to server stdout via _log_debug();
        # no UI toggle needed to keep the interface clean.

        # Output & Seed just beneath Generate (no separate heading to reduce clutter)
        prompt_seconds = st.slider(
            "Prompt seconds",
            1.0,
            12.0,
            4.0,
            0.5,
            help="Length of the generated prompt (also used when prepending).",
        )
        st.markdown("**Prompt Level**")
        ag_col1, ag_col2 = st.columns([1, 1])
        with ag_col1:
            auto_gain = st.checkbox(
                "Auto gain (match input audio)",
                value=False,
                help="Detect input audio loudness and set the prompt level automatically (RMS in dBFS).",
            )
        with ag_col2:
            auto_gain_offset_db = st.slider(
                "Prompt relative to input (dB)",
                -12.0,
                6.0,
                -3.0,
                0.5,
                help="Target prompt loudness relative to input RMS (e.g., −3 dB makes the prompt slightly quieter).",
                disabled=not auto_gain,
            )
        # Manual gain and fades
        colo1, colo2, colo3 = st.columns(3)
        with colo1:
            prompt_gain_db = st.slider(
                "Prompt gain (dB)",
                -24.0,
                6.0,
                -3.0,
                0.5,
                help="Level for the prepended prompt.",
                disabled=auto_gain,
            )
        with colo2:
            fade_in_ms = st.slider(
                "Fade-in (ms)", 0, 200, 10, 1, help="Smooth ramp at the start."
            )
        with colo3:
            fade_out_ms = st.slider(
                "Fade-out (ms)", 0, 500, 50, 1, help="Smooth ramp at the end."
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
                seed_to_use = (
                    int(np.random.randint(0, 10_000_000))
                    if seed_input == -1
                    else seed_input
                )
                _log_debug(f"[gen] seed={seed_to_use}; mem_before={_mem_usage_mb()} MB")
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
                    timbre_params=timbre_params,
                )
                st.session_state["y_prompt"], st.session_state["events"] = (
                    y_prompt,
                    events,
                )
                st.session_state["seed_used"] = seed_to_use
                st.session_state["prompt_sr"] = int(sr)
                _log_debug(
                    f"[gen] prompt_len={len(y_prompt)} samples @ {int(sr)} Hz; mem_after={_mem_usage_mb()} MB"
                )

        # Outputs header
        st.subheader("Outputs")

        # Audio and download buttons directly under Outputs
        if "y_prompt" in st.session_state:
            # Build prompt WAV using stored prompt SR
            y_prompt_local = st.session_state["y_prompt"]
            sr_prompt_local = int(st.session_state.get("prompt_sr", int(sr)))

            # File naming for prompt/combined
            seed_val_local = int(
                st.session_state.get("seed_used", st.session_state.get("seed", 7))
            )
            base_stem_local = (
                Path(uploaded.name).stem if uploaded is not None else "prompt"
            )
            suffix_local = tag_suffix(
                enable_melody,
                melody_scale,
                enable_focus,
                focus_params.get("preset"),
                band if focus_params.get("preset") is None else None,
                seed_val_local,
                output_suffix,
            )
            prompt_only_name_local = f"{base_stem_local}_prompt_scale-{melody_scale if enable_melody else 'none'}_root-{melody_root}_focus-"
            if enable_focus:
                if focus_params.get("preset"):
                    prompt_only_name_local += f"{focus_params['preset']}"
                elif band:
                    prompt_only_name_local += f"band-{int(band[0])}-{int(band[1])}"
                else:
                    prompt_only_name_local += "custom"
            else:
                prompt_only_name_local += "none"
            prompt_only_name_local += f"_seed-{seed_val_local}.wav"

            # Prompt audio + download (render below in this column)
            # Reuse cached prompt bytes if parameters unchanged
            st.caption(f"Seed used: {seed_val_local}")
            st.markdown("**Prompt**")
            prompt_key_now = ("prompt", int(sr_prompt_local), int(seed_val_local))
            if (
                st.session_state.get("prompt_key") != prompt_key_now
                or "prompt_bytes" not in st.session_state
            ):
                prompt_wav_local = wav_bytes(y_prompt_local, sr_prompt_local)
                st.session_state["prompt_bytes"] = prompt_wav_local
                st.session_state["prompt_key"] = prompt_key_now
                _log_debug(
                    f"[prompt] wav_bytes={len(prompt_wav_local) / 1e6:.2f} MB; mem={_mem_usage_mb()} MB (encoded)"
                )
            else:
                prompt_wav_local = st.session_state["prompt_bytes"]
                _log_debug(f"[prompt] reused cache; mem={_mem_usage_mb()} MB")
            st.markdown("<div class='dl-row'>", unsafe_allow_html=True)
            ap_col_audio, ap_col_btn = st.columns([4, 1], gap="small")
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
                    st.error(
                        "Failed to read input audio. Prefer WAV/FLAC/OGG. "
                        "MP3 support depends on your libsndfile build.\n"
                        f"Error: {e}"
                    )
                    x_local = None
                if x_local is not None:
                    target_len_local = int(round(float(prompt_seconds) * int(sr)))
                    prompt_local = y_prompt_local
                    # Resample prompt if SR differs
                    if sr_prompt_local != int(sr):
                        g_local = np.gcd(sr_prompt_local, int(sr))
                        prompt_local = resample_poly(
                            prompt_local, int(sr) // g_local, sr_prompt_local // g_local
                        ).astype(np.float32)
                    prompt_local = prompt_local[:target_len_local]
                    # Auto/Manual gain
                    if auto_gain:
                        try:
                            seed_rms_db_local = _rms_dbfs(x_local)
                            prompt_rms_db_local = (
                                _rms_dbfs(prompt_local[:target_len_local])
                                if target_len_local > 0
                                else _rms_dbfs(prompt_local)
                            )
                            desired_db_local = seed_rms_db_local + float(
                                auto_gain_offset_db
                            )
                            gain_db_local = desired_db_local - prompt_rms_db_local
                        except Exception:
                            gain_db_local = float(prompt_gain_db)
                    else:
                        gain_db_local = float(prompt_gain_db)
                    gain_local = 10 ** (gain_db_local / 20.0)
                    prompt_local = apply_fades(
                        prompt_local * gain_local,
                        int(sr),
                        int(fade_in_ms),
                        int(fade_out_ms),
                    )

                    # Build or reuse Combined WAV without allocating a large concatenated array
                    combined_key_now = (
                        "combined",
                        int(sr),
                        float(prompt_seconds),
                        int(seed_val_local),
                        bool(auto_gain),
                        float(auto_gain_offset_db)
                        if auto_gain
                        else float(prompt_gain_db),
                        int(fade_in_ms),
                        int(fade_out_ms),
                        getattr(uploaded, "name", None),
                        getattr(uploaded, "size", None),
                    )
                    if (
                        st.session_state.get("combined_key") != combined_key_now
                        or "combined_bytes" not in st.session_state
                    ):
                        combined_wav_local = wav_bytes_concat_segments(
                            [prompt_local, x_local.astype(np.float32, copy=False)],
                            int(sr),
                        )
                        st.session_state["combined_bytes"] = combined_wav_local
                        st.session_state["combined_key"] = combined_key_now
                        _log_debug(
                            f"[combined] built; wav_bytes={len(combined_wav_local) / 1e6:.2f} MB; mem={_mem_usage_mb()} MB"
                        )
                    else:
                        combined_wav_local = st.session_state["combined_bytes"]
                        _log_debug(f"[combined] reused cache; mem={_mem_usage_mb()} MB")
                    st.markdown("**Combined**")
                    st.markdown("<div class='dl-row'>", unsafe_allow_html=True)
                    cap_col_audio, cap_col_btn = st.columns([4, 1], gap="small")
                    with cap_col_audio:
                        st.audio(combined_wav_local, format="audio/wav")
                    with cap_col_btn:
                        combined_name_local = f"{Path(uploaded.name).stem}{suffix_local}_root-{melody_root}.wav"
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
        # Blue placeholders when outputs are not ready (pill style)
        if "y_prompt" not in st.session_state:
            st.markdown(
                "<div class='spec-placeholder'>Set your parameters and press Generate Prompt.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='spec-placeholder'>No input file uploaded; only the prompt is generated.</div>",
                unsafe_allow_html=True,
            )
        elif uploaded is None:
            st.markdown(
                "<div class='spec-placeholder'>No input file uploaded; only the prompt is generated.</div>",
                unsafe_allow_html=True,
            )

# Footer: brief Terms & Privacy notice (public hosting)
st.markdown("---")
st.markdown(
    "<div class='footer-note'>Terms & Privacy: Upload only content you have rights to. By using this app you confirm you have permission to process any uploaded audio.</div>",
    unsafe_allow_html=True,
)

# Inject drag-over highlight script at the end to avoid top spacer
st.markdown("<div class='js-hook'></div>", unsafe_allow_html=True)
components.html(_DRAG_JS, height=0)
