import os
import sys
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
from audioprompt_core.audio import load_audio_mono, apply_fades, wav_bytes, tag_suffix
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
    /* Stack audio + download button vertically for consistent layout */
    .dl-row + div[data-testid="stHorizontalBlock"] { flex-direction: column; align-items: stretch; gap: 0.5rem; }
    .dl-row + div[data-testid="stHorizontalBlock"] > div { width: 100% !important; }
    .dl-row + div[data-testid="stHorizontalBlock"] .stButton button { width: 100%; }
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
        gate = rhythmic_gate_from_events(events, sr, n_samples=n, attack=0.01, release=0.03)
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
    )
    sr = st.number_input(
        "Sample rate (Hz)",
        min_value=8000,
        max_value=96000,
        value=48000,
        step=1000,
        help="Processing rate; inputs are resampled. Higher SR costs more CPU.",
    )

    # Toggles moved next to relevant sections below
    # Prompt seconds moved to Output & Seed section for better workflow alignment

# Single divider spanning both columns to separate top row from main content
st.divider()
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
            bpm = st.slider("BPM", 40, 220, 96, 1, help="Tempo driving randomized note durations.")
        with col2:
            vib_hz = st.slider("Vibrato Hz", 3.0, 9.0, 5.5, 0.1, help="Rate of pitch modulation.")
        vib_depth = st.slider("Vibrato depth", 0.0, 0.05, 0.02, 0.001, help="Depth of pitch modulation (fraction).")

        with st.expander("Melody – Advanced", expanded=False):
            colA1, colA2 = st.columns(2, gap="small")
            with colA1:
                low_midi = st.slider("Low MIDI", 24, 84, 55, 1, help="Register floor (C4=60).")
                step_bias = st.slider("Step bias", 0.0, 1.0, 0.5, 0.01, help="Probability of moving to a neighboring scale degree.")
                glide_prob = st.slider("Glide prob", 0.0, 1.0, 0.25, 0.01, help="Probability of sliding into the next note.")
            with colA2:
                high_midi = st.slider("High MIDI", 36, 96, 79, 1, help="Register ceiling. Keep Low < High.")
                leap_steps = st.slider("Max leap (scale steps)", 1, 8, 7, 1, help="Largest jump when not stepping.")
                glide_frac = st.slider("Glide frac", 0.0, 0.9, 0.35, 0.01, help="Portion of the note duration spent gliding.")
            rest_prob = st.slider("Rest prob", 0.0, 0.5, 0.12, 0.01, help="Chance of rests vs notes.")
    else:
        # Provide defaults when melody disabled to keep variables defined
        melody_root = "E"
        melody_scale = "minor_blues"
        bpm = 96
        low_midi, high_midi = 55, 79
        step_bias, leap_steps, rest_prob = 0.5, 7, 0.12
        glide_prob, glide_frac = 0.25, 0.35
        vib_hz, vib_depth = 5.5, 0.02

    # Small vertical gap between Melody and Focus sections
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
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
                band = st.slider("Focus Hz band", 20, 20000, (120, 3200), step=10, help="Twin‑handle slider: low/high cutoff in Hz.")
            else:
                band = None
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                imprint_gain = st.slider("Imprint gain", 0.0, 16.0, 8.0, 0.5, help="Strength of harmonic emphasis.")
            with colf2:
                harmonics = st.slider("Harmonics", 0, 16, 10, 1, help="Number of harmonic peaks.")
            with colf3:
                bw_frac = st.slider("BW frac", 0.002, 0.05, 0.01, 0.001, help="Relative bandwidth around each harmonic.")
            colf4, colf5 = st.columns(2)
            with colf4:
                floor_db = st.slider("Band floor (dB)", -36, 0, -18, 1, help="Attenuation outside the focus band.")
            with colf5:
                sharpness = st.slider("Band edge sharpness", 6, 24, 12, 1, help="Steepness of the band edges.")
            # Low‑end advanced controls hidden for now; using sensible defaults
    else:
        band = None
        imprint_gain, harmonics, bw_frac, floor_db, sharpness = 8.0, 10, 0.01, -18.0, 12

    # Build Focus/imprint/low‑end configs for generation
    focus_params = dict(preset=focus_preset if focus_preset != "none" else None, band=band)
    imprint_params = dict(gain=locals().get('imprint_gain', 8.0), harmonics=locals().get('harmonics', 10), bw_frac=locals().get('bw_frac', 0.01), floor_db=locals().get('floor_db', -18.0), sharpness=locals().get('sharpness', 12), n_fft=2048)
    # Defaults while advanced controls are hidden
    hpf_taps = 2049  # Steep by default
    lowend_cfg = dict(
        tame_low_end=bool(tame_low_end),
        hpf_cutoff_hz=25,
        hpf_taps=int(hpf_taps),
        mono_lows=True,
        mono_cutoff_hz=120,
    )

    # Build Focus/imprint/low‑end configs for generation
    focus_params = dict(preset=focus_preset if focus_preset != "none" else None, band=band)
    imprint_params = dict(gain=locals().get('imprint_gain', 8.0), harmonics=locals().get('harmonics', 10), bw_frac=locals().get('bw_frac', 0.01), floor_db=locals().get('floor_db', -18.0), sharpness=locals().get('sharpness', 12), n_fft=2048)
    hpf_taps = 2049  # Steep by default
    lowend_cfg = dict(
        tame_low_end=bool(tame_low_end),
        hpf_cutoff_hz=25,
        hpf_taps=int(hpf_taps),
        mono_lows=True,
        mono_cutoff_hz=120,
    )

with right:
    # Reserve a container at the top for Generate & Outputs so it's visually above
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
    focus_params = dict(preset=focus_preset if focus_preset != "none" else None, band=band)
    imprint_params = dict(gain=locals().get('imprint_gain', 8.0), harmonics=locals().get('harmonics', 10), bw_frac=locals().get('bw_frac', 0.01), floor_db=locals().get('floor_db', -18.0), sharpness=locals().get('sharpness', 12), n_fft=2048)
    # Low-end config dict for core processing
    hpf_taps = 2049  # Steep by default
    lowend_cfg = dict(
        tame_low_end=bool(tame_low_end),
        hpf_cutoff_hz=25,
        hpf_taps=int(hpf_taps),
        mono_lows=True,
        mono_cutoff_hz=120,
    )

    # Render Generate & Outputs at the top of the right column
    with gen_top:
        st.subheader("Generate & Outputs")
        st.markdown("<div class='gen-row'>", unsafe_allow_html=True)
        btn_col, seed_col = st.columns([3,1], gap="medium")
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

        # Generate immediately after controls so results are available below in this same run
        # Only generate when the button is pressed (no auto-generate on first load)
        if pressed:
            with st.spinner("Generating prompt..."):
                # Resolve seed: -1 means random each generation
                seed_input = int(st.session_state.get("seed", 7))
                seed_to_use = int(np.random.randint(0, 10_000_000)) if seed_input == -1 else seed_input
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
                )
                st.session_state["y_prompt"], st.session_state["events"] = y_prompt, events
                st.session_state["seed_used"] = seed_to_use
                st.session_state["prompt_sr"] = int(sr)

        # Audio and download buttons directly under gain/fade sliders
        if "y_prompt" in st.session_state:
            # Build prompt WAV using stored prompt SR
            y_prompt_local = st.session_state["y_prompt"]
            sr_prompt_local = int(st.session_state.get("prompt_sr", int(sr)))

            # File naming for prompt/combined
            seed_val_local = int(st.session_state.get("seed_used", st.session_state.get("seed", 7)))
            base_stem_local = Path(uploaded.name).stem if uploaded is not None else "prompt"
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

            # Prompt audio + download
            st.markdown("**Prompt**")
            prompt_wav_local = wav_bytes(y_prompt_local, sr_prompt_local)
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
                    combined_local = np.concatenate([prompt_local.astype(np.float32, copy=False), x_local.astype(np.float32, copy=False)], axis=0)
                    peak_local = float(np.max(np.abs(combined_local)) + 1e-12)
                    if peak_local > 0.999:
                        combined_local = (combined_local / peak_local * 0.999).astype(np.float32)

                    combined_wav_local = wav_bytes(combined_local, int(sr))
                    st.markdown("**Combined**")
                    st.markdown("<div class='dl-row'>", unsafe_allow_html=True)
                    cap_col_audio, cap_col_btn = st.columns([4,1], gap="small")
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
        if "y_prompt" not in st.session_state:
            st.info("Set your parameters and press Generate Prompt.")
        else:
            y_prompt = st.session_state["y_prompt"]
            events = st.session_state.get("events")

        # Spectrogram previews only (no audio or downloads)
        # Show seed info above the spectrograms (after audio players)
        if "y_prompt" in st.session_state:
            seed_val = int(st.session_state.get("seed_used", st.session_state.get("seed", 7)))
            st.caption(f"Seed used: {seed_val}")
        st.subheader("Spectrograms")
        if "y_prompt" in st.session_state:
            st.markdown("**Prompt**")
            # Ensure local variables exist before use
            y_prompt = st.session_state["y_prompt"]
            sr_prompt = int(st.session_state.get("prompt_sr", int(sr)))
            # Spectrogram preview (Prompt)
            try:
                from scipy.signal import spectrogram as _spectrogram
                fig_p, ax_p = plt.subplots(figsize=(6, 2.4))
                f_p, t_p, Sxx_p = _spectrogram(y_prompt.astype(np.float32, copy=False), sr_prompt, nperseg=1024, noverlap=768)
                Sxx_p_db = 10 * np.log10(Sxx_p + 1e-12)
                pcm = ax_p.pcolormesh(t_p, f_p, Sxx_p_db, shading="auto", cmap="magma")
                ax_p.set_ylabel("Hz")
                ax_p.set_xlabel("s")
                ax_p.set_title("Prompt – Spectrogram")
                plt.colorbar(pcm, ax=ax_p, fraction=0.046, pad=0.02, label="dB")
                st.pyplot(fig_p, width="stretch")
                plt.close(fig_p)
            except Exception:
                pass

        # Combined output spectrogram if file provided
        if uploaded is not None and "y_prompt" in st.session_state:
            try:
                x, sr_in = load_audio_mono(uploaded, int(sr))
            except Exception as e:
                st.error(
                    "Failed to read input audio. Prefer WAV/FLAC/OGG. "
                    "MP3 support depends on your libsndfile build.\n"
                    f"Error: {e}"
                )
                x = None
            if x is not None:
                # Prepare prepend prompt slice with gain & fades
                target_len = int(round(float(prompt_seconds) * int(sr)))
                prompt = y_prompt
                # Resample prompt if its SR differs
                if sr_prompt != int(sr):
                    g = np.gcd(sr_prompt, int(sr))
                    prompt = resample_poly(prompt, int(sr) // g, sr_prompt // g).astype(np.float32)
                prompt = prompt[:target_len]
                # Determine prompt gain: auto or manual
                if auto_gain:
                    try:
                        seed_rms_db = _rms_dbfs(x)
                        prompt_rms_db = _rms_dbfs(prompt[:target_len]) if target_len > 0 else _rms_dbfs(prompt)
                        desired_db = seed_rms_db + float(auto_gain_offset_db)
                        gain_db = desired_db - prompt_rms_db
                    except Exception:
                        gain_db = float(prompt_gain_db)
                else:
                    gain_db = float(prompt_gain_db)
                gain = 10 ** (gain_db / 20.0)
                prompt = apply_fades(prompt * gain, int(sr), int(fade_in_ms), int(fade_out_ms))
                combined = np.concatenate([prompt.astype(np.float32, copy=False), x.astype(np.float32, copy=False)], axis=0)
                peak = float(np.max(np.abs(combined)) + 1e-12)
                if peak > 0.999:
                    combined = (combined / peak * 0.999).astype(np.float32)

                # Combined spectrogram label above the image, then plot
                st.markdown("**Combined**")
                try:
                    fig_c, ax_c = plt.subplots(figsize=(6, 2.4))
                    f_c, t_c, Sxx_c = _spectrogram(combined.astype(np.float32, copy=False), int(sr), nperseg=1024, noverlap=768)
                    Sxx_c_db = 10 * np.log10(Sxx_c + 1e-12)
                    pcm2 = ax_c.pcolormesh(t_c, f_c, Sxx_c_db, shading="auto", cmap="magma")
                    ax_c.set_ylabel("Hz")
                    ax_c.set_xlabel("s")
                    ax_c.set_title("Combined – Spectrogram")
                    plt.colorbar(pcm2, ax=ax_c, fraction=0.046, pad=0.02, label="dB")
                    st.pyplot(fig_c, width="stretch")
                    plt.close(fig_c)
                except Exception:
                    pass
            else:
                st.info("Upload a WAV/FLAC/OGG file to create a combined output.")
        else:
            st.info("No input file uploaded; only the prompt is generated.")

# Footer: brief Terms & Privacy notice (public hosting)
st.markdown("---")
st.markdown(
    "<div class='footer-note'>Terms & Privacy: Upload only content you have rights to. By using this app you confirm you have permission to process any uploaded audio.</div>",
    unsafe_allow_html=True,
)

# Inject drag-over highlight script at the end to avoid top spacer
st.markdown("<div class='js-hook'></div>", unsafe_allow_html=True)
components.html(_DRAG_JS, height=0)
