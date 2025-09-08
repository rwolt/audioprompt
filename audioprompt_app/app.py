import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from scipy.signal import resample_poly
import matplotlib.pyplot as plt


def _rms_dbfs(y: np.ndarray) -> float:
    y = y.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(y * y)) + 1e-12)
    return 20.0 * np.log10(rms)

# Import core from ./src
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from audioprompt_core import (
    SCALES,
    load_audio_mono,
    apply_fades,
    wav_bytes,
    tag_suffix,
    pink_noise,
    imprint_melody_focus,
    rhythmic_gate_from_events,
    generate_random_melody,
    events_to_f0,
    apply_hpf,
    apply_mono_lows,
)


st.set_page_config(page_title="AudioPrompt", layout="wide")
st.markdown(
    """
    <style>
    .block-container{padding-top:1rem;padding-bottom:2rem;}
    /* Keep Streamlit's primary theme color for the button; minimal tweaks only */
    div.stButton > button[kind="primary"] { padding: 0.9rem 1.25rem; font-size: 1.05rem; border-radius: 10px; }
    /* Style the file uploader like a drop zone */
    div[data-testid="stFileUploader"] > section {
        border: 2px dashed rgba(255,255,255,0.2);
        padding: 1rem; border-radius: 10px; transition: border-color .2s, background-color .2s;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: var(--primary-color, #FF6B6B); background: rgba(255,107,107,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
top_left, top_right = st.columns([1, 1])
with top_left:
    st.subheader("Quick Start")
    st.markdown(
        """
        1. Drag‑drop an audio file (optional).
        2. Set Prompt seconds and choose Melody settings (scale, range, BPM).
        3. (Optional) Enable Focus to emphasize a vocal/guitar/bass band or custom Hz range.
        4. Press Generate Prompt, then preview and download the Prompt and Combined outputs.
        
        Tips: 3–6 s prompts give a clear steer without masking; try Minor Blues + Vocal focus for vocal‑friendly nudges.
        """
    )
with top_right:
    st.subheader("Input & Output")
    uploaded = st.file_uploader(
        "Input audio (optional)",
        type=["wav", "flac", "ogg", "aiff", "aif"],
        accept_multiple_files=False,
        help="Drag & drop a file. If provided, the prompt will be prepended to create a combined output.",
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
left, right = st.columns(2)

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
        melody_root = st.selectbox("Root", roots, index=roots.index("E"), help="Root note for the scale (C4=60).")
        scales = sorted(list(SCALES.keys()))
        melody_scale = st.selectbox("Scale", options=scales, index=scales.index("minor_blues") if "minor_blues" in scales else 0, help="Choose from major/modes, pentatonics, blues, etc.")

        col1, col2 = st.columns(2)
        with col1:
            bpm = st.slider("BPM", 40, 220, 96, 1, help="Tempo driving randomized note durations.")
        with col2:
            vib_hz = st.slider("Vibrato Hz", 3.0, 9.0, 5.5, 0.1, help="Rate of pitch modulation.")
        vib_depth = st.slider("Vibrato depth", 0.0, 0.05, 0.02, 0.001, help="Depth of pitch modulation (fraction).")

        with st.expander("Melody – Advanced", expanded=False):
            colA1, colA2 = st.columns(2)
            with colA1:
                low_midi = st.slider("Low MIDI", 24, 84, 55, 1, help="Register floor (C4=60).")
                step_bias = st.slider("Step bias", 0.0, 1.0, 0.8, 0.01, help="Probability of moving to a neighboring scale degree.")
                glide_prob = st.slider("Glide prob", 0.0, 1.0, 0.25, 0.01, help="Probability of sliding into the next note.")
            with colA2:
                high_midi = st.slider("High MIDI", 36, 96, 79, 1, help="Register ceiling. Keep Low < High.")
                leap_steps = st.slider("Max leap (scale steps)", 1, 8, 4, 1, help="Largest jump when not stepping.")
                glide_frac = st.slider("Glide frac", 0.0, 0.9, 0.35, 0.01, help="Portion of the note duration spent gliding.")
            rest_prob = st.slider("Rest prob", 0.0, 0.5, 0.12, 0.01, help="Chance of rests vs notes.")
    else:
        # Provide defaults when melody disabled to keep variables defined
        melody_root = "E"
        melody_scale = "minor_blues"
        bpm = 96
        low_midi, high_midi = 55, 79
        step_bias, leap_steps, rest_prob = 0.8, 4, 0.12
        glide_prob, glide_frac = 0.25, 0.35
        vib_hz, vib_depth = 5.5, 0.02

    # Output & Seed and Focus controls moved to the right column to balance layout

with right:
    st.subheader("Generate & Outputs")
    # Focus with compact defaults and expandable advanced controls
    st.markdown("**Focus**")
    enable_focus = st.checkbox(
        "Enable focus band",
        value=False,
        help="Emphasize energy in a vocal/guitar/bass band or a custom Hz range.",
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
    # Simple low-end checkbox directly under presets (sane default on)
    tame_low_end = st.checkbox(
        "Tame Low End",
        value=True,
        help="Reduce inaudible sub‑bass to free headroom and keep starts clean. Advanced controls in the expander.",
    )

    # Advanced controls (auto‑expand for custom preset)
    with st.expander("Focus Band – Advanced Settings", expanded=(enable_focus and focus_preset == "custom")):
        if enable_focus and focus_preset == "custom":
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

        st.markdown("**Low‑End (advanced)**")
        # Defaults for low‑end advanced section
        hpf_cutoff_hz = st.slider(
            "HPF cutoff (Hz)", 20, 40, 25, 1,
            help="Frequencies below this are rolled off with a linear‑phase FIR filter.",
        )
        steepness = st.select_slider(
            "Steepness",
            options=["Normal", "Steep"],
            value="Steep",
            help="Filter length: Normal (~512 taps) or Steep (~2048 taps). Steeper = cleaner cutoff (more CPU).",
        )
        mono_lows = st.checkbox(
            "Mono low frequencies",
            value=True,
            help="Sum bass to mono (e.g., <120 Hz) to keep low end tight. Mostly relevant for stereo prompts.",
        )
        mono_cutoff_hz = st.slider(
            "Mono below (Hz)", 80, 180, 120, 5,
            help="Frequencies below this are summed to mono while highs remain unchanged.",
        )

    # Small spacing between Focus and Output & Seed
    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    # Output & Seed
    st.markdown("**Output & Seed**")
    prompt_seconds = st.slider(
        "Prompt seconds",
        1.0,
        12.0,
        4.0,
        0.5,
        help="Length of the generated prompt (also used when prepending).",
    )
    # Prompt level group
    st.markdown("**Prompt Level**")
    ag_col1, ag_col2 = st.columns([1,1])
    with ag_col1:
        auto_gain = st.checkbox(
            "Auto gain (match seed)",
            value=False,
            help="Detect seed loudness and set the prompt level automatically. Uses RMS in dBFS."
        )
    with ag_col2:
        auto_gain_offset_db = st.slider(
            "Prompt relative to seed (dB)",
            -12.0, 6.0, -3.0, 0.5,
            help="Target prompt loudness relative to seed RMS (e.g., −3 dB means the prompt is slightly quieter).",
            disabled=not auto_gain,
        )

    # Manual gain (disabled when auto-gain is on)
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
    # Seed: enter -1 to randomize each generation
    seed = st.number_input(
        "Seed",
        min_value=-1,
        max_value=10_000_000,
        value=-1,
        step=1,
        help="Controls randomness for pink noise and the melody (notes, glides, etc.). Set to -1 to use a new random seed each generation.",
        key="seed",
    )
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
    imprint_params = dict(gain=imprint_gain, harmonics=harmonics, bw_frac=bw_frac, floor_db=floor_db, sharpness=sharpness, n_fft=2048)
    # Low-end config dict for core processing
    hpf_taps = 2049 if steepness == "Steep" else 513
    lowend_cfg = dict(
        tame_low_end=bool(tame_low_end),
        hpf_cutoff_hz=int(hpf_cutoff_hz),
        hpf_taps=int(hpf_taps),
        mono_lows=bool(mono_lows),
        mono_cutoff_hz=int(mono_cutoff_hz),
    )

    # Big colorful button to generate the prompt
    pressed = st.button("Generate Prompt", type="primary", use_container_width=True)

    # Auto-generate once on first load
    auto_first = ("y_prompt" not in st.session_state)
    if pressed or auto_first:
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

    if "y_prompt" not in st.session_state:
        st.info("Set your parameters and press Generate Prompt.")
        st.stop()

    y_prompt = st.session_state["y_prompt"]
    events = st.session_state.get("events")

    # Prompt preview and download
    st.markdown("**Prompt**")
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
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)
    except Exception:
        pass
    prompt_wav = wav_bytes(y_prompt, sr_prompt)
    st.audio(prompt_wav, format="audio/wav")

    # Tagged filenames
    if uploaded is not None:
        base_stem = Path(uploaded.name).stem
    else:
        base_stem = "prompt"
    seed_val = int(st.session_state.get("seed_used", st.session_state.get("seed", 7)))
    suffix = tag_suffix(enable_melody, melody_scale, enable_focus, focus_params.get("preset"), band if focus_params.get("preset") is None else None, seed_val, output_suffix)
    prompt_only_name = f"{base_stem}_prompt_scale-{melody_scale if enable_melody else 'none'}_focus-"
    if enable_focus:
        if focus_params.get("preset"):
            prompt_only_name += f"{focus_params['preset']}"
        elif band:
            prompt_only_name += f"band-{int(band[0])}-{int(band[1])}"
        else:
            prompt_only_name += "custom"
    else:
        prompt_only_name += "none"
    prompt_only_name += f"_seed-{seed_val}.wav"

    st.download_button("Download prompt", data=prompt_wav, file_name=prompt_only_name, mime="audio/wav")

    # Combined output if file provided
    if uploaded is not None:
        try:
            x, sr_in = load_audio_mono(uploaded, int(sr))
        except Exception as e:
            st.error(f"Failed to read input audio. Prefer WAV/FLAC/OGG. Error: {e}")
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

            # Spectrogram preview (Combined)
            try:
                fig_c, ax_c = plt.subplots(figsize=(6, 2.4))
                f_c, t_c, Sxx_c = _spectrogram(combined.astype(np.float32, copy=False), int(sr), nperseg=1024, noverlap=768)
                Sxx_c_db = 10 * np.log10(Sxx_c + 1e-12)
                pcm2 = ax_c.pcolormesh(t_c, f_c, Sxx_c_db, shading="auto", cmap="magma")
                ax_c.set_ylabel("Hz")
                ax_c.set_xlabel("s")
                ax_c.set_title("Combined – Spectrogram")
                plt.colorbar(pcm2, ax=ax_c, fraction=0.046, pad=0.02, label="dB")
                st.pyplot(fig_c, use_container_width=True)
                plt.close(fig_c)
            except Exception:
                pass

            combined_wav = wav_bytes(combined, int(sr))
            st.markdown("**Combined**")
            st.audio(combined_wav, format="audio/wav")

            combined_name = f"{Path(uploaded.name).stem}{suffix}.wav"
            st.download_button("Download combined", data=combined_wav, file_name=combined_name, mime="audio/wav")
        else:
            st.info("Upload a WAV/FLAC/OGG file to create a combined output.")
    else:
        st.info("No input file uploaded; only the prompt is generated.")

# Footer: brief Terms & Privacy notice (public hosting)
st.markdown("---")
st.caption(
    "Terms & Privacy: Upload only content you have rights to. By using this app you confirm you have permission to process any uploaded audio."
)
