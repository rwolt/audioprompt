# AudioPrompt — Streamlit App and CLI

Create short, steerable “audio prompts” to nudge AI music tools. AudioPrompt can imprint a scale‑constrained melody onto pink noise, emphasize vocal/guitar/bass bands, and prepend the prompt to your track for upload.

Demo: [https://audioprompt.streamlit.app/](https://audioprompt.streamlit.app/)
---

## ✨ Features
- 🧠 Scale‑driven melody imprint (randomized, in‑key, with vibrato/glides)
- 🎛️ Lightweight character controls for melody tone and note shape
- 🎚️ Spectral focus (vocal/guitar/bass presets or custom Hz band)
- 🥁 Rhythmic gate from note events (phrase‑like envelope)
- 🥁 **Drum MIDI Imprint** — upload a MIDI drum track to generate a rhythmic noise layer (velocity‑sensitive kick/snare/hat/perc layers with tone, tune, decay, and blend controls)
- 📎 Drag‑and‑drop input; tagged downloads (scale/focus/drum/seed)
- 🧰 Pure Python DSP (NumPy/SciPy/soundfile)

---

## 📦 What’s Included
- Streamlit App (local or Streamlit Community Cloud)
- Minimal CLI for analysis/prompt/prepend

Supported formats
- Input: WAV/FLAC/OGG/AIFF via libsndfile; MP3 is supported if your libsndfile build includes MP3 (mpg123). If MP3 fails to load, convert to WAV/FLAC/OGG.
- Output: WAV

Requirements
- Python 3.9+
- Install deps in a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Conda alternative
```bash
conda create -n audioprompt python=3.11 -y
conda activate audioprompt
python -m pip install -r requirements.txt
```

---

## 🚀 Streamlit App (recommended)

Run locally
```bash
cd audioprompt_app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Conda alternative
```bash
cd audioprompt_app
conda create -n audioprompt-app python=3.11 -y
conda activate audioprompt-app
pip install -r requirements.txt
streamlit run app.py
```

Deploy to Streamlit Community Cloud
```text
1) Push this repo to GitHub
2) Create a new Streamlit app and point it to audioprompt_app/app.py
3) It installs audioprompt_app/requirements.txt and runs automatically
```

Quick start (in the app)
- Drag‑drop input audio to create a combined output, or leave empty to generate a prompt only
- Set Prompt seconds and Melody (root/scale/BPM/range)
- Use Focus (vocal/guitar/bass/custom Hz band) and enable Bass Roll-Off if needed
- Press “Generate Prompt” → preview Prompt and (if provided) Combined → download tagged WAVs

Tips
- 3–6 s prompts give a clear steer without masking
- Use WAV/FLAC inputs for best reliability; MP3 works when libsndfile has MP3 enabled

---

## 🧪 CLI (headless)

Usage
```text
python audioprompt.py analyze <in.wav> [--bands N] [--bins M] [--outdir DIR] [--print-eq]
python audioprompt.py prompt  <in.wav> <out.wav> [--duration S] [--max-gain dB]
python audioprompt.py prepend <prompt.wav> <seed.wav> <out.wav>

Options
  analyze:
    --bands N      Number of log-spaced bands for the profile (default: 1024)
    --bins M       Downsampled bands for convenience/preview (default: 24)
    --outdir DIR   Directory for profile outputs (default: .)
    --print-eq     Print firequalizer-style entries to stdout

  prompt:
    --duration S   Prompt length in seconds (default: 6.0)
    --max-gain dB  Max inverse-EQ boost (default: 6.0 dB)
```

Examples
```bash
# Analyze and print firequalizer entries
python audioprompt.py analyze input.wav --print-eq

# Generate an 8 s prompt with stronger shaping
python audioprompt.py prompt input.wav prompt.wav --duration 8 --max-gain 8

# Prepend prompt to seed for upload
python audioprompt.py prepend prompt.wav input.wav upload.wav
```

---

## 📁 Project Layout
- `audioprompt_app/` — Streamlit app (two‑column UI, dark theme, drag‑drop upload)
- `audioprompt_app/src/audioprompt_core/` — shared core (audio, melody, prompt, mididrums, drumnoise)
- `audioprompt.py` — CLI (analyze/prompt/prepend)
- `requirements.txt` — minimal dependencies for the CLI/analysis tools

---

## 🧬 How It Works
- Pink noise base: builds 1/√f spectrum in the frequency domain and inverse‑FFTs to time domain; normalizes to headroom.
- Random melody (optional):
  - Picks scale notes (root + mode) within a MIDI range; durations from BPM‑scaled choices.
  - Step/leap behavior, rests, and seed‑driven randomness; converts notes to an f0 trajectory.
  - Expression: optional glides between notes and subtle vibrato; light smoothing to reduce discontinuities.
  - Character controls adjust harmonic balance and note envelope using simple musician-facing presets.
- Spectral imprint:
  - STFT on the pink noise; per frame, builds a harmonic mask centered at the current f0 (and harmonics) with bandwidth proportional to frequency.
  - Multiplies the STFT magnitude by this mask to emphasize harmonic structure; preserves phase; ISTFT back to time domain; re‑normalize.
- Focus band (optional): applies a global soft band‑pass mask in log‑frequency (preset or custom low/high Hz), with outside‑band floor.
- Rhythmic gate (optional): constructs an amplitude envelope from event onsets/offsets (attack/release), and applies it to the prompt.
- Drum MIDI Imprint (optional): parses a General MIDI drum track, maps notes to kick/snare/hat/perc lanes, generates band-limited pink noise per lane, and applies velocity-sensitive AD envelopes. Drum character presets shift lane frequency bands, decay, and light drive; snare tune shifts the snare lane by semitones before synthesis. Drum BPM can match Melody BPM or independently time-scale the MIDI, and optional looping repeats short grooves to fill the prompt.
- Prepend path: resamples the prompt if needed, trims to “prompt seconds”, applies gain and fades, concatenates with the input, and trims peaks (≤ −1 dBFS).
- Tagged filenames: use compact tags for root, scale, melody BPM, melody/noise seed, drum BPM, drum seed, and optional focus/character settings.
- Determinism: fixed `seed` yields repeatable results; `-1` chooses a new random seed at each generation (Streamlit app).

---

## 🛡️ Legal & Content Use
- Upload only content you have the rights and permissions to use.
- By using the app, you confirm you have permission to process any uploaded audio.
- This project is provided “as is” without warranties; see the MIT license for details.

---

## 📄 Terms & Privacy
- You retain all rights to your audio. You grant permission to process your uploaded file(s) for the purpose of generating prompts.
- Do not upload third‑party copyrighted material without authorization.
- No personal data is collected beyond what Streamlit requires to serve the app.

---

## 📜 License
MIT License — see `LICENSE`.
