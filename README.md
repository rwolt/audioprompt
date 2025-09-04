# AudioPrompt — Streamlit App, Notebook, and CLI

Create short, steerable “audio prompts” to nudge AI music tools. AudioPrompt can imprint a scale‑constrained melody onto pink noise, emphasize vocal/guitar/bass bands, and prepend the prompt to your track for upload.

---

## ✨ Features
- 🧠 Scale‑driven melody imprint (randomized, in‑key, with vibrato/glides)
- 🎚️ Spectral focus (vocal/guitar/bass presets or custom Hz band)
- 🥁 Rhythmic gate from note events (phrase‑like envelope)
- 📎 Drag‑and‑drop input; tagged downloads (scale/focus/seed)
- 🧰 Pure Python DSP (NumPy/SciPy/soundfile) — no ffmpeg

---

## 📦 What’s Included
- Streamlit App (local or Streamlit Community Cloud)
- Jupyter Notebook (`audioprompt_workbook.ipynb`)
- Minimal CLI for analysis/prompt/prepend

Supported formats
- Input: WAV/FLAC/OGG/AIFF via libsndfile
- Output: WAV

Requirements
- Python 3.9+
- Install deps in a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
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

Deploy to Streamlit Community Cloud
```text
1) Push this repo to GitHub
2) Create a new Streamlit app and point it to audioprompt_app/app.py
3) It installs audioprompt_app/requirements.txt and runs automatically
```

Quick start (in the app)
- Set Prompt seconds and Melody (root/scale/BPM/range)
- (Optional) Enable Focus (vocal/guitar/bass/custom Hz band)
- Press “Generate Prompt” → preview Prompt and Combined → download

Tips
- 3–6 s prompts give a clear steer without masking
- Use WAV/FLAC inputs for reliability

---

## 📓 Notebook (`audioprompt_workbook.ipynb`)
- Open the workbook and run cells top‑to‑bottom.
- Set `INPUT_AUDIO`, `PROMPT_SECONDS`, and toggles (Melody, Focus, Gate).
- Preview `y_prompt`; optionally prepend and export tagged WAVs.

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
- `audioprompt_app/src/audioprompt_core/` — shared core (audio, melody, prompt)
- `audioprompt_workbook.ipynb` — shareable workbook version of the workflow
- `audioprompt.py` — CLI (analyze/prompt/prepend)
- `requirements.txt` — minimal dependencies for the CLI/analysis tools

---

## 🧬 How It Works (quick refresher)
- Pink noise base: builds 1/√f spectrum in the frequency domain and inverse‑FFTs to time domain; normalizes to headroom.
- Random melody (optional):
  - Picks scale notes (root + mode) within a MIDI range; durations from BPM‑scaled choices.
  - Step/leap behavior, rests, and seed‑driven randomness; converts notes to an f0 trajectory.
  - Expression: optional glides between notes and subtle vibrato; light smoothing to reduce discontinuities.
- Spectral imprint:
  - STFT on the pink noise; per frame, builds a harmonic mask centered at the current f0 (and harmonics) with Gaussian bandwidth proportional to frequency.
  - Multiplies the STFT magnitude by this mask to emphasize harmonic structure; preserves phase; ISTFT back to time domain; re‑normalize.
- Focus band (optional): applies a global soft band‑pass mask in log‑frequency (preset or custom low/high Hz), with outside‑band floor.
- Rhythmic gate (optional): constructs an amplitude envelope from event onsets/offsets (attack/release), and applies it to the prompt.
- Prepend path: resamples the prompt if needed, trims to “prompt seconds”, applies gain and fades, concatenates with the input, and trims peaks (≤ −1 dBFS).
- Tagged filenames: include `scale`, `focus` (preset or band), and `seed_used` for easy A/B comparisons.
- Determinism: fixed `seed` yields repeatable results; `-1` chooses a new random seed at each generation (Streamlit app).

---

## ⚖️ Licensing Notes (MIT vs GPL?)
- MIT is permissive: you (and others) can use the code in closed‑source and commercial apps with attribution. This aligns well if you might build a commercial app later.
- GPL (and AGPL) are copyleft: if you distribute a derived app, you must open‑source it under the same license (AGPL extends to network use). This deters closed‑source reuse but also constrains your own future commercial, closed distribution.
- Middle ground: dual‑license (e.g., MIT for open use + a separate commercial license), or keep advanced features proprietary in a separate, non‑open repo.
- Recommendation for this repo: MIT (you can still build proprietary successors); see `LICENSE`.

---

## 🛡️ Legal & Content Use
- You are responsible for the audio you upload. Only use content you own or are licensed to process.
- Streamlit app behavior (suggested best practice):
  - Process files in memory and do not retain them on the server.
  - Display an on‑screen notice: “Upload only content you have rights to. Files are processed ephemerally and not stored.”
  - Add a brief Terms/Privacy note to the README and app footer if you host it publicly.
- If you receive takedown requests (e.g., DMCA), remove infringing content promptly. Consider a simple contact email in the README/app footer.
- This repository provides code only and no legal advice; consult an attorney if you need specific guidance for a hosted service.

---

## 📜 License
MIT License — see `LICENSE`.
