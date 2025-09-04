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

## 📜 License
MIT (or your preferred license)

