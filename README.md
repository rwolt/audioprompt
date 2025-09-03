# AudioPrompt — Streamlit App, Notebook, and CLI

AudioPrompt helps you create short “guidance” audio clips to bias AI music tools that accept an audio prompt or seed. It uses pure Python (NumPy/SciPy/soundfile) and ships in three forms:
- Streamlit app (local or Streamlit Community Cloud)
- Jupyter notebook (`audioprompt_workbook.ipynb`)
- Minimal CLI commands

Supported formats
- Input: WAV/FLAC/OGG/AIFF via libsndfile
- Output: WAV

Install
- Python 3.9+
- Create/activate a virtual environment (recommended), then:
  - python -m pip install -r requirements.txt

Ways to use AudioPrompt

1) Streamlit app (recommended)
- Run locally:
  - cd audioprompt_app
  - python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
  - pip install -r requirements.txt
  - streamlit run app.py
- Deploy to Streamlit Community Cloud:
  - Push this repo to GitHub
  - Create an app on Streamlit Community Cloud and point it to `audioprompt_app/app.py`
  - It installs requirements and serves your app automatically

What the app does
- Generates a pink‑noise prompt, optionally imprinted with a randomized, scale‑constrained melody
- Optional spectral focus (vocal/guitar/bass/custom Hz band) and rhythmic gate
- Drag‑and‑drop input (optional) and prepends the prompt to create a combined output
- Downloads: prompt‑only and combined WAVs with tagged filenames (scale/focus/seed)

2) Notebook (`audioprompt_workbook.ipynb`)
- Open audioprompt_workbook.ipynb and run top to bottom
- Flip the toggles to switch between straight pink noise, focused bands, and scale‑driven melodies
- Set INPUT_AUDIO to your track to prepend and export; or generate prompt‑only clips
- The workbook mirrors the Streamlit app behavior and includes the same features

3) CLI (basic)
- Analyze to a long‑term profile (optional):
  - python audioprompt.py analyze input.wav --bands 1024 --bins 24 --outdir .
- Generate a shaped pink‑noise prompt from an input track:
  - python audioprompt.py prompt input.wav prompt.wav --duration 6 --max-gain 6
- Prepend an existing prompt to a seed track:
  - python audioprompt.py prepend prompt.wav input.wav upload.wav

Tips
- Prompt length: 3–6 seconds is usually enough to “prime” generation
- Use WAV/FLAC inputs for maximum reliability
- If the prompt masks the start of the track, reduce prompt gain or shorten the length

Project layout
- audioprompt_app/ — Streamlit app (two‑column UI, dark theme, drag‑drop upload)
- audioprompt_app/src/audioprompt_core/ — shared core (audio, melody, prompt)
- audioprompt_workbook.ipynb — shareable workbook version of the workflow
- audioprompt.py — CLI (analyze/prompt/prepend)

License
MIT (or your preferred license)
