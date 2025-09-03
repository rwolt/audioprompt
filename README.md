# audioprompt — Cross‑platform Textual TUI + CLI (Pure Python DSP)

AudioPrompt helps you turn a track’s long‑term spectrum into a short “guidance” audio clip that can bias AI music tools with audio upload/extend features. It now uses pure Python/Numpy/Scipy for DSP (no ffmpeg dependency), so it’s clean, portable, and easy to extend on macOS, Windows, and Linux.

Bulletpoint Overview
- 1D frequency profile: collapses a log spectrogram over time (long‑term spectrum).
- Inverse EQ curve: maps the profile to `firequalizer` gain entries.
- Pink‑noise prompt: generates a shaped noise bed to “open” underused bands.
- Prepend tool: concatenates prompt + seed with matched sample rate/channels.
- Textual TUI: simple panels for Analyze, Prompt, and Prepend with logging.
- CLI parity: same features from the command line.

Requirements
- Python: 3.9+
- Packages: numpy, scipy, soundfile, textual (see `requirements.txt`).

Audio I/O
- Input formats: WAV/FLAC/OGG and more via libsndfile (bundled in wheels for most platforms).
- Output format: WAV.

Install the App
- Using pip (system Python or venv):
  - Create/activate a venv (recommended):
    - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
    - Windows: `py -m venv .venv && .venv\\Scripts\\activate`
  - Install deps: `python -m pip install -r requirements.txt`
- Using conda (works cross‑platform):
  - `conda create -n audioprompt python=3.11 -y`
  - `conda activate audioprompt`
  - `python -m pip install -r requirements.txt`

How to Run
- TUI (recommended): `python audioprompt.py` (or `python audioprompt.py ui`)
- CLI (headless):
  - Analyze: `python audioprompt.py analyze input.wav --bands 1024 --bins 24 --outdir .`
  - Prompt: `python audioprompt.py prompt input.wav out.wav --duration 6 --max-gain 6`
  - Prepend: `python audioprompt.py prepend prompt.wav seed.wav out.wav`

All Options at a Glance (CLI)
- `ui`
  - Launch the Textual TUI.
- `analyze <in.wav> [--bands N] [--bins M] [--outdir DIR] [--print-eq]`
  - Builds a time‑averaged 1D profile. Writes `profile.txt` and `profile.json` to `--outdir`.
  - `--print-eq` prints `firequalizer` entries to stdout in `entry(f,g);…` format.
- `prompt <in.wav> <out.wav> [--duration S] [--max-gain dB]`
  - Generates a pink‑noise prompt, shaped by the inverse EQ from the profile.
- `prepend <prompt.wav> <seed.wav> <out.wav>`
  - Concatenates prompt + seed after auto‑matching sample rate/channels.

Quick Start (most basic)
1) Launch the TUI: `python audioprompt.py`
2) In Analyze, select your `input.wav` and click “Run Analyze” (creates `profile.txt`/`profile.json`).
3) In Prompt, set the same `input.wav`, an output like `prompt.wav`, and click “Run Prompt”.
4) In Prepend, choose `prompt.wav` + your seed (original input) and click “Run Prepend”.

Examples (CLI)
- Analyze with defaults and print EQ entries for manual EQ:
  - `python audioprompt.py analyze input.wav --print-eq`
- Generate stronger shaping and longer prompt:
  - `python audioprompt.py prompt input.wav prompt.wav --duration 8 --max-gain 8`
- Make a subtle, quick prompt:
  - `python audioprompt.py prompt input.wav prompt.wav --duration 5 --max-gain 4`
- Prepend prompt to seed for upload:
  - `python audioprompt.py prepend prompt.wav input.wav upload.wav`

What It Produces
- `profile.txt`: 1024 values (0..255), low→high frequency bands (time‑averaged).
- `profile.json`: metadata, 24 downsampled values, and the EQ entries.
- `prompt.wav`: pink‑noise bed shaped by inverse EQ with a limiter.

Tips
- Start subtle: set `--max-gain 4` to 6 dB; stronger shaping can sound too directive.
- Use WAV for uploads; avoid lossy pre-processing to keep frequency content stable.
- Prompt length: 4–8 seconds is usually enough to “prime” the generation.
- The EQ is intentionally conservative (boosts/fades are clamped). You can experiment by raising `--max-gain`.
 - Inputs with very low bandwidth (e.g., telephone-band audio) will naturally emphasize midrange in the inverse EQ; consider lowering `--max-gain` for those.

Troubleshooting
- pip not found (conda/macOS):
  - Ensure you’re using the env’s Python: `which python` should point to your conda env.
  - Prefer: `python -m pip install -r requirements.txt` (uses the active interpreter’s pip).
  - If pip truly isn’t installed in the env: `conda install pip` (or `mamba install pip`).
  - On macOS with Homebrew Python, `pip3` may exist; still prefer `python3 -m pip ...`.
- “Permission denied” writing files:
  - Check folder write permissions; choose an output path you own.
- TUI won’t launch / Textual missing:
  - Install dependencies: `python -m pip install -r requirements.txt`.
 - soundfile import error:
   - Ensure you installed via `python -m pip install -r requirements.txt`. Wheels bundle libsndfile on most platforms. If building from source, install libsndfile from your OS package manager.

Roadmap
- Optional sweep and pulse prompt types.
- Configurable log‑spaced EQ centers with fmin/fmax and smoothing.
- Stereo analysis modes (L/R/avg/mid/side) and stereo prompt generation.

License
MIT (or your preferred license).
