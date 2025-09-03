AudioPrompt (Streamlit)

Overview
- Two‑column Streamlit app for generating “audio prompts” that steer models: pink noise optionally imprinted with a randomized scale‑constrained melody, plus optional spectral focus (bass/guitar/vocal/custom Hz band) and rhythmic gating.
- Drag‑and‑drop input (optional). The app always generates a prompt; it prepends the prompt to your input if a file is provided.
- Big “Generate Prompt” button updates the audio when pressed; settings are adjustable between runs.
- Downloads: prompt‑only and combined (prepended) WAVs with tagged filenames.

Quick Start
1) Install
   - python -m venv .venv && source .venv/bin/activate
   - pip install -r requirements.txt
2) Run
   - streamlit run app.py
3) Use
   - Left column: upload (optional), set toggles, choose Prompt seconds and melody.
   - Right column: (optional) set Focus and Output settings, press Generate Prompt, then preview/download.

Supported formats
- Input: WAV/FLAC/OGG/AIFF via libsndfile. (MP3/M4A are not guaranteed; convert to WAV for best reliability.)
- Output: WAV (PCM16).

UI Layout & Features
Left column (Controls)
- Input & Output
  - Input audio (optional): Drag and drop a file (styled drop zone). If provided, the prompt is prepended to create a combined output.
  - Sample rate: Default 48000 Hz. All processing runs at this rate; inputs are resampled.
- Toggles
  - Enable melody: Imprint a randomized melody (scale‑constrained) onto pink noise.
  - Enable focus band: Emphasize energy in a preset band (vocal/guitar/bass) or a custom Hz range.
  - Enable rhythmic gate: Apply a syllabic/phrase‑like amplitude envelope from the melody events.
- Prompt seconds: Duration of the generated prompt (also used for prepend length).
- Melody (when enabled)
  - Root, Scale: Choose any from the built‑in list (includes minor_blues, pentatonic, modes, etc.).
  - BPM: Tempo used for randomized event durations.
  - Range (Low/High MIDI): Register for the melody notes.
  - Step bias / Max leap: Controls stepwise motion vs. larger interval jumps.
  - Rest prob: Fraction of time devoted to rests.
  - Glide prob/frac: Probability and portion of each note gliding toward the next.
  - Vibrato Hz/Depth: Subtle expressive pitch modulation.
 
Right column (Focus, Output & Generate, Outputs)
- Focus (optional)
  - Preset: vocal (≈120–3200 Hz), guitar (≈80–6000 Hz), bass (≈40–300 Hz), or custom.
  - Custom band: Twin‑handle Hz slider for low/high cutoff.
  - Imprint gain: Strength of harmonic emphasis around the melody’s partials.
  - Harmonics: Number of harmonics in the emphasis mask.
  - BW frac: Fractional bandwidth of harmonic peaks (smaller = sharper pitch focus).
  - Band floor (dB): Attenuation outside the focused band.
  - Band edge sharpness: Steepness of the band edges.
- Output & Seed
  - Prompt gain (dB): Level applied when prepending to the input track.
  - Fade‑in/out (ms): Smooth start/end on the prepended prompt segment.
  - Seed: Controls randomness for pink noise and melody generation. Default is -1 (new random seed each generation). Set to a fixed integer for reproducible results.
- Generate & Outputs
  - Generate Prompt button
  - Prompt: In‑page audio player and “Download prompt” button.
  - Combined: In‑page audio player and “Download combined” button (enabled if an input file is uploaded).
  - Status messages: Informative notes when only prompt is generated or if format issues occur.

Filename tagging
- Output names include scale, focus, and seed to avoid overwrites and track settings.
  - Combined: <input_stem>_with_prompt_scale-<scale|none>_focus-<preset|band-lo-hi|none>_seed-<seed>.wav
  - Prompt only: <input_stem>_prompt_scale-<...>_focus-<...>_seed-<seed>.wav (or “prompt_...” if no input file).

Performance & Limits
- Defaults are tuned for responsiveness: 48 kHz, 4 s prompts, n_fft=2048.
- Recommend prompt seconds 3–6 for clear steer without masking.
- Max prompt seconds capped at 12 by the UI; adjust in code if needed. (Prepend uses prompt length.)
- Input files are resampled to SR; large multi‑minute files aren’t recommended (trim externally).

How it works (core pieces)
- Pink noise: 1/√f spectrum generated in the frequency domain.
- Melody imprint: STFT magnitude shaped by time‑varying harmonic masks built from an f0 trajectory.
- Randomized melody: scale‑constrained note events with step/leap/rest behavior and optional glides.
- Focus band: global EQ mask in log‑frequency with soft edges and outside‑band floor.
- Gate: time‑domain envelope matched to note onsets/offsets.

Troubleshooting
- “Failed to read input audio”: Convert to WAV/FLAC/OGG; ensure the app has file read permission.
- macOS privacy: If running locally and reading from protected folders (e.g., Documents), grant Full Disk Access to your terminal/IDE.
- Clipping: If combined output sounds harsh, reduce Prompt gain (dB) or increase fade‑in/out.
- Weak steer: Increase Imprint gain and/or Harmonics; narrow BW frac; extend Prompt seconds.

Deploying publicly (Streamlit Community Cloud)
1) Push this folder to GitHub.
2) Create a new Streamlit app, point to app.py, and select the repo.
3) It installs requirements.txt and runs automatically; no secrets are needed.

Code structure
- app.py: Streamlit UI and orchestration.
- src/audioprompt_core/
  - audio.py: audio loading, fades, WAV bytes, filename tagging.
  - melody.py: scales, random melody generation, f0 trajectory.
  - prompt.py: pink noise, melody imprint with optional focus, rhythmic gate.

License
- Choose and add a license for public use if desired.
