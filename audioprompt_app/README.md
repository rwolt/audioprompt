AudioPrompt (Streamlit)

Overview
- Two-column Streamlit app for generating "audio prompts" that steer models: pink noise optionally imprinted with a randomized scale-constrained melody, optional spectral focus (bass/guitar/vocal/custom Hz band), rhythmic gating, and now a **Drum MIDI Imprint** layer.
- Drag-and-drop input (optional). The app always generates a prompt; it prepends the prompt to your input if a file is provided.
- Big "Generate Prompt" button updates the audio when pressed; settings are adjustable between runs.
- Downloads: prompt-only and combined (prepended) WAVs with tagged filenames.

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
  - Enable melody: Imprint a randomized melody (scale-constrained) onto pink noise.
  - Enable focus band: Emphasize energy in a preset band (vocal/guitar/bass) or a custom Hz range.
  - Enable rhythmic gate: Apply a syllabic/phrase-like amplitude envelope from the melody events.
- Prompt seconds: Duration of the generated prompt (also used for prepend length).
- Melody (when enabled)
  - Root, Scale: Choose any from the built-in list (includes minor_blues, pentatonic, modes, etc.).
  - BPM: Tempo used for randomized event durations. If you also uploaded a Drum MIDI, match this BPM to the drum track.
  - Range (Low/High MIDI): Register for the melody notes.
  - Step bias / Max leap: Controls stepwise motion vs. larger interval jumps.
  - Rest prob: Fraction of time devoted to rests.
  - Glide prob/frac: Probability and portion of each note gliding toward the next.
  - Vibrato Hz/Depth: Subtle expressive pitch modulation.
- Drum MIDI Imprint (when enabled)
  - Drum MIDI (.mid / .midi): Upload a General MIDI drum track.
  - Per-lane gains: Kick, Snare, Hat, Perc amounts.
- Focus (when enabled)
  - Preset: vocal (approximately 120-3200 Hz), guitar (approximately 80-6000 Hz), bass (approximately 40-300 Hz), or custom.
  - Custom band: Twin-handle Hz slider for low/high cutoff.
  - Imprint gain: Strength of harmonic emphasis around the melody's partials.
  - Harmonics: Number of harmonics in the emphasis mask.
  - BW frac: Fractional bandwidth of harmonic peaks (smaller = sharper pitch focus).
  - Band floor (dB): Attenuation outside the focused band.
  - Band edge sharpness: Steepness of the band edges.

Right column (Output & Seed, Layer Blend, Generate, Outputs)
- Output & Seed
  - Prompt gain (dB): Level applied when prepending to the input track.
  - Fade-in/out (ms): Smooth start/end on the prepended prompt segment.
  - Seed: Controls randomness for pink noise and melody generation. Default is -1 (new random seed each generation). Set to a fixed integer for reproducible results.
- Layer Blend
  - Melody level: Gain for the melody / focus / pink-noise layer.
  - Drum level: Gain for the drum MIDI layer.
- Generate & Outputs
  - Generate Prompt button
  - Prompt: In-page audio player and "Download prompt" button.
  - Combined: In-page audio player and "Download combined" button (enabled if an input file is uploaded).
  - Status messages: Informative notes when only prompt is generated or if format issues occur.

Filename tagging
- Output names include scale, focus, drum, and seed to avoid overwrites and track settings.
  - Combined: <input_stem>_with_prompt_scale-<scale|none>_focus-<preset|band-lo-hi|none>_drum-<drum|nodrum>_seed-<seed>.wav
  - Prompt only: <input_stem>_prompt_scale-<...>_focus-<...>_drum-<drum|nodrum>_seed-<seed>.wav (or "prompt_..." if no input file).

MIDI Drum Map (General MIDI -> lanes)
| Note | Lane | Description |
|------|------|-------------|
| 35, 36 | kick | Acoustic Kick, Bass Drum |
| 37, 38, 39, 40 | snare | Side Stick, Acoustic Snare, Hand Clap, Electric Snare |
| 41, 43, 45, 47, 48, 50 | snare | Low/Mid/High-Tom variants |
| 42, 44, 46 | hat | Closed Hi-Hat, Pedal Hi-Hat, Open Hi-Hat |
| 49, 51, 52, 55, 57, 59 | hat | Crash/Ride Cymbal variants |
| 54, 56, 58, 62-85 | perc | Tambourine, Cowbell, Vibraslap, Conga/Bongo/Claves/etc. |
- Any note not in the table goes to the perc lane.
- The parser reads all tracks because some DAWs export drums across multiple tracks.

Drum MIDI Imprint notes
- Each lane (kick / snare / hat / perc) is generated from band-limited pink noise with velocity-sensitive envelopes.
- Velocity controls both loudness and decay time: ghost notes (low velocity) are softer and shorter; hard hits (high velocity) are louder and longer.
- Use the Layer Blend sliders to balance melody level vs. drum level.
- For drums-only output: disable Melody and Focus, enable Drum MIDI Imprint, and click Generate.

Performance & Limits
- Defaults are tuned for responsiveness: 48 kHz, 4 s prompts, n_fft=2048.
- Recommend prompt seconds 3-6 for clear steer without masking.
- Max prompt seconds capped at 12 by the UI; adjust in code if needed. (Prepend uses prompt length.)
- Input files are resampled to SR; large multi-minute files are not recommended (trim externally).

How it works (core pieces)
- Pink noise: 1/sqrt(f) spectrum generated in the frequency domain.
- Melody imprint: STFT magnitude shaped by time-varying harmonic masks built from an f0 trajectory.
- Randomized melody: scale-constrained note events with step/leap/rest behavior and optional glides.
- Focus band: global EQ mask in log-frequency with soft edges and outside-band floor.
- Gate: time-domain envelope matched to note onsets/offsets.
- Drum MIDI Imprint: parses MIDI note events, maps them to lanes, generates band-limited pink noise per lane, and applies velocity-sensitive AD envelopes.

Troubleshooting
- "Failed to read input audio": Convert to WAV/FLAC/OGG; ensure the app has file read permission.
- macOS privacy: If running locally and reading from protected folders (e.g., Documents), grant Full Disk Access to your terminal/IDE.
- Clipping: If combined output sounds harsh, reduce Prompt gain (dB) or increase fade-in/out.
- Weak steer: Increase Imprint gain and/or Harmonics; narrow BW frac; extend Prompt seconds.
- Drum MIDI upload fails: Try the click-to-browse button instead of drag-and-drop. Ensure the file is a valid .mid / .midi.
- Drums too quiet in mix: Raise the Drum level blend slider, or lower Melody level.

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
  - mididrums.py: MIDI drum track parsing and lane mapping.
  - drumnoise.py: per-lane band-limited pink noise synthesis with velocity envelopes.

License
- MIT - see ../../LICENSE. This app is part of the AudioPrompt repository and is covered by the root license.
