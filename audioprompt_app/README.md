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
- Prompt length: Without drum MIDI, use the Prompt seconds slider. With drum MIDI, Match drum MIDI is the default and uses the uploaded MIDI region length adjusted to the target drum BPM; Manual seconds is still available up to 20 seconds.
- Melody (when enabled)
  - Root, Scale: Choose any from the built-in list (includes minor_blues, pentatonic, modes, etc.).
  - BPM: Tempo used for randomized event durations. If you also uploaded a Drum MIDI, match this BPM to the drum track.
  - Melody character: Simple harmonic-color presets for the imprinted melody. Voice/reed/bell/pluck/bright/wide change the tone without exposing low-level mask settings.
  - Note shape: Sets the rhythmic envelope feel for imprinted notes. Tight and pluck are shorter; smooth is softer and more legato.
  - Range (Low/High MIDI): Register for the melody notes.
  - Step bias / Max leap: Controls stepwise motion vs. larger interval jumps.
  - Rest prob: Fraction of time devoted to rests.
  - Glide prob/frac: Probability and portion of each note gliding toward the next.
  - Vibrato Hz/Depth: Subtle expressive pitch modulation.
- Drum MIDI Imprint (when enabled)
  - Drum MIDI (.mid / .midi): Upload a General MIDI drum track exported from your DAW or drum sequencer.
  - Per-lane gains: Kick, Snare, Hat, Perc amounts. These are broad noise lanes, not separate sample instruments.
  - Drum character: Clean, tight, deep, bright, or breakbeat lane presets. These shift lane bands, decay, and light drive under the hood.
  - Snare tune: Shifts the snare noise band by semitones before synthesis, so the snare can sit lower or higher without changing MIDI note mapping.
  - Drum decay: Global envelope length for the drum layer. Lower values are tighter; higher values ring longer.
  - Match melody BPM: Uses the Melody BPM as the drum target tempo.
  - Use detected BPM: Resets the independent drum BPM to the tempo found in the uploaded MIDI.
  - Independent drum BPM: Independent target tempo for the uploaded MIDI when Match melody BPM is off. New uploads start at their detected MIDI BPM, so unchecking Match melody BPM returns to the original groove tempo.
  - Loop drums to prompt length: Repeats the uploaded MIDI when the prompt is longer than the MIDI region. Turn it off to let drums stop after the uploaded MIDI ends.
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
- Output names use compact tags for the most important generation settings.
  - Prompt only: ap_prompt_r-e_s-minpent_mb-96_ms-1234_db-172_ds-1235.wav
  - Combined: my-song-take-1_combined_r-e_s-minpent_mb-96_ms-1234_db-172_ds-1235.wav
  - Tags: r = root, s = scale, mb = melody BPM, ms = melody/noise seed, db = drum BPM, ds = drum seed. Focus appears only when enabled, e.g. f-voc or f-b120-3200. Character appears only when non-neutral, e.g. ch-voice.

MIDI Drum Map (General MIDI -> lanes)
- AudioPrompt follows General MIDI drum note numbers. Use the MIDI number as the source of truth: octave labels vary by DAW, so MIDI 36 may appear as C1, C2, or another octave label depending on the piano-roll setting.
| MIDI notes | Possible piano-roll labels | Lane | Typical source |
|------------|-------------------------|------|----------------|
| 35, 36 | Often B0/C1 or B1/C2 | kick | Acoustic Kick, Bass Drum |
| 37, 38, 39, 40 | Often C#1/D1/D#1/E1 or C#2/D2/D#2/E2 | snare | Side Stick, Acoustic Snare, Clap, Electric Snare |
| 41, 43, 45, 47, 48, 50 | Toms; labels vary by octave setting | snare | Tom notes currently share the snare/body lane |
| 42, 44, 46 | Often F#1/G#1/A#1 or F#2/G#2/A#2 | hat | Closed, Pedal, Open Hi-Hat |
| 49, 51, 52, 55, 57, 59 | Cymbals; labels vary by octave setting | hat | Crash/Ride/Splash cymbals share the hat lane |
| 54, 56, 58, 62-85 | varies | perc | Tambourine, Cowbell, Congas, Bongos, Claves, etc. |
- Any note not in the table goes to the perc lane so unexpected drum hits still make sound.
- The parser reads all tracks because some DAWs export drums across multiple tracks.
- For the quickest setup from Logic Session Drummer, export/convert the drummer region to MIDI and leave the notes on their General MIDI drum pitches. Start with kick/snare/hat, then use Perc amount if your groove includes auxiliary percussion.

Drum MIDI Imprint notes
- Each lane (kick / snare / hat / perc) is generated from its own band-limited pink noise source with velocity-sensitive envelopes.
- Perc is a real fourth noise lane for miscellaneous percussion, but it is intentionally generic in this prototype.
- Velocity controls both loudness and decay time: ghost notes (low velocity) are softer and shorter; hard hits (high velocity) are louder and longer.
- Drum character controls are deliberately compact: they prove tone/tuning/decay control without adding a full synthesizer panel.
- Drum BPM is a target tempo, not pitch shifting. A 100 BPM MIDI file rendered at 125 BPM has its event times scaled by 1.25x.
- Match drum MIDI prompt length uses the MIDI region length from the longest track, then adjusts that length to the target drum BPM.
- Loop drums repeats the parsed MIDI event pattern when the prompt is longer than the MIDI region; disable it when you want the drums to stop after the uploaded MIDI ends.
- Use the Layer Blend sliders to balance melody level vs. drum level.
- For drums-only output: disable Melody and Focus, enable Drum MIDI Imprint, and click Generate.

Performance & Limits
- Defaults are tuned for responsiveness: 48 kHz, 4 s prompts, n_fft=2048.
- Recommend prompt seconds 3-6 for clear steer without masking.
- Manual prompt seconds are capped at 20 by the UI. Match drum MIDI can produce longer prompts when the uploaded MIDI region is longer.
- Input files are resampled to SR; large multi-minute files are not recommended (trim externally).

How it works (core pieces)
- Pink noise: 1/sqrt(f) spectrum generated in the frequency domain.
- Melody imprint: STFT magnitude shaped by time-varying harmonic masks built from an f0 trajectory.
- Melody character: musician-facing presets alter harmonic weights, note envelope shape, and optional doubled-mask spread.
- Randomized melody: scale-constrained note events with step/leap/rest behavior and optional glides.
- Focus band: global EQ mask in log-frequency with soft edges and outside-band floor.
- Gate: time-domain envelope matched to note onsets/offsets.
- Drum MIDI Imprint: parses MIDI note events, maps them to lanes, generates band-limited pink noise per lane, applies velocity-sensitive AD envelopes, then applies compact tone/tune/decay presets.

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
