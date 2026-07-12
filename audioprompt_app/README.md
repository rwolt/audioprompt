# AudioPrompt (Streamlit) — App Manual

> This is the **technical manual** for the Streamlit app: every control, the MIDI drum map, configuration, and how the DSP works. For the project overview and quick start, see the [root README](../README.md) ([日本語](../README.ja.md)).

A two-column Streamlit app for generating **audio prompts** that steer AI music models: pink noise optionally imprinted with a randomized scale-constrained melody, optional spectral focus (bass/guitar/vocal/custom Hz band), rhythmic gating, a **Drum MIDI Imprint** layer, and a **Bass MIDI Imprint** layer with pitch-bend and velocity support.

- Drag-and-drop input audio (optional). The app always generates a prompt; if you provide a file, the prompt is **prepended** to it to create a combined output.
- Press **Generate Prompt** to render; settings are adjustable between runs.
- Download prompt-only and combined WAVs with tagged filenames.

## Quick Start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Then in the app:

1. **Left column** — upload input audio (optional), enable Melody / Drum MIDI / Bass MIDI layers, and set Focus.
2. **Right column** — choose the prompt length, press **Generate Prompt**, then preview and download.

### Supported formats

- **Input:** WAV / FLAC / OGG / AIFF via libsndfile. (MP3/M4A are not guaranteed; convert to WAV for best reliability.)
- **Output:** WAV (PCM16).

## Language & accessibility

- **UI languages:** English and 日本語. The app defaults to your browser's language on first visit (via `st.context.locale`) and can be switched anytime with the **Language / 言語** selector at the top of the page. The choice persists in the URL — `?lang=ja` / `?lang=en` make shareable language-specific links.
- **Show help as visible text** (checkbox at the top, or `?help=text`): renders every control's explanation as plain text under the control instead of a hover tooltip. Streamlit tooltips are hover-only and not exposed to screen readers or browser translation, so this mode is the accessible and fully-translatable alternative. It also gives the Seed field a real visible label.
- **How it's built:** all UI strings live in [`ui_strings.py`](ui_strings.py) (English literal → translation, gettext-style with English fallback), and [`i18n.py`](i18n.py) wraps the Streamlit widgets so labels, tooltips, and option names translate without touching call sites. `tests/check_i18n_coverage.py` verifies every string in `app.py` has a translation entry.
- **Adding a language:** add a table to `ui_strings.py` and register it in `TABLES` — the selector, URL param, and locale detection pick it up automatically. Translation corrections from native speakers are very welcome.
- One known trade-off: switching language mid-session resets most controls to defaults (Streamlit widget identity is label-based). Pick your language first.
- **Download filename tags stay in English in every language** — deliberately. They're compact parameter codes (`r-e_s-minpent_mb-96…`) meant for reproducibility and cross-platform-safe ASCII filenames, not prose.

## Configuration (environment variables)

Both are optional. With neither set, the app runs fully unrestricted and collects nothing — this is the default for local use and self-hosting.

| Variable | Effect when set | Default (unset) |
|----------|-----------------|-----------------|
| `AUDIOPROMPT_MAX_SECONDS` | Caps the resolved prompt length (all length modes). Over-cap renders are blocked with a warning — never truncated. The public demo uses `60`. | No cap |
| `GOATCOUNTER_URL` | Injects a [GoatCounter](https://www.goatcounter.com/) analytics snippet pointed at this endpoint (aggregate page views + referrers; no cookies, no personal data). | No analytics, nothing phones home |

## UI Reference

### Left column — layers & focus

**Input & sample rate**

- **Input audio (optional):** drag and drop. If provided, the prompt is prepended to create a combined output.
- **Sample rate:** default 48000 Hz. All processing runs at this rate; inputs are resampled.

**Melody** (when enabled)

- **Root / Scale:** any of the built-in scales (modes, pentatonics, blues, etc.).
- **BPM:** tempo for randomized note durations. If you uploaded drum MIDI, match this to the drum track.
- **Melody character:** harmonic color preset — neutral, warm, voice, reed, bell, pluck, bright, or wide (7-cent detune spread).
- **Note shape / Note decay:** envelope feel for imprinted notes; lower decay = staccato.
- **Advanced:** MIDI range (low/high), step bias vs. leaps, rest probability, glides, vibrato, root drone level, imprint gain, harmonics, **Harmonic bandwidth** (the main textural control — the width of each harmonic peak as a fraction of its frequency, so 0.01 keeps energy within ±1% of each harmonic; narrow = tight pitch instruction, wide = diffuse), and **noise floor (dB)** to attenuate non-harmonic static.
- **Vowel character (expander):** off by default (vowel-neutral buzz gives the AI freedom). On = biases the output toward voice-like vowel color in a chosen language — English, Japanese, or Spanish. See [Vowel imprint & languages](#vowel-imprint--languages) for what this does and doesn't do.

**Drum MIDI Imprint** (when enabled)

- **Drum MIDI (.mid/.midi):** upload a General MIDI drum track. Detected BPM and length are shown on upload.
- **Per-lane gains:** Kick / Snare / Hat / Perc amounts (broad noise lanes, not sample instruments).
- **Drum character / Snare tune / Drum decay:** compact tone, tuning, and envelope-length controls.
- **Match melody BPM / Independent drum BPM / Use detected BPM:** tempo targeting. New uploads start at their detected MIDI BPM.
- **Loop drums to prompt length:** repeats the MIDI when the prompt is longer than the region.
- **Trim DAW export padding:** removes the empty bars that appear when the exported region didn't start at bar 1 of the project; silence you wrote into the region (e.g. drums resting for the first bars of a loop) is preserved. See [Leading silence in MIDI exports](#leading-silence-in-midi-exports).

**Bass MIDI Imprint** (when enabled)

- **Bass MIDI (.mid/.midi):** pitch bend and note velocity are preserved. Detected BPM and length are shown on upload.
- **Bass character:** Upright, Fingerstyle, Picked, Synth, or Sub — changes harmonic balance and frequency band.
- **Note shape base / Decay offset:** envelope controls before velocity is applied.
- **Pitch bend range:** must match the pitch-wheel range of the instrument that generated the MIDI. When the file declares its range (an RPN "pitch bend sensitivity" message — Logic's Bass Player exports include one), the slider defaults to the detected value and a caption confirms it; otherwise set it by hand (12, 24, and 48 semitones are common for slide-capable bass patches). If slides come out too shallow or too extreme, this number is wrong.
- **Tempo, looping, and trim-silence controls:** same pattern as the drum layer; the bass trim additionally preserves a pickup written on the first note. See [Leading silence in MIDI exports](#leading-silence-in-midi-exports).
- **Advanced:** bass imprint gain, harmonic bandwidth, and noise floor (dB).

**Focus** (when enabled)

- **Preset:** vocal (~120–3200 Hz), guitar (~80–6000 Hz), bass (~40–300 Hz), or custom twin-handle Hz band.
- **Band floor (dB):** attenuation outside the focused band.
- **Band edge sharpness:** sigmoid rolloff steepness (6 = gentle, 12 = moderate, 24 ≈ near-brick-wall).
- **Bass Roll-Off:** 25 Hz high-pass on the prompt noise (on by default); does not EQ your uploaded audio.

### Right column — generate & outputs

- **Generate Prompt** button and **Seed** (default −1 = new random seed each generation; set a fixed integer for reproducible results).
- **Prompt length:** *Match bass MIDI*, *Match drum MIDI* (when the respective MIDI is loaded; region length adjusted to target BPM), or *Manual seconds* (1–20 s). The resolved length is shown beneath the control.
- **Prompt Level:** auto gain (match input audio RMS with a relative dB offset) or manual prompt gain, plus fade-in/out.
- **Layer Blend:** Melody / Drum / Bass level sliders — layers are loudness-matched to the same RMS before these gains apply.
- **Outputs:** in-page players and download buttons for Prompt and (with input audio) Combined.

## Filename tagging

Output names encode the key generation settings:

- Prompt only: `ap_prompt_r-e_s-minpent_mb-96_ms-1234_db-172_ds-1235.wav`
- Combined: `my-song-take-1_combined_r-e_s-minpent_mb-96_ms-1234_db-172_ds-1235.wav`

Tags: `r` = root, `s` = scale, `mb` = melody BPM, `ms` = melody/noise seed, `db` = drum BPM, `ds` = drum seed, `len` = length. Focus appears only when enabled (`f-voc`, `f-b120-3200`); character only when non-neutral (`ch-voice`); vowel imprint when enabled (`vow-EN-s60-a2` = English, strength 60%, accent period 2; `vow-JA-s60` / `vow-ES-s60` for Japanese/Spanish, which have no accent period); `bw-N` when Harmonic bandwidth differs from the 0.01 default.

## MIDI drum map (General MIDI → lanes)

AudioPrompt follows General MIDI drum note numbers. Use the MIDI number as the source of truth — octave labels vary by DAW, so MIDI 36 may appear as C1, C2, or another label depending on the piano-roll setting.

| MIDI notes | Typical piano-roll labels | Lane | Typical source |
|------------|---------------------------|------|----------------|
| 35, 36 | B0/C1 or B1/C2 | kick | Acoustic Kick, Bass Drum |
| 37, 38, 39, 40 | C#1–E1 or C#2–E2 | snare | Side Stick, Acoustic Snare, Clap, Electric Snare |
| 41, 43, 45, 47, 48, 50 | varies (toms) | snare | Toms currently share the snare/body lane |
| 42, 44, 46 | F#1/G#1/A#1 or F#2/G#2/A#2 | hat | Closed, Pedal, Open Hi-Hat |
| 49, 51, 52, 55, 57, 59 | varies (cymbals) | hat | Crash/Ride/Splash share the hat lane |
| 54, 56, 58, 62–85, all others | varies | perc | Tambourine, Cowbell, Congas, etc. — and any unmapped note |

Notes:

- Every note not matched by the kick/snare/hat rows routes to the **perc** lane, including non-GM notes.
- The parser reads **all tracks**, since some DAWs export drums across multiple tracks.
- Quickest setup from a DAW drummer / session-player feature: convert the generated region to MIDI and leave notes on their GM drum pitches. Start with kick/snare/hat, then use Perc amount for auxiliary percussion.

## Leading silence in MIDI exports

Both MIDI layers have a **Trim DAW export padding** switch (on by default). The problem it solves: DAWs export MIDI with **tick 0 at bar 1 of the project**, not at the start of the exported region. If your region sat at bar 2 of the arrangement, the file arrives with a bar of silence that was never part of the music — and in a 3–6 s prompt, one bar of dead air is a third or more of the steer.

The rule trim follows:

- Logic marks the region's true start inside the file (a tempo/SMPTE meta stamp at the region position). Everything **before** that marker is export padding and is removed.
- Silence **after** the marker is music and is preserved: a bass entering at bar 5 of an 8-bar loop keeps its 4-bar rest, and a pickup written on the first note keeps its offset.
- Files without any marker fall back to simpler heuristics — the drum layer shifts the earliest hit to zero; the bass layer strips only lead-in carried by non-note messages (some Logic Bass Player exports park a controller event a bar before the first note).
- Turn trim **off** to use the file's raw timing exactly as exported.

## How it works (core pieces)

- **Pink noise:** 1/√f spectrum generated in the frequency domain.
- **Melody imprint:** STFT magnitude shaped by time-varying harmonic masks built from an f0 trajectory.
- **Randomized melody:** scale-constrained note events with a weighted random walk — leaps prefer the root and characteristic scale intervals.
- **Focus band:** global EQ mask in log-frequency with soft edges and an outside-band floor.
- **Rhythmic gate:** time-domain velocity-sensitive envelope matched to note onsets/offsets.
- **Drum MIDI Imprint:** maps MIDI events to lanes, generates band-limited pink noise per lane with velocity-sensitive AD envelopes (velocity controls both loudness and decay), then applies tone/tune/decay presets. Drum BPM is a *target tempo* — event times are scaled, not pitch-shifted.
- **Bass MIDI Imprint:** parses notes with pitch bend into a continuous f0 trajectory (legato slides and bend curves included) and imprints it via the same STFT mask engine as the melody.
- **Vowel imprint (optional):** resolves a per-note vowel plan into a per-frame F1/F2/F3 formant envelope (resonant peaks with glides between targets) multiplied into the STFT magnitude. When off, output is unchanged. See the next section for details.

## Vowel imprint & languages

The **Vowel character** expander (inside Melody) shapes the melody imprint toward *sung vowel sounds*. A plain harmonic stack has pitch but no vowel identity — it reads as a buzzy "uh". Turning on **Imprint vowels** multiplies vowel resonances (the F1/F2/F3 formant peaks that make "ee" sound different from "ah") into the spectrum, note by note, with speech-like glides between targets.

**What it does NOT do — read this before filing a bug:**

- It adds **vowels only** — no consonants, no words. It cannot make the AI sing specific lyrics.
- It is a **gentle acoustic bias, not a guarantee**. It nudges the AI toward voice-like output in the selected language's vowel space, but the AI tool's own text prompt (style, language, lyrics) has a much bigger influence on what you get. Think of it as tilting the odds, not steering the wheel.
- **Off is a valid choice** — the default vowel-neutral buzz gives the AI the most freedom and often produces better instrumental results.

**Language presets.** Each preset pairs a vowel inventory with that language's rhythm rule:

| Language | Vowels | Rhythm rule |
|----------|--------|-------------|
| English | 7 full vowels + schwa (Peterson–Barney averages) | Stress-timed: accented notes get a full vowel, weak notes **reduce to schwa**. The **Stress pattern** control (2 = strong-weak, 3 = strong-weak-weak, 4) sets the accent spacing — this strong/reduced alternation is the acoustic signature of English. |
| Japanese | The five vowels a-i-u-e-o, including the brighter **unrounded Japanese u** ([ɯ]) | Mora-timed: **every note keeps a full vowel** — Japanese has no schwa reduction, so the Stress pattern control is hidden. |
| Spanish | The five Spanish vowels a-e-i-o-u | Syllable-timed: **every note keeps a full vowel** — no reduction, Stress pattern hidden. |

**Vowel strength** (0–1) dials the effect from a subtle hint to a strong vowel character; start around 0.6 — very high values can sound robotic.

Other languages: syllable- and mora-timed languages without vowel reduction (Korean, Italian, Mandarin, Hindi/Urdu, …) are closest to the Japanese/Spanish pattern; stress-timed languages with reduction (German, Russian) are closest to English. More presets are easy to add — the language data lives in one table in `src/audioprompt_core/formants.py`.

## Performance & limits

- Defaults are tuned for responsiveness: 48 kHz, 4 s prompts, n_fft = 2048.
- 3–6 s prompts give a clear steer without masking.
- Manual prompt seconds are capped at 20 by the UI; *Match drum MIDI* / *Match bass MIDI* can produce longer prompts when the uploaded region is longer. If `AUDIOPROMPT_MAX_SECONDS` is set (e.g. the public demo's 60 s), over-cap renders are blocked with a clear warning.
- Input files are resampled to the working sample rate; trim large multi-minute files externally.

## Troubleshooting

- **"Failed to read input audio":** convert to WAV/FLAC/OGG; ensure the app has file read permission.
- **macOS privacy:** if reading from protected folders (e.g. Documents), grant Full Disk Access to your terminal/IDE.
- **Clipping:** reduce Prompt gain (dB) or increase fade-in/out.
- **Weak steer:** increase Imprint gain and/or Harmonics; narrow the Harmonic bandwidth; extend the prompt length.
- **MIDI upload fails:** try click-to-browse instead of drag-and-drop; ensure a valid `.mid`/`.midi` file.
- **Drums too quiet in the mix:** raise the Drum level blend slider, or lower Melody level.

## Deploying publicly

1. Push the repo to GitHub and point your host (Railway, Streamlit Community Cloud, etc.) at `audioprompt_app/app.py`.
2. It installs `requirements.txt` and runs; no secrets are required.
3. Recommended env vars for a public demo: `AUDIOPROMPT_MAX_SECONDS=60` and `GOATCOUNTER_URL=<your GoatCounter endpoint>` (see Configuration above).

The app does not store uploaded audio or any user data. Analytics exist only when the operator sets `GOATCOUNTER_URL`.

## Code structure

- `app.py` — Streamlit UI and orchestration.
- `src/audioprompt_core/`
  - `audio.py` — audio loading, fades, WAV bytes.
  - `melody.py` — scales, random melody generation, f0 trajectory.
  - `prompt.py` — pink noise, melody imprint with optional focus, rhythmic gate.
  - `mididrums.py` — MIDI drum track parsing and lane mapping.
  - `drumnoise.py` — per-lane band-limited pink noise synthesis with velocity envelopes.
  - `midibass.py` — bass MIDI parsing with pitch bend, legato slides, and BPM scaling.
  - `formants.py` — English vowel formant tables and STFT formant envelopes.

## License

MIT — see [`../LICENSE`](../LICENSE). This app is part of the AudioPrompt repository and is covered by the root license.
