# audioprompt — Analyze a track and synthesize guidance prompts

audioprompt helps you turn a track’s long‑term spectrum into a short “guidance” audio clip that can bias AI music tools with audio upload/extend features.

- Analyze: produce a 1D frequency profile by time‑averaging a log spectrogram
- Suggest EQ: turn the profile into a simple inverse‑gain firequalizer curve
- Prompt: generate pink‑noise prompts shaped to “open up” underrepresented bands
- Prepend: concatenate the prompt in front of your seed audio for upload

Requirements: ffmpeg (with showspectrum), awk. No Python/JQ required.

Quick start

```bash
# 1) Analyze your seed track
./audioprompt analyze input.wav           # writes profile.txt and profile.json

# 2) Generate a 6s pink‑noise guidance prompt using inverse EQ
./audioprompt prompt input.wav prompt.wav --duration 6

# 3) Prepend prompt to seed for upload
./audioprompt prepend prompt.wav input.wav upload.wav
```

Examples

```bash
# Analyze with 1024 bands (default) and export 24‑band inverse EQ string
./audioprompt analyze input.wav --bands 1024 --bins 24 --print-eq

# Stronger shaping (+8 dB max), slightly longer prompt
./audioprompt prompt input.wav prompt.wav --duration 8 --max-gain 8

# Make a very subtle prompt (max +4 dB)
./audioprompt prompt input.wav prompt.wav --duration 5 --max-gain 4
```

What it produces
- profile.txt: 1024 values (0..255), low→high bands
- profile.json: metadata plus downsampled bins
- prompt.wav: pink/noise bed shaped by a smooth inverse EQ curve

Notes
- The “inverse” is conservative by default (clamped −3..+6 dB); tweak with --max-gain.
- Start subtle (−18 to −12 dB RMS prompt). The model should “hear” it without overfitting noise.
- You can also compare to a reference track (future: reference mode), or generate tone‑pulse prompts.

License: MIT (add your own if you prefer).

