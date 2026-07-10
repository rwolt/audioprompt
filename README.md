# AudioPrompt

**[日本語版 README はこちら → README.ja.md](README.ja.md)**

Create short, steerable "audio prompts" to nudge AI music tools. AudioPrompt imprints a scale-constrained melody onto pink noise, adds optional drum and bass layers from your MIDI files, emphasizes vocal/guitar/bass frequency bands, and can prepend the prompt to your track for upload.

**Live demo:** [https://audioprompt.streamlit.app/](https://audioprompt.streamlit.app/)

📖 **Full app manual** (every control, the MIDI drum map, how the DSP works): [`audioprompt_app/README.md`](audioprompt_app/README.md)

## ✨ Features

- 🧠 Scale-driven melody imprint — randomized, in-key, with vibrato/glides and tone presets
- 🥁 **Drum MIDI Imprint** — upload a General MIDI drum track for a velocity-sensitive rhythmic noise layer
- 🎸 **Bass MIDI Imprint** — upload a bass MIDI track with pitch bend and velocity for a continuous bass layer
- 🎚️ Spectral focus (vocal/guitar/bass presets or custom Hz band) and rhythmic gating
- 🗣️ **Vowel imprint** — optional sung-vowel formant shaping with English, Japanese, and Spanish presets
- 🌏 **UI in English and 日本語** — auto-detected from your browser, switchable anytime
- ♿ **Help as visible text** — accessibility mode that shows every explanation as readable text instead of hover tooltips
- 📎 Drag-and-drop input; tagged WAV downloads; pure Python DSP (NumPy/SciPy/soundfile)

## 🚀 Run the app

```bash
cd audioprompt_app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then: drop in audio (optional), set melody/drum/bass layers, press **Generate Prompt**, and download the tagged WAVs. 3–6 second prompts steer clearly without masking. See the [full manual](audioprompt_app/README.md) for every control.

## 🌏 Languages & accessibility

- The UI is available in **English** and **日本語**. The app detects your browser language on first visit; switch anytime with the selector at the top, or link directly with `?lang=ja` / `?lang=en`.
- **Show help as visible text** (top of the page, or `?help=text`) renders every control's explanation as plain text under the control instead of hover tooltips — recommended for screen readers, and it also makes all explanations translatable.
- Translations were machine-drafted and reviewed for natural audio-production terminology — **corrections from native speakers are very welcome** (all strings live in [`audioprompt_app/ui_strings.py`](audioprompt_app/ui_strings.py)).

## 🧪 CLI (headless)

A minimal CLI for analysis/prompt/prepend lives at the repo root:

```bash
python audioprompt.py analyze <in.wav> [--bands N] [--bins M] [--print-eq]
python audioprompt.py prompt  <in.wav> <out.wav> [--duration S] [--max-gain dB]
python audioprompt.py prepend <prompt.wav> <seed.wav> <out.wav>
```

## 📁 Project layout

- `audioprompt_app/` — the Streamlit app ([manual](audioprompt_app/README.md))
- `audioprompt_app/src/audioprompt_core/` — DSP core (pink noise, STFT imprint, MIDI parsing, formants)
- `audioprompt.py` — CLI

## 🛡️ Legal & content use

- Upload only content you have the rights and permissions to use.
- By using the app, you confirm you have permission to process any uploaded audio.
- This project is provided "as is" without warranties; see the MIT license.

## 📄 Terms & privacy

- You retain all rights to your audio. You grant permission to process your uploaded file(s) for the purpose of generating prompts.
- Do not upload third-party copyrighted material without authorization.
- This app does not store uploaded audio or user data. By default the app collects no analytics at all — if you run it yourself (locally or self-hosted), nothing is tracked and nothing phones home unless you set the optional `GOATCOUNTER_URL` environment variable to point at your own analytics endpoint.
- The public demo sets `GOATCOUNTER_URL` and uses [GoatCounter](https://www.goatcounter.com/) for privacy-friendly aggregate analytics: page views and referrers only. No cookies, no personal data, and no individual or cross-site tracking.

## 📜 License

MIT License — see `LICENSE`.
