#!/usr/bin/env python3
"""
AudioPrompt: Cross-platform analyzer and prompt generator with Textual TUI.

Pure Python/Numpy/Scipy DSP (no ffmpeg dependency):
  - analyze: build a 1D frequency profile via Welch PSD on mono audio
  - prompt:  generate shaped pink-noise in frequency domain from inverse EQ
  - prepend: concatenate prompt + seed (sample-rate convert if needed)

TUI: run `python audioprompt.py ui` (or with no args) to open the Textual app.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from scipy import signal
import soundfile as sf


# ------------------------------ Core utilities ------------------------------ #


def _ensure_numpy():
    _ = np  # import check


# 24 log-spaced centers used by the original script (31 Hz .. 15 kHz).
DEFAULT_CENTERS = [
    31, 41, 53, 69, 90, 117, 153, 200, 262, 343, 449, 588,
    770, 1008, 1320, 1728, 2263, 2964, 3883, 5085, 6658, 8718, 11417, 14945,
]


@dataclass
class AnalysisResult:
    values_0_255: List[int]  # low->high frequency, length = bands
    avg: float
    eq_entries: List[Tuple[int, float]]  # (frequency, gain_dB) at DEFAULT_CENTERS


def _load_audio_mono(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load audio, convert to mono float32 and resample to target_sr.

    Uses soundfile; supports WAV/FLAC/OGG and more (libsndfile-backed).
    """
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if data.shape[1] > 1:
        x = data.mean(axis=1)
    else:
        x = data[:, 0]
    if sr != target_sr:
        # High-quality polyphase resampling
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = signal.resample_poly(x, up, down).astype(np.float32)
        sr = target_sr
    return x, sr


def _welch_psd(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD for a stable long-term spectrum."""
    nperseg = 4096
    noverlap = int(nperseg * 0.75)
    f, Pxx = signal.welch(
        x,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        detrend=False,
        average="mean",
    )
    return f, Pxx


def _interp_log_space(f_src: np.ndarray, y_src: np.ndarray, *, bands: int, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate y(f) onto log-spaced frequencies between fmin..fmax."""
    f_dst = np.logspace(np.log10(fmin), np.log10(fmax), bands)
    # Avoid zeros for log; clip frequency range
    idx = (f_src >= max(1.0, fmin * 0.8)) & (f_src <= fmax * 1.05)
    f = f_src[idx]
    y = y_src[idx]
    if len(f) < 4:
        raise RuntimeError("Not enough frequency support to compute profile (check input or sample rate)")
    # Interpolate in log-frequency domain; use linear interp on magnitude
    y_dst = np.interp(f_dst, f, y, left=y[0], right=y[-1])
    return f_dst, y_dst


def analyze_profile(
    in_path: Path,
    *,
    bands: int = 1024,
    bins: int = 24,
    sr: int = 48000,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> AnalysisResult:
    """Compute a 1D frequency profile by Welch PSD and log-frequency sampling.

    - Loads audio, converts to mono, resamples to `sr`.
    - Computes Welch PSD for a robust long-term spectrum.
    - Interpolates onto `bands` log-spaced frequencies between fmin..fmax.
    - Normalizes to 0..255 (per-track) for a compact profile vector.
    - Produces 24-band EQ entries at DEFAULT_CENTERS for convenience.
    """
    x, sr_in = _load_audio_mono(in_path, sr)
    if fmax is None:
        fmax = sr_in / 2.0
    f, Pxx = _welch_psd(x, sr_in)
    # Convert to dB for perceptual scaling, then back to linear range for profile normalization
    eps = 1e-20
    P_db = 10.0 * np.log10(np.maximum(Pxx, eps))
    f_dst, y_db = _interp_log_space(f, P_db, bands=bands, fmin=fmin, fmax=fmax)
    # Normalize to 0..255 per track (min->0, max->255)
    y_db_min = float(np.min(y_db))
    y_db_max = float(np.max(y_db))
    if y_db_max - y_db_min < 1e-9:
        norm = np.zeros_like(y_db)
    else:
        norm = (y_db - y_db_min) / (y_db_max - y_db_min)
    values = (norm * 255.0).astype(np.uint8).tolist()
    avg = float(np.mean(values)) if values else 0.0

    # Build EQ entries by sampling the normalized vector at DEFAULT_CENTERS
    eq: List[Tuple[int, float]] = []
    # Build a helper map from f_dst->value for interpolation
    def sample_value_at(freq_hz: float) -> float:
        return float(np.interp(freq_hz, f_dst, norm))  # 0..1

    for f_c in DEFAULT_CENTERS:
        v01 = sample_value_at(f_c) if (fmin <= f_c <= fmax) else 0.0
        g = 6.0 - v01 * 6.0  # match legacy mapping for analyze preview
        g = max(-3.0, min(9.0, g))
        eq.append((int(f_c), float(g)))

    return AnalysisResult(values_0_255=values, avg=avg, eq_entries=eq)


def write_profile(outdir: Path, result: AnalysisResult) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    txt = outdir / "profile.txt"
    json_path = outdir / "profile.json"
    txt.write_text("\n".join(str(v) for v in result.values_0_255) + "\n", encoding="utf-8")
    data = {
        "bins": 24,
        "avg": round(result.avg, 2),
        "values": [result.values_0_255[int(b * len(result.values_0_255) / 24)] for b in range(24)],
        "eq": [{"f": f, "g": g} for (f, g) in result.eq_entries],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return txt, json_path


def eq_entries_string(eq_entries: Sequence[Tuple[int, float]]) -> str:
    # firequalizer gain_entry format: entry(f_Hz,g_dB);entry(...)
    return "".join(f"entry({int(f)},{float(g):.2f});" for f, g in eq_entries)


def _build_inverse_eq_curve(values_0_255: List[int], *, max_gain: float, bands: int, fmin: float, fmax: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build a dense frequency response from 24 centers mapped from the profile values.

    Returns (freqs, gain_linear) for rFFT bin centers up to Nyquist.
    """
    values = np.asarray(values_0_255, dtype=np.float32)
    norm = values / 255.0
    # Prepare dense frequency axis for the output length; we'll interpolate later per-N
    f_dense = np.logspace(np.log10(fmin), np.log10(fmax), 2048)

    # Build 24-point inverse gains and interpolate across f_dense
    centers = np.array(DEFAULT_CENTERS, dtype=np.float32)
    v_centers = np.interp(centers, np.linspace(fmin, fmax, len(norm)), norm)
    g_centers_db = np.clip(max_gain - v_centers * max_gain, -3.0, max_gain)
    g_dense_db = np.interp(f_dense, centers, g_centers_db, left=g_centers_db[0], right=g_centers_db[-1])
    g_dense_lin = 10.0 ** (g_dense_db / 20.0)
    return f_dense, g_dense_lin


def _synthesize_shaped_noise(duration: float, sr: int, f_dense: np.ndarray, g_dense_lin: np.ndarray, *, baseline: str = "pink") -> np.ndarray:
    """Synthesize colored noise directly in the frequency domain and IFFT.

    baseline: 'pink' uses 1/sqrt(f) magnitude; 'white' uses flat.
    """
    n = int(round(duration * sr))
    n = max(n, 1)
    # rFFT frequency bins
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    # Build baseline magnitude
    mag = np.ones_like(freqs)
    if baseline == "pink":
        with np.errstate(divide="ignore"):
            mag = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
        mag[0] = 0.0
    # Interpolate shaping curve onto rFFT bins and multiply
    g_lin_bins = np.interp(freqs, f_dense, g_dense_lin, left=g_dense_lin[0], right=g_dense_lin[-1])
    mag *= g_lin_bins
    # Random phases
    rng = np.random.default_rng()
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    spectrum = mag * np.exp(1j * phases)
    spectrum[0] = 0.0  # no DC
    y = np.fft.irfft(spectrum, n=n).astype(np.float32)
    # Normalize to avoid clipping; conservative headroom
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = (0.9 / peak) * y
    return y


def generate_prompt(
    in_path: Path,
    out_path: Path,
    *,
    duration: float = 6.0,
    max_gain: float = 6.0,
    sr: int = 48000,
    channels: int = 1,
) -> None:
    """Analyze and generate a shaped pink-noise prompt (pure Python)."""
    res = analyze_profile(in_path, bands=1024, bins=24, sr=sr)
    f_dense, g_dense_lin = _build_inverse_eq_curve(
        res.values_0_255, max_gain=max_gain, bands=1024, fmin=20.0, fmax=sr / 2.0, sr=sr
    )
    y = _synthesize_shaped_noise(duration, sr, f_dense, g_dense_lin, baseline="pink")
    if channels == 2:
        y = np.stack([y, y], axis=1)
    sf.write(str(out_path), y, sr)


def prepend_audio(prompt_path: Path, seed_path: Path, out_path: Path, *, sr: int = 48000, channels: int = 2) -> None:
    """Concatenate prompt + seed, matching sample rate and channels (pure Python)."""
    p, sr_p = _load_audio_mono(prompt_path, sr)
    s, sr_s = _load_audio_mono(seed_path, sr)
    # Upmix if needed
    if channels == 2:
        p = np.stack([p, p], axis=1)
        s = np.stack([s, s], axis=1)
    y = np.concatenate([p, s], axis=0)
    sf.write(str(out_path), y, sr)


# ------------------------------ Textual TUI ------------------------------ #

def _ensure_textual():
    try:
        import textual  # noqa: F401
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Textual is not installed. Install with: pip install textual"
        ) from e


def run_tui() -> None:  # pragma: no cover - interactive
    _ensure_textual()
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    # Textual changed TextLog to Log in newer versions; support both.
    from textual.widgets import Button, Footer, Header, Input, Label, Static
    from textual.widgets import DirectoryTree
    from textual.screen import ModalScreen
    try:
        from textual.widgets import TextLog as LogWidget  # type: ignore
    except Exception:  # pragma: no cover
        from textual.widgets import Log as LogWidget  # type: ignore

    class FilePicker(ModalScreen[Path | None]):
        def __init__(self, start: Path | None = None, title: str = "Select a file"):
            super().__init__()
            # Root at filesystem root to allow navigating to parent directories.
            # We ignore the given start for rooting and use it only to hint the title.
            self._start = Path("/")
            self._title = title

        def compose(self) -> ComposeResult:
            yield Vertical(
                Label(self._title),
                DirectoryTree(str(self._start), id="fp_tree"),
                Horizontal(
                    Button("Cancel", id="fp_cancel"),
                ),
                id="fp_panel",
            )

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "fp_cancel":
                self.dismiss(None)

        def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:  # type: ignore[attr-defined]
            try:
                path = Path(event.path)  # Textual >=0.50
            except Exception:
                path = Path(getattr(event, "path", str(event)))
            self.dismiss(path)

    class AnalyzePane(Vertical):
        def compose(self) -> ComposeResult:
            yield Label("Analyze")
            yield Static("Build a frequency profile from audio", classes="help")
            yield Label("Input audio file")
            with Horizontal(classes="row"):
                yield Input(id="an_in")
                yield Button("Browse…", id="an_in_browse", classes="browse")
            yield Label("Bands")
            yield Input(value="1024", id="an_bands")
            yield Label("Output directory (default: .)")
            with Horizontal(classes="row"):
                yield Input(id="an_out")
                yield Button("Browse…", id="an_out_browse", classes="browse")
            yield Button("Run Analyze", id="an_run")

    class PromptPane(Vertical):
        def compose(self) -> ComposeResult:
            yield Label("Prompt")
            yield Static("Generate a shaped pink-noise prompt", classes="help")
            yield Label("Input audio file")
            with Horizontal(classes="row"):
                yield Input(id="pr_in")
                yield Button("Browse…", id="pr_in_browse", classes="browse")
            yield Label("Output WAV path")
            with Horizontal(classes="row"):
                yield Input(id="pr_out")
                yield Button("Browse…", id="pr_out_browse", classes="browse")
            yield Label("Duration (seconds)")
            yield Input(value="6", id="pr_dur")
            yield Label("Max gain (dB)")
            yield Input(value="6", id="pr_mg")
            yield Button("Run Prompt", id="pr_run")

    class PrependPane(Vertical):
        def compose(self) -> ComposeResult:
            yield Label("Prepend")
            yield Static("Concatenate prompt + seed", classes="help")
            yield Label("Prompt WAV path")
            with Horizontal(classes="row"):
                yield Input(id="pp_p")
                yield Button("Browse…", id="pp_p_browse", classes="browse")
            yield Label("Seed WAV path")
            with Horizontal(classes="row"):
                yield Input(id="pp_s")
                yield Button("Browse…", id="pp_s_browse", classes="browse")
            yield Label("Output WAV path")
            with Horizontal(classes="row"):
                yield Input(id="pp_o")
                yield Button("Browse…", id="pp_o_browse", classes="browse")
            yield Button("Run Prepend", id="pp_run")

    class APApp(App):
        CSS = """
        Screen { layout: vertical; }
        #main { height: 1fr; }
        TextLog { height: 1fr; }
        Vertical { padding: 1 2; border: solid #444; }
        Label { margin: 0 0 1 0; }
        Input { margin: 0 1 1 0; width: 1fr; }
        Button { margin-top: 0; }
        .help { color: #AAA; margin: 0 0 1 0; text-style: italic; }
        #fp_panel { padding: 1 2; border: solid #666; width: 80%; height: 80%; }
        .row { layout: grid; grid-columns: 1fr auto; }
        .browse { min-width: 10; }
        """

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                yield AnalyzePane(id="pane_an")
                yield PromptPane(id="pane_pr")
                yield PrependPane(id="pane_pp")
            yield LogWidget(id="log")
            yield Footer()

        def _apply_layout(self) -> None:
            # Stack panes vertically when terminal is narrow
            main = self.query_one("#main")
            try:
                width = self.size.width
            except Exception:
                width = 120
            main.styles.layout = "vertical" if width < 120 else "horizontal"

        def on_mount(self) -> None:  # type: ignore[override]
            self._apply_layout()

        def on_resize(self) -> None:  # type: ignore[override]
            self._apply_layout()

        def _open_file_picker(self, target_input_id: str, title: str = "Select a file") -> None:
            start_val = self.query_one(f"#{target_input_id}", Input).value.strip()
            start = Path(start_val).parent if start_val else Path.cwd()
            def _set_value(result: Path | None) -> None:
                if result is not None:
                    self.query_one(f"#{target_input_id}", Input).value = str(result)
            self.push_screen(FilePicker(start=start, title=title), _set_value)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            log: LogWidget = self.query_one("#log", LogWidget)
            try:
                if event.button.id == "an_run":
                    in_path = Path(self.query_one("#an_in", Input).value.strip())
                    bands = int(self.query_one("#an_bands", Input).value or 1024)
                    outdir_txt = self.query_one("#an_out", Input).value.strip() or "."
                    outdir = Path(outdir_txt)
                    log.write(f"Analyzing {in_path} …")
                    res = analyze_profile(in_path, bands=bands, bins=24)
                    txt, js = write_profile(outdir, res)
                    log.write(f"Wrote {txt}")
                    log.write(f"Wrote {js}")
                elif event.button.id == "an_in_browse":
                    self._open_file_picker("an_in", title="Select input audio file")
                elif event.button.id == "an_out_browse":
                    # Use file picker; user can select any file to fill its directory quickly.
                    self._open_file_picker("an_out", title="Select a file in the output directory")
                elif event.button.id == "pr_run":
                    in_path = Path(self.query_one("#pr_in", Input).value.strip())
                    out_path = Path(self.query_one("#pr_out", Input).value.strip())
                    dur = float(self.query_one("#pr_dur", Input).value or 6)
                    mg = float(self.query_one("#pr_mg", Input).value or 6)
                    log.write(f"Generating prompt from {in_path} → {out_path} …")
                    generate_prompt(in_path, out_path, duration=dur, max_gain=mg)
                    log.write(f"Wrote {out_path}")
                elif event.button.id == "pr_in_browse":
                    self._open_file_picker("pr_in", title="Select input audio file")
                elif event.button.id == "pr_out_browse":
                    self._open_file_picker("pr_out", title="Select output file (will overwrite)")
                elif event.button.id == "pp_run":
                    p = Path(self.query_one("#pp_p", Input).value.strip())
                    s = Path(self.query_one("#pp_s", Input).value.strip())
                    o = Path(self.query_one("#pp_o", Input).value.strip())
                    log.write(f"Prepending {p} + {s} → {o} …")
                    prepend_audio(p, s, o)
                    log.write(f"Wrote {o}")
                elif event.button.id == "pp_p_browse":
                    self._open_file_picker("pp_p", title="Select prompt WAV file")
                elif event.button.id == "pp_s_browse":
                    self._open_file_picker("pp_s", title="Select seed WAV file")
                elif event.button.id == "pp_o_browse":
                    self._open_file_picker("pp_o", title="Select output file (will overwrite)")
            except Exception as e:
                log.write(f"Error: {e}")

    APApp().run()


# ------------------------------ CLI ------------------------------ #


def _cmd_analyze(args: argparse.Namespace) -> int:
    res = analyze_profile(Path(args.input), bands=args.bands, bins=args.bins)
    outdir = Path(args.outdir)
    txt, js = write_profile(outdir, res)
    if args.print_eq:
        print(eq_entries_string(res.eq_entries))
    else:
        print(f"Wrote {txt}")
        print(f"Wrote {js}")
    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    generate_prompt(Path(args.input), Path(args.output), duration=args.duration, max_gain=args.max_gain)
    print(f"Wrote {args.output}")
    return 0


def _cmd_prepend(args: argparse.Namespace) -> int:
    prepend_audio(Path(args.prompt), Path(args.seed), Path(args.output))
    print(f"Wrote {args.output}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:  # noqa: ARG001
    run_tui()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audioprompt", description="Analyze audio and generate frequency-shaped prompts (Textual TUI and CLI)")
    sub = p.add_subparsers(dest="cmd")

    p_an = sub.add_parser("analyze", help="Analyze an audio file to a 1D profile")
    p_an.add_argument("input", help="Input audio path")
    p_an.add_argument("--bands", type=int, default=1024)
    p_an.add_argument("--bins", type=int, default=24)
    p_an.add_argument("--outdir", default=".")
    p_an.add_argument("--print-eq", action="store_true")
    p_an.set_defaults(func=_cmd_analyze)

    p_pr = sub.add_parser("prompt", help="Generate a shaped pink-noise prompt from input audio")
    p_pr.add_argument("input", help="Input audio path")
    p_pr.add_argument("output", help="Output wav path")
    p_pr.add_argument("--duration", type=float, default=6.0)
    p_pr.add_argument("--max-gain", type=float, default=6.0)
    p_pr.set_defaults(func=_cmd_prompt)

    p_pp = sub.add_parser("prepend", help="Concatenate prompt + seed, matching format")
    p_pp.add_argument("prompt", help="Prompt wav path")
    p_pp.add_argument("seed", help="Seed wav path")
    p_pp.add_argument("output", help="Output wav path")
    p_pp.set_defaults(func=_cmd_prepend)

    p_ui = sub.add_parser("ui", help="Launch Textual TUI")
    p_ui.set_defaults(func=_cmd_ui)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    # Default to UI if no args.
    if not argv:
        return _cmd_ui(argparse.Namespace())
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
