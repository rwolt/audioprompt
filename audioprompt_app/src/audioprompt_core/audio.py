from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _as_readable(file_or_bytes: Union[str, Path, bytes, bytearray, BytesIO, object]):
    """Return something soundfile can read: path str or file-like with read/seek."""
    if isinstance(file_or_bytes, (str, Path)):
        return str(file_or_bytes)
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return BytesIO(file_or_bytes)
    # Streamlit's UploadedFile has .read()/.seek()
    if hasattr(file_or_bytes, "read"):
        return file_or_bytes
    raise TypeError("Unsupported input type for audio loading.")


def load_audio_mono(file_or_bytes: Union[str, Path, bytes, bytearray, BytesIO, object], target_sr: int) -> Tuple[np.ndarray, int]:
    """Load audio, mix to mono float32, resample to target_sr.

    Supported formats depend on libsndfile (WAV/FLAC/OGG, sometimes AIFF). MP3/M4A are not
    guaranteed unless your environment's libsndfile supports them. For public apps, prefer
    WAV/FLAC uploads.
    """
    f = _as_readable(file_or_bytes)
    data, sr = sf.read(f, always_2d=True, dtype="float32")
    x = data.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return x, sr


def apply_fades(y: np.ndarray, sr: int, fade_in_ms: int = 10, fade_out_ms: int = 50) -> np.ndarray:
    y = y.astype(np.float32, copy=True)
    fi = max(0, int(sr * fade_in_ms / 1000.0))
    fo = max(0, int(sr * fade_out_ms / 1000.0))
    if fi > 0:
        y[:fi] *= np.linspace(0.0, 1.0, fi, endpoint=False, dtype=np.float32)
    if fo > 0:
        y[-fo:] *= np.linspace(1.0, 0.0, fo, endpoint=True, dtype=np.float32)
    return y


def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Encode PCM16 WAV into bytes for download or in-page audio preview.

    When writing to a file-like object (BytesIO), libsndfile requires the format
    to be specified explicitly (can't infer from filename).
    """
    bio = BytesIO()
    sf.write(bio, y.astype(np.float32, copy=False), sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def tag_suffix(enable_melody: bool, melody_scale: str, enable_focus: bool, focus_preset: Union[str, None], focus_band: Union[Tuple[int, int], None], seed: int, output_suffix: str) -> str:
    scale_tag = melody_scale if enable_melody else "none"
    if enable_focus:
        if isinstance(focus_preset, str) and focus_preset:
            focus_tag = focus_preset
        elif focus_band is not None:
            lo, hi = focus_band
            focus_tag = f"band-{int(lo)}-{int(hi)}"
        else:
            focus_tag = "custom"
    else:
        focus_tag = "none"
    return f"{output_suffix}_scale-{scale_tag}_focus-{focus_tag}_seed-{seed}"
