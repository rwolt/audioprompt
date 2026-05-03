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
    # Streamlit's UploadedFile and other file-like objects: rewind before reading
    if hasattr(file_or_bytes, "read"):
        try:
            if hasattr(file_or_bytes, "seek"):
                file_or_bytes.seek(0)
        except Exception:
            pass
        return file_or_bytes
    raise TypeError("Unsupported input type for audio loading.")


def _is_mp3_name_or_type(obj: object) -> bool:
    name = getattr(obj, "name", "") or str(obj)
    mtype = getattr(obj, "type", "")
    return str(name).lower().endswith(".mp3") or "mpeg" in str(mtype).lower() or "mp3" in str(mtype).lower()


def _mp3_supported() -> bool:
    try:
        fmts = sf.available_formats()
        return "MP3" in fmts
    except Exception:
        return False


def load_audio_mono(file_or_bytes: Union[str, Path, bytes, bytearray, BytesIO, object], target_sr: int) -> Tuple[np.ndarray, int]:
    """Load audio, mix to mono float32, resample to target_sr.

    Supported formats depend on libsndfile (WAV/FLAC/OGG, sometimes AIFF). MP3/M4A are not
    guaranteed unless your environment's libsndfile supports them. For public apps, prefer
    WAV/FLAC uploads.
    """
    f = _as_readable(file_or_bytes)
    # Helpful message if the user uploads MP3 but libsndfile lacks MP3 support
    if _is_mp3_name_or_type(file_or_bytes) and not _mp3_supported():
        raise RuntimeError(
            "MP3 not supported by this libsndfile build. Install libsndfile with MP3 support "
            "(mpg123) or convert to WAV/FLAC/OGG."
        )
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


def wav_bytes_concat_segments(segments: list[np.ndarray], sr: int, peak_limit: float = 0.999) -> bytes:
    """Write multiple mono segments back-to-back into a single WAV without
    allocating a large concatenated array. Applies a single peak limiter
    across all segments by a constant scale factor.

    segments: list of 1D float arrays (any dtype) to be written sequentially.
    sr: sample rate for the output WAV.
    peak_limit: if overall peak exceeds this, scale all segments down so that
                max amplitude equals peak_limit.
    Returns WAV bytes (PCM16).
    """
    # Determine scale from overall peak across segments
    overall_peak = 0.0
    for s in segments:
        if s is None:
            continue
        try:
            p = float(np.max(np.abs(s)) + 1e-12)
        except Exception:
            p = 0.0
        if p > overall_peak:
            overall_peak = p
    scale = 1.0
    if overall_peak > peak_limit and overall_peak > 0.0:
        scale = float(peak_limit / overall_peak)

    bio = BytesIO()
    # Open a streaming writer to avoid building full array in memory
    with sf.SoundFile(bio, mode="w", samplerate=int(sr), channels=1, format="WAV", subtype="PCM_16") as f:
        for s in segments:
            if s is None:
                continue
            f.write((s.astype(np.float32, copy=False) * scale))
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
