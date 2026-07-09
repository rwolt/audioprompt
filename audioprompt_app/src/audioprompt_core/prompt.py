from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft, firwin, filtfilt

from audioprompt_core.formants import vowel_traj_from_plan, apply_formants


def pink_noise(n: int, sr: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    X[1:] /= np.sqrt(np.maximum(f[1:], 1e-6))
    y = np.fft.irfft(X, n)
    y /= np.max(np.abs(y) + 1e-12)
    return y.astype(np.float32)


def _soft_band_envelope(f_hz, low_hz, high_hz, sharpness=12.0):
    f = np.asarray(f_hz, dtype=float)
    f_safe = np.maximum(f, 1e-6)
    lf = np.log2(f_safe)
    l0 = np.log2(max(low_hz, 1e-6))
    h0 = np.log2(max(high_hz, low_hz + 1e-6))
    lo = 1.0 / (1.0 + np.exp(-sharpness * (lf - l0)))
    hi = 1.0 / (1.0 + np.exp(sharpness * (lf - h0)))
    return lo * hi


def _timbre_weights(k: int, character: str) -> float:
    """Return a harmonic weight for a small set of musician-facing characters."""
    if character == "bright":
        return 0.85 + 0.12 * k
    if character == "warm":
        return np.exp(-0.18 * (k - 1))
    if character == "voice":
        return 1.0 if k <= 5 else 0.35
    if character == "reed":
        return 1.0 / k if k % 2 == 1 else 0.08 / k
    if character == "bell":
        return np.exp(-((k - 3) ** 2) / 4.0)
    if character == "pluck":
        return np.exp(-0.38 * (k - 1))
    return 1.0


def _mask_shape_kernel(freqs, fk, bw, shape: str) -> np.ndarray:
    x = (freqs - fk) / (bw + 1e-6)
    if shape == "soft":
        return 1.0 / (1.0 + x**2)
    if shape == "tight":
        return np.maximum(0.0, 1.0 - np.abs(x) / np.sqrt(2))
    return np.exp(-0.5 * x**2)


def imprint_melody_focus(
    noise: np.ndarray,
    sr: int,
    f0_hz,
    *,
    gain: float = 8.0,
    harmonics: int = 10,
    bw_frac: float = 0.01,
    focus=None,
    band_floor_db: float = -18.0,
    sharpness: float = 12.0,
    n_fft: int = 2048,
    character: str = "neutral",
    note_shape: str = "natural",
    detune_spread_cents: float = 0.0,
    vowel_plan=None,
    formant_strength: float = 1.0,
    melody_noise_floor_db: float = 0.0,
) -> np.ndarray:
    hop = n_fft // 4
    # Use Hann window with default boundary handling to satisfy NOLA/overlap-add conditions
    freqs, times, Z = stft(noise, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    mag, ph = np.abs(Z), np.angle(Z)
    del Z  # free the complex STFT matrix; only mag/ph are used from here on

    if focus is not None:
        if isinstance(focus, str):
            presets = {
                "bass": (40, 300),
                "guitar": (80, 6000),
                "vocal": (120, 3200),
            }
            if focus not in presets:
                raise ValueError(f"Unknown focus preset: {focus}")
            low_hz, high_hz = presets[focus]
        else:
            low_hz, high_hz = focus
        band = _soft_band_envelope(freqs, low_hz, high_hz, sharpness=sharpness)
        floor = 10.0 ** (band_floor_db / 20.0)
        eq_mask = floor + (1.0 - floor) * band
        mag *= eq_mask[:, None]

    if np.isscalar(f0_hz):
        f0_traj = np.full_like(times, float(f0_hz))
    else:
        f0_time = np.linspace(0, len(noise) / sr, num=len(f0_hz), endpoint=False)
        f0_traj = np.interp(times, f0_time, f0_hz)

    for i, f0 in enumerate(f0_traj):
        if f0 <= 0 or not np.isfinite(f0):
            continue
        mask = np.zeros_like(freqs)
        for k in range(1, harmonics + 1):
            fk = k * f0
            if fk > freqs[-1]:
                break
            bw = bw_frac * fk
            weight = _timbre_weights(k, character)
            shape = "tight" if note_shape == "tight" else "soft" if note_shape == "smooth" else "gaussian"
            mask += weight * _mask_shape_kernel(freqs, fk, bw, shape)
            if detune_spread_cents > 0:
                spread = 2.0 ** (detune_spread_cents / 1200.0)
                mask += 0.35 * weight * _mask_shape_kernel(freqs, fk / spread, bw, shape)
                mask += 0.35 * weight * _mask_shape_kernel(freqs, fk * spread, bw, shape)
        if mask.max() > 0:
            noise_floor_linear = 10.0 ** (melody_noise_floor_db / 20.0)
            mask = noise_floor_linear + (gain * (mask / mask.max()))
            mag[:, i] *= mask

    if vowel_plan is not None:
        vtraj = vowel_traj_from_plan(vowel_plan, times)
        apply_formants(mag, freqs, times, vtraj, strength=float(formant_strength))

    # Inverse STFT with matching window and hop
    spec = mag * np.exp(1j * ph)
    del mag, ph
    _, y = istft(spec, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    del spec
    y = y[: len(noise)]
    y /= np.max(np.abs(y) + 1e-12)
    return y.astype(np.float32)


def rhythmic_gate_from_events(events, sr: int, n_samples: int, shape: str = "natural", decay_mult: float = 1.0):
    env = np.zeros(n_samples, dtype=float)
    for event in events:
        if len(event) == 3:
            t0, t1, midi = event
            vel = 1.0
        else:
            t0, t1, midi, vel = event
            
        if midi is None:
            continue
        s0 = int(np.round(t0 * sr))
        s1 = int(np.round(t1 * sr))
        s0 = max(0, min(n_samples - 1, s0))
        s1 = max(0, min(n_samples, s1))
        if s1 <= s0:
            continue
            
        dur_s = (s1 - s0) / sr
        t_rel = np.linspace(0, dur_s, s1 - s0, endpoint=False)
        
        if shape == "pluck":
            tau = 0.08 * decay_mult
            seg = np.exp(-t_rel / max(tau, 0.001))
            a_samples = int(0.002 * sr)
            if a_samples > 0 and a_samples < len(seg):
                seg[:a_samples] *= np.linspace(0, 1, a_samples)
        elif shape == "tight":
            tau = 0.15 * decay_mult
            seg = np.exp(-t_rel / max(tau, 0.001))
            a_samples = int(0.005 * sr)
            if a_samples > 0 and a_samples < len(seg):
                seg[:a_samples] *= np.linspace(0, 1, a_samples)
        elif shape == "smooth":
            seg = np.ones_like(t_rel)
            a_samples = min(int(0.08 * sr), len(seg)//2)
            r_samples = min(int(0.08 * sr), len(seg)//2)
            if a_samples > 0:
                seg[:a_samples] *= np.linspace(0, 1, a_samples)
            if r_samples > 0:
                seg[-r_samples:] *= np.linspace(1, 0, r_samples)
        else: # natural
            tau = 0.5 * decay_mult
            seg = np.exp(-t_rel / max(tau, 0.001))
            seg = np.maximum(seg, 0.5 * np.ones_like(seg)) # Sustain level
            a_samples = min(int(0.01 * sr), len(seg)//2)
            r_samples = min(int(0.03 * sr), len(seg)//2)
            if a_samples > 0:
                seg[:a_samples] *= np.linspace(0, 1, a_samples)
            if r_samples > 0:
                seg[-r_samples:] *= np.linspace(1, 0, r_samples)

        env[s0:s1] = np.maximum(env[s0:s1], seg * float(vel))
    return env


# ---------------------- Low-end utilities (HPF / Mono lows) ---------------------- #

def _design_hpf(sr: int, cutoff_hz: float, taps: int = 1025) -> np.ndarray:
    """Design a linear-phase FIR high-pass filter using a Kaiser window.

    taps should be odd for exact linear phase; use filtfilt for zero-phase application.
    """
    cutoff = max(5.0, min(cutoff_hz, sr / 2.5))  # basic sanity
    taps = int(taps) if taps % 2 == 1 else int(taps + 1)
    # Use Kaiser beta tuned for ~60 dB stopband
    beta = 8.6
    return firwin(taps, cutoff, fs=sr, window=("kaiser", beta), pass_zero=False)


def apply_hpf(y: np.ndarray, sr: int, cutoff_hz: float = 25.0, taps: int = 1025) -> np.ndarray:
    """Apply a zero-phase FIR high-pass to y.

    Works for mono (1D) or multi-channel (2D with shape [n, ch]).
    """
    if cutoff_hz <= 0:
        return y
    b = _design_hpf(sr, cutoff_hz, taps)
    if y.ndim == 1:
        return filtfilt(b, [1.0], y).astype(np.float32)
    # apply per-channel
    out = np.zeros_like(y, dtype=np.float32)
    for c in range(y.shape[1]):
        out[:, c] = filtfilt(b, [1.0], y[:, c]).astype(np.float32)
    return out


def apply_mono_lows(y: np.ndarray, sr: int, cutoff_hz: float = 120.0, taps: int = 1025) -> np.ndarray:
    """Sum low frequencies below cutoff to mono while keeping highs stereo.

    If y is mono (1D), returns y unchanged.
    """
    if y.ndim == 1:
        return y
    cutoff = max(20.0, min(cutoff_hz, sr / 3.0))
    taps = int(taps) if taps % 2 == 1 else int(taps + 1)
    beta = 8.6
    # Low-pass for lows
    lp = firwin(taps, cutoff, fs=sr, window=("kaiser", beta), pass_zero=True)
    left_lp = filtfilt(lp, [1.0], y[:, 0])
    right_lp = filtfilt(lp, [1.0], y[:, 1]) if y.shape[1] > 1 else left_lp
    mono_low = 0.5 * (left_lp + right_lp)
    # High components = original - lowpass
    left_hi = y[:, 0] - left_lp
    right_hi = (y[:, 1] - right_lp) if y.shape[1] > 1 else left_hi
    out_left = (left_hi + mono_low).astype(np.float32)
    out_right = (right_hi + mono_low).astype(np.float32)
    return np.stack([out_left, out_right], axis=1)
