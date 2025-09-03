from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft


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
) -> np.ndarray:
    hop = n_fft // 4
    freqs, times, Z = stft(noise, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    mag, ph = np.abs(Z), np.angle(Z)

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
            mask += np.exp(-0.5 * ((freqs - fk) / (bw + 1e-6)) ** 2)
        if mask.max() > 0:
            mask = 1.0 + (gain * (mask / mask.max()))
            mag[:, i] *= mask

    _, y = istft(mag * np.exp(1j * ph), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    y = y[: len(noise)]
    y /= np.max(np.abs(y) + 1e-12)
    return y.astype(np.float32)


def rhythmic_gate_from_events(events, sr: int, n_samples: int, attack: float = 0.01, release: float = 0.03):
    env = np.zeros(n_samples, dtype=float)
    for (t0, t1, midi) in events:
        s0 = int(np.round(t0 * sr))
        s1 = int(np.round(t1 * sr))
        s0 = max(0, min(n_samples - 1, s0))
        s1 = max(0, min(n_samples, s1))
        if s1 <= s0:
            continue
        a = max(1, int(attack * sr))
        r = max(1, int(release * sr))
        seg = np.ones(s1 - s0, dtype=float)
        seg[: min(a, len(seg))] *= np.linspace(0, 1, num=min(a, len(seg)), endpoint=False)
        if r < len(seg):
            seg[-r:] *= np.linspace(1, 0, num=r, endpoint=True)
        env[s0:s1] = np.maximum(env[s0:s1], seg)
    return env

