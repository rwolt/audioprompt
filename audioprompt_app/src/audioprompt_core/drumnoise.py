from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import firwin, filtfilt


# --------------------------------------------------------------------------- #
# Per-lane defaults tuned for pink-noise drum feel
# --------------------------------------------------------------------------- #

LANE_DEFAULTS = {
    "kick": {
        "low_hz": 30.0,
        "high_hz": 200.0,
        "attack_s": 0.003,
        "decay_s": 0.35,
        "velocity_decay_factor": 0.5,  # how much velocity lengthens decay
        "taps": 1025,
    },
    "snare": {
        "low_hz": 150.0,
        "high_hz": 3500.0,
        "attack_s": 0.001,
        "decay_s": 0.18,
        "velocity_decay_factor": 0.4,
        "taps": 1025,
    },
    "hat": {
        "low_hz": 5000.0,
        "high_hz": 16000.0,
        "attack_s": 0.0005,
        "decay_s": 0.04,
        "velocity_decay_factor": 0.2,
        "taps": 1025,
    },
    "perc": {
        "low_hz": 200.0,
        "high_hz": 6000.0,
        "attack_s": 0.002,
        "decay_s": 0.12,
        "velocity_decay_factor": 0.3,
        "taps": 1025,
    },
}


def _band_limited_pink_noise(
    n: int, sr: int, seed: int | None, low_hz: float, high_hz: float, taps: int = 1025
) -> np.ndarray:
    """Generate pink noise and band-pass it between low_hz and high_hz."""
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    X[1:] /= np.sqrt(np.maximum(f[1:], 1e-6))
    y = np.fft.irfft(X, n)
    y /= np.max(np.abs(y) + 1e-12)

    # Bandpass via FIR (linear phase with filtfilt for zero-phase)
    nyq = sr / 2.0
    lo = max(10.0, min(low_hz, nyq - 1))
    hi = max(lo + 1, min(high_hz, nyq))
    taps = int(taps) if taps % 2 == 1 else int(taps + 1)
    # Use transition bandwidth relative to center freq
    tbw = min(abs(hi - lo) * 0.15, hi * 0.3, nyq * 0.4)
    b = firwin(
        taps,
        [lo, hi],
        fs=sr,
        pass_zero=False,
        window=("kaiser", 8.6),
        width=tbw,
    )
    y = filtfilt(b, [1.0], y).astype(np.float32)
    return y


def _hit_envelope(hit_len: int, attack_n: int, decay_n: int) -> np.ndarray:
    """Build an AD style envelope (attack then exponential decay).

    Parameters
    ----------
    hit_len : int
        Total length in samples for the hit.
    attack_n : int
        Number of samples for the attack rise.
    decay_n : int
        Decay half-life in samples (lower = faster fade).
    """
    env = np.zeros(hit_len, dtype=np.float32)
    atk = min(attack_n, hit_len)
    if atk > 0:
        env[:atk] = np.linspace(0.0, 1.0, atk, endpoint=False, dtype=np.float32)
    rem = hit_len - atk
    if rem > 0:
        # Exponential decay: exp(-t / decay_n)
        t = np.arange(rem, dtype=np.float32)
        env[atk:] = np.exp(-t / max(decay_n, 1.0))
    return env


def build_lane_envelope(
    events: List[Tuple[float, float, int]],
    sr: int,
    n_total_samples: int,
    attack_s: float,
    decay_s: float,
    velocity_decay_factor: float = 0.0,
    velocity_gain_factor: float = 1.0,
) -> np.ndarray:
    """Convert lane events (start_s, duration_s, velocity) into an envelope.

    Parameters
    ----------
    events : list of (start_s, duration_s, velocity)
    sr : int
    n_total_samples : int
        Length of the output envelope in samples.
    attack_s : float
        Attack time in seconds.
    decay_s : float
        Base decay time (half-life-ish) in seconds.
    velocity_decay_factor : float
        Additional decay scaling per velocity [0..1]. Higher velocity -> longer decay.
    velocity_gain_factor : float
        How much velocity affects amplitude. 1.0 = full velocity scaling.

    Returns
    -------
    envelope : np.ndarray, shape (n_total_samples,)
    """
    env = np.zeros(n_total_samples, dtype=np.float32)
    for start_s, _, vel in events:
        s0 = int(np.round(start_s * sr))
        if s0 >= n_total_samples:
            continue

        vel_norm = vel / 127.0

        # Scale decay by velocity (soft hits decay faster)
        this_decay_s = decay_s * (1.0 + velocity_decay_factor * vel_norm)
        decay_n = int(this_decay_s * sr)

        # Hit length: allow up to 3 * decay_s or until next note, whichever shorter
        hit_len = min(int(3.0 * this_decay_s * sr), n_total_samples - s0)
        if hit_len <= 0:
            continue

        attack_n = int(attack_s * sr)
        hit_env = _hit_envelope(hit_len, attack_n, decay_n)

        # Velocity scaling for amplitude
        amp = vel_norm ** velocity_gain_factor if velocity_gain_factor != 1.0 else vel_norm
        hit_env *= amp

        # Write into global envelope at s0 (max of overlapping events)
        seg_end = s0 + hit_len
        env[s0:seg_end] = np.maximum(env[s0:seg_end], hit_env)
    return env


def synthesize_drum_layer(
    lanes: Dict[str, List[Tuple[float, float, int]]],
    sr: int,
    prompt_seconds: float,
    seed: int | None = None,
    lane_gains: Dict[str, float] | None = None,
    lane_params: Dict[str, dict] | None = None,
    master_gain: float = 1.0,
) -> np.ndarray:
    """Synthesize a drum prompt from parsed MIDI lane events.

    Each lane gets its own band-limited pink noise, envelope-shaped by hits.
    Lanes are summed and normalized.

    Parameters
    ----------
    lanes : dict
        From `parse_midi_drum_events`: lane -> list of (start_s, duration_s, velocity)
    sr : int
        Sample rate.
    prompt_seconds : float
        Duration of the output.
    seed : int or None
        Seed for pink noise RNG.
    lane_gains : dict or None
        Override gain per lane (0..1+). Defaults to 1.0 for all.
    lane_params : dict or None
        Override defaults per lane (attack_s, decay_s, low_hz, high_hz, etc.).
    master_gain : float
        Final gain on the summed output.

    Returns
    -------
    y : np.ndarray shape (n_samples,)
        Normalized drum layer.
    """
    n = int(sr * prompt_seconds)
    lane_gains = lane_gains or {}
    lane_params = lane_params or {}
    combined = np.zeros(n, dtype=np.float32)

    # Unique seed per lane derived from base so we get different noise textures
    seed = seed if seed is not None else np.random.default_rng().integers(0, 1_000_000)
    rng = np.random.default_rng(seed)
    lane_seeds = {lane: int(rng.integers(0, 1_000_000)) for lane in lanes}

    for lane_name, events in lanes.items():
        if not events:
            continue
        params = {**LANE_DEFAULTS.get(lane_name, LANE_DEFAULTS["perc"]), **lane_params.get(lane_name, {})}
        gain = lane_gains.get(lane_name, 1.0)

        noise = _band_limited_pink_noise(
            n,
            sr,
            seed=lane_seeds[lane_name],
            low_hz=params["low_hz"],
            high_hz=params["high_hz"],
            taps=params["taps"],
        )

        env = build_lane_envelope(
            events,
            sr=sr,
            n_total_samples=n,
            attack_s=params["attack_s"],
            decay_s=params["decay_s"],
            velocity_decay_factor=params["velocity_decay_factor"],
        )

        combined += noise * env * gain

    if master_gain != 1.0:
        combined *= master_gain

    peak = float(np.max(np.abs(combined)) + 1e-12)
    if peak > 1.0:
        combined = combined / peak
    return combined.astype(np.float32)
