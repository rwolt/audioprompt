"""Formant shaping for vocal-leaning audio prompts.

The melody imprint in ``prompt.py`` emphasizes a clean harmonic stack at f0.
That gives a model a clear *pitch* to latch onto, but a flat harmonic series
has no vowel identity (it is closest to a buzzy schwa). This module multiplies
in a slow-moving *formant envelope* -- two to three resonant peaks that drift
between vowel targets -- so a hallucinated voice has actual vowels to grab.

Language support
----------------
Each language preset pairs a vowel inventory with a rhythm rule:

- ``english`` -- stress-timed: accented notes get a full vowel, weak beats
  *reduce toward schwa*. That long-strong-then-reduced alternation is the
  acoustic signature that separates English from languages where every vowel
  stays full, so it biases language identity toward English.
- ``japanese`` -- mora-timed: the five vowels a/i/u/e/o on EVERY note, no
  reduction (Japanese has no schwa collapse). The /u/ target is the unrounded
  Japanese [ɯ] (F2 ~1300 Hz), noticeably brighter than English "who".
- ``spanish`` -- syllable-timed: the five Spanish vowels on every note, no
  reduction.

The formant machinery below (trajectories, glides, resonance envelopes) is
language-agnostic; only the vowel tables and the plan builder know about
languages.

Design notes
------------
- Everything here is OPTIONAL and additive. The pipeline only calls it when a
  ``vowel_plan`` is supplied; otherwise behaviour is unchanged.
- Operates on STFT magnitude in the same ``[freq, time]`` orientation that
  ``prompt.imprint_melody_focus`` already uses, so it composes by multiplying
  into ``mag`` per frame -- same pattern as the existing focus band.
- Pure NumPy. No new dependencies.

Formant values are adult-average ballparks (Peterson-Barney lineage for
English; published adult averages for Japanese and Spanish); they are meant to
steer a model, not to pass a phonetics exam.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# (F1, F2, F3) in Hz. English entries are corners of the English vowel space
# plus schwa; ja_*/es_* entries are the Japanese and Spanish five-vowel systems.
VOWEL_FORMANTS: dict[str, tuple[float, float, float]] = {
    "iy": (270.0, 2300.0, 3000.0),   # heed
    "ih": (400.0, 2000.0, 2550.0),   # hid
    "eh": (530.0, 1850.0, 2500.0),   # head
    "ae": (660.0, 1700.0, 2400.0),   # had
    "aa": (730.0, 1100.0, 2440.0),   # father
    "ao": (570.0, 840.0, 2410.0),    # hawed
    "uh": (640.0, 1190.0, 2390.0),   # hud
    "uw": (300.0, 870.0, 2240.0),    # who
    "schwa": (500.0, 1500.0, 2500.0),  # unstressed reduced vowel
    # Japanese: a i u e o. The u is unrounded [ɯ] -- its high F2 (~1300 Hz vs
    # ~870 Hz for English uw) is the signature difference from English.
    "ja_a": (750.0, 1250.0, 2600.0),
    "ja_i": (290.0, 2250.0, 3000.0),
    "ja_u": (320.0, 1300.0, 2350.0),
    "ja_e": (470.0, 1950.0, 2600.0),
    "ja_o": (470.0, 850.0, 2500.0),
    # Spanish: a e i o u. Rounded [u], clean five-vowel system.
    "es_a": (700.0, 1300.0, 2500.0),
    "es_e": (460.0, 1950.0, 2600.0),
    "es_i": (300.0, 2300.0, 3000.0),
    "es_o": (460.0, 880.0, 2500.0),
    "es_u": (320.0, 750.0, 2300.0),
}

# Per-language plan config: which full vowels to rotate through, and whether
# weak (unaccented) beats reduce to a target vowel. ``reduce_weak_to=None``
# means every sung note gets a full vowel and the accent period is ignored --
# correct for mora-timed (Japanese) and syllable-timed (Spanish) languages,
# which have no English-style vowel reduction.
LANGUAGES: dict[str, dict] = {
    "english": {
        "full_vowels": ("aa", "ae", "eh", "iy", "ao", "uw", "ih"),
        "reduce_weak_to": "schwa",
    },
    "japanese": {
        "full_vowels": ("ja_a", "ja_i", "ja_u", "ja_e", "ja_o"),
        "reduce_weak_to": None,
    },
    "spanish": {
        "full_vowels": ("es_a", "es_e", "es_i", "es_o", "es_u"),
        "reduce_weak_to": None,
    },
}

# A compact set of "full" vowels to rotate through on accented beats.
_FULL_VOWELS: tuple[str, ...] = LANGUAGES["english"]["full_vowels"]


def build_vowel_plan(
    events: Sequence[tuple[float, float, object]],
    *,
    language: str = "english",
    accent_period: int = 2,
    accent_offset: int = 0,
    seed: int | None = None,
) -> list[tuple[float, float, str]]:
    """Assign vowels to gate events using a language's vowel system.

    For stress-timed languages (``reduce_weak_to`` set), every
    ``accent_period``-th *sung* note (rests are skipped) gets a full vowel and
    the rest collapse to the reduction target. For languages without vowel
    reduction (Japanese, Spanish), every sung note gets a full vowel and
    ``accent_period`` has no effect.

    Returns a list of ``(t0, t1, vowel_key)`` for sung notes only.
    """
    if language not in LANGUAGES:
        raise ValueError(f"Unknown language: {language!r} (have {sorted(LANGUAGES)})")
    cfg = LANGUAGES[language]
    full_vowels = cfg["full_vowels"]
    reduce_to = cfg["reduce_weak_to"]

    rng = np.random.default_rng(seed)
    plan: list[tuple[float, float, str]] = []
    sung_idx = 0
    for (t0, t1, midi) in events:
        if midi is None:
            continue
        is_accent = ((sung_idx + accent_offset) % max(1, accent_period)) == 0
        if reduce_to is None or is_accent:
            vowel = full_vowels[rng.integers(0, len(full_vowels))]
        else:
            vowel = reduce_to
        plan.append((float(t0), float(t1), vowel))
        sung_idx += 1
    return plan


def english_vowel_plan(
    events: Sequence[tuple[float, float, object]],
    *,
    accent_period: int = 2,
    accent_offset: int = 0,
    seed: int | None = None,
) -> list[tuple[float, float, str]]:
    """Back-compat wrapper for ``build_vowel_plan(language="english")``."""
    return build_vowel_plan(
        events,
        language="english",
        accent_period=accent_period,
        accent_offset=accent_offset,
        seed=seed,
    )


def vowel_traj_from_plan(
    vowel_plan: Sequence[tuple[float, float, str]],
    times: np.ndarray,
    *,
    glide_s: float = 0.045,
) -> np.ndarray:
    """Resolve a vowel plan into a per-STFT-frame ``(F1, F2, F3)`` trajectory.

    Formant *transitions* (glides between targets) are a large part of what
    reads as speech, so targets are linearly interpolated across ``glide_s``
    rather than switched abruptly. Frames that fall in gaps between notes hold
    the previous vowel's formants (the band just goes quiet there anyway).

    Returns an array of shape ``(len(times), 3)``.
    """
    times = np.asarray(times, dtype=float)
    if len(vowel_plan) == 0:
        # Neutral schwa everywhere -> effectively a gentle, vowel-ambiguous tilt.
        f = np.array(VOWEL_FORMANTS["schwa"], dtype=float)
        return np.tile(f, (len(times), 1))

    # Build target formant value at each note center, then interpolate in time.
    centers = np.array([(t0 + t1) * 0.5 for (t0, t1, _) in vowel_plan])
    targets = np.array([VOWEL_FORMANTS[v] for (_, _, v) in vowel_plan], dtype=float)
    order = np.argsort(centers)
    centers, targets = centers[order], targets[order]

    traj = np.empty((len(times), 3), dtype=float)
    for j in range(3):
        traj[:, j] = np.interp(times, centers, targets[:, j])

    # Soften abrupt interpolation kinks into glide-like ramps.
    if glide_s > 0 and len(times) > 1:
        dt = float(np.median(np.diff(times))) if len(times) > 1 else glide_s
        win = max(1, int(round(glide_s / max(dt, 1e-6))))
        if win > 1:
            kernel = np.ones(win) / win
            for j in range(3):
                traj[:, j] = np.convolve(traj[:, j], kernel, mode="same")
    return traj


def _resonance(freqs: np.ndarray, fc: float, bw: float) -> np.ndarray:
    """A single resonant peak (Lorentzian) centered at ``fc`` with bandwidth ``bw``."""
    x = (freqs - fc) / (bw + 1e-6)
    return 1.0 / (1.0 + x * x)


def formant_envelope(
    freqs: np.ndarray,
    f123: Sequence[float],
    *,
    bws: Sequence[float] = (90.0, 110.0, 160.0),
    gains: Sequence[float] = (1.0, 0.75, 0.45),
    floor_db: float = -14.0,
) -> np.ndarray:
    """Build a per-frequency multiplicative gain for one frame's vowel.

    The envelope is a sum of resonant peaks at F1/F2/F3, normalized to a peak of
    1.0, then lifted by a floor so non-formant regions are attenuated but not
    erased (erasing them entirely makes the prompt sound gated and unnatural).
    """
    freqs = np.asarray(freqs, dtype=float)
    env = np.zeros_like(freqs)
    for fc, bw, g in zip(f123, bws, gains):
        if fc <= 0:
            continue
        env += g * _resonance(freqs, fc, bw)
    peak = env.max()
    if peak > 0:
        env /= peak
    floor = 10.0 ** (floor_db / 20.0)
    return floor + (1.0 - floor) * env


def apply_formants(
    mag: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    vowel_traj: np.ndarray,
    *,
    strength: float = 1.0,
    bws: Sequence[float] = (90.0, 110.0, 160.0),
    gains: Sequence[float] = (1.0, 0.75, 0.45),
    floor_db: float = -14.0,
) -> np.ndarray:
    """Multiply a per-frame formant envelope into an STFT magnitude.

    Parameters
    ----------
    mag:
        STFT magnitude, shape ``(n_freqs, n_frames)`` (same as ``prompt.py``).
    freqs, times:
        Frequency and frame-time axes from ``scipy.signal.stft``.
    vowel_traj:
        ``(n_frames, 3)`` array of ``(F1, F2, F3)`` per frame.
    strength:
        0.0 = no effect, 1.0 = full envelope. Lets the UI dial it in gently.

    Returns the modified ``mag`` (also mutated in place).
    """
    if mag.shape[1] != len(times):
        raise ValueError(f"mag has {mag.shape[1]} frames but times has {len(times)}")
    if vowel_traj.shape[0] != len(times):
        raise ValueError("vowel_traj must have one row per frame")

    s = float(np.clip(strength, 0.0, 1.0))
    for i in range(len(times)):
        env = formant_envelope(
            freqs, vowel_traj[i], bws=bws, gains=gains, floor_db=floor_db
        )
        # Blend toward unity by strength so the effect is dial-able.
        env = 1.0 + s * (env - 1.0)
        mag[:, i] *= env
    return mag
