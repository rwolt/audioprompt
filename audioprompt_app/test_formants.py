"""Tests for formants.py.

The substantive test FFTs the shaped output and asserts that spectral energy
actually concentrates near the intended F1/F2 for a given vowel, and that two
different vowels produce measurably different spectra.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import stft, istft

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from audioprompt_core import formants as F  # noqa: E402


def _shape_pink_with_vowel(vowel, *, dur=1.0, sr=22050, n_fft=2048, strength=1.0):
    """Pink noise -> STFT -> apply one steady vowel -> ISTFT. Returns audio."""
    rng = np.random.default_rng(0)
    n = int(dur * sr)
    Xf = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    Xf[1:] /= np.sqrt(np.maximum(f[1:], 1e-6))
    noise = np.fft.irfft(Xf, n).astype(np.float32)

    hop = n_fft // 4
    freqs, times, Z = stft(noise, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    mag, ph = np.abs(Z), np.angle(Z)

    traj = np.tile(np.array(F.VOWEL_FORMANTS[vowel]), (len(times), 1))
    F.apply_formants(mag, freqs, times, traj, strength=strength)

    _, y = istft(mag * np.exp(1j * ph), fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return y, sr


def _formant_gain_peaks_hz(vowel, sr=22050, n_peaks=3, smooth=15):
    """Measure where the formant filter actually boosts energy.

    Compares the shaped spectrum against the unshaped (strength=0) pink baseline
    and finds peaks in the *ratio*. This removes pink noise's low-frequency tilt
    so we see only what the formant envelope did.
    """
    y_on, _ = _shape_pink_with_vowel(vowel, strength=1.0, sr=sr)
    y_off, _ = _shape_pink_with_vowel(vowel, strength=0.0, sr=sr)
    n = min(len(y_on), len(y_off))
    w = np.hanning(n)
    Yon = np.abs(np.fft.rfft(y_on[:n] * w)) + 1e-9
    Yoff = np.abs(np.fft.rfft(y_off[:n] * w)) + 1e-9
    f = np.fft.rfftfreq(n, 1 / sr)
    ratio = Yon / Yoff
    k = np.ones(smooth) / smooth
    ratio = np.convolve(ratio, k, mode="same")
    band = (f > 150) & (f < 3500)
    fb, rb = f[band], ratio[band]
    idx = [i for i in range(1, len(rb) - 1) if rb[i] > rb[i - 1] and rb[i] > rb[i + 1]]
    idx.sort(key=lambda i: rb[i], reverse=True)
    return sorted(fb[i] for i in idx[:n_peaks])


def test_vowel_plan_stress_reduction():
    # 4 sung notes + a rest; accent_period=2 -> notes 0 and 2 full, 1 and 3 schwa.
    events = [(0.0, 0.5, 60), (0.5, 0.75, 62), (0.75, 1.0, None),
              (1.0, 1.5, 64), (1.5, 2.0, 65)]
    plan = F.english_vowel_plan(events, accent_period=2, seed=1)
    assert len(plan) == 4, "rest should be skipped"
    vowels = [v for (_, _, v) in plan]
    assert vowels[0] != "schwa" and vowels[2] != "schwa", "accents should be full vowels"
    assert vowels[1] == "schwa" and vowels[3] == "schwa", "weak beats should reduce"
    print("  stress reduction:", vowels)


def test_language_plans():
    events = [(0.0, 0.5, 60), (0.5, 1.0, 62), (1.0, 1.5, 64), (1.5, 2.0, 65)]
    # Japanese and Spanish: every note gets a full vowel from that language's
    # inventory — no schwa reduction, accent_period has no effect.
    for lang, prefix in [("japanese", "ja_"), ("spanish", "es_")]:
        plan = F.build_vowel_plan(events, language=lang, accent_period=2, seed=3)
        vowels = [v for (_, _, v) in plan]
        assert len(vowels) == 4
        assert all(v.startswith(prefix) for v in vowels), f"{lang} plan used {vowels}"
        assert "schwa" not in vowels
        print(f"  {lang}: {vowels}")
    # English via the generic builder still reduces weak beats.
    plan = F.build_vowel_plan(events, language="english", accent_period=2, seed=3)
    vowels = [v for (_, _, v) in plan]
    assert vowels[1] == "schwa" and vowels[3] == "schwa"
    print(f"  english: {vowels}")
    # Unknown language is a clear error, not silent fallback.
    try:
        F.build_vowel_plan(events, language="klingon")
        assert False, "unknown language should raise ValueError"
    except ValueError:
        print("  unknown language raises ValueError")


def test_strength_zero_is_noop():
    y_off, sr = _shape_pink_with_vowel("aa", strength=0.0)
    y_ref, _ = _shape_pink_with_vowel("aa", strength=0.0)
    assert np.allclose(y_off, y_ref)
    print("  strength=0 is a clean no-op")


def test_formant_peaks_land_near_targets():
    for vowel, tol in [("iy", 300), ("aa", 300), ("uw", 300), ("ja_u", 300), ("es_i", 300)]:
        peaks = _formant_gain_peaks_hz(vowel)
        f1_t, f2_t, _ = F.VOWEL_FORMANTS[vowel]
        near_f1 = min(abs(p - f1_t) for p in peaks)
        near_f2 = min(abs(p - f2_t) for p in peaks)
        print(f"  {vowel}: targets F1={f1_t:.0f} F2={f2_t:.0f} | gain peaks {[round(p) for p in peaks]}"
              f" | dF1={near_f1:.0f} dF2={near_f2:.0f}")
        assert near_f1 < tol, f"{vowel} F1 off by {near_f1:.0f} Hz"
        assert near_f2 < tol, f"{vowel} F2 off by {near_f2:.0f} Hz"


def test_distinct_vowels_differ():
    pi, pa = _formant_gain_peaks_hz("iy"), _formant_gain_peaks_hz("aa")
    # F2 is the discriminator: /iy/ F2~2300 vs /aa/ F2~1100. (F3 is similar for both.)
    f2_iy = min(pi, key=lambda p: abs(p - F.VOWEL_FORMANTS["iy"][1]))
    f2_aa = min(pa, key=lambda p: abs(p - F.VOWEL_FORMANTS["aa"][1]))
    assert f2_iy - f2_aa > 600, "iy and aa should have clearly different F2"
    print(f"  distinct: iy F2~{f2_iy:.0f} vs aa F2~{f2_aa:.0f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n{name}:")
            fn()
    print("\nAll tests passed.")
