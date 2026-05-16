from __future__ import annotations

import numpy as np

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic": [0, 3, 5, 7, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    # Extras
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_blues": [0, 3, 5, 6, 7, 10],
    "major_blues": [0, 2, 3, 4, 7, 9],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "harmonic_major": [0, 2, 4, 5, 7, 8, 11],
    "double_harmonic": [0, 1, 4, 5, 7, 8, 11],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "octatonic_whole_half": [0, 2, 3, 5, 6, 8, 9, 11],
    "octatonic_half_whole": [0, 1, 3, 4, 6, 7, 9, 10],
    "chromatic": list(range(12)),
}

CHARACTERISTIC_INTERVALS = {
    "major": [4, 11],
    "natural_minor": [3, 8],
    "dorian": [3, 9],
    "mixolydian": [4, 10],
    "pentatonic": [3, 7],
    "harmonic_minor": [3, 8, 11],
    "minor_pentatonic": [3, 7],
    "major_pentatonic": [4, 9],
    "minor_blues": [3, 6],
    "major_blues": [3, 4],
    "lydian": [4, 6],
    "phrygian": [1, 3],
    "aeolian": [3, 8],
    "locrian": [1, 6],
    "melodic_minor": [3, 9, 11],
    "harmonic_major": [4, 8, 11],
    "double_harmonic": [1, 4, 8, 11],
    "whole_tone": [4, 8],
    "octatonic_whole_half": [3, 6, 9],
    "octatonic_half_whole": [1, 4, 7],
    "chromatic": [],
}

NOTE_TO_MIDI = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63, "E": 64, "F": 65,
    "F#": 66, "Gb": 66, "G": 67, "G#": 68, "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
}


def midi_to_hz(m):
    m = np.asarray(m, dtype=float)
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def build_scale_pitches(root: str = "C", octave: int = 4, scale: str = "major", low_midi: int = 48, high_midi: int = 84):
    root_midi = NOTE_TO_MIDI[root] + 12 * (octave - 4)
    pattern = [p % 12 for p in SCALES[scale]]
    allowed = [m for m in range(low_midi, high_midi + 1) if ((m - root_midi) % 12) in pattern]
    return np.array(allowed, dtype=int)


def generate_random_melody(
    duration_s: float,
    bpm: int = 100,
    root: str = "C",
    octave: int = 4,
    scale: str = "major",
    low_midi: int = 55,
    high_midi: int = 79,
    step_bias: float = 0.8,
    leap_max_scale_steps: int = 4,
    rest_prob: float = 0.1,
    durations_beats=(0.25, 0.5, 1.0),
    duration_probs=(0.25, 0.5, 0.25),
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    allowed = build_scale_pitches(root, octave, scale, low_midi, high_midi)
    if len(allowed) < 2:
        raise ValueError("Pitch set too small; adjust range/scale.")
    spb = 60.0 / bpm
    t, t_end = 0.0, duration_s
    events = []
    
    root_midi_class = NOTE_TO_MIDI[root] % 12
    root_indices = [i for i, m in enumerate(allowed) if m % 12 == root_midi_class]
    
    if root_indices:
        current_idx = root_indices[0] if len(root_indices) == 1 else root_indices[len(root_indices) // 2 - 1]
    else:
        current_idx = rng.integers(0, len(allowed))

    # Build Intro Motif
    char_ivs = CHARACTERISTIC_INTERVALS.get(scale, [])
    scale_pattern = SCALES.get(scale, [])
    char_offsets = [scale_pattern.index(iv) for iv in char_ivs if iv in scale_pattern]
    
    if not char_offsets:
        char_offsets = [len(scale_pattern) // 3, len(scale_pattern) * 2 // 3]

    motif_offsets = [0]
    if len(char_offsets) >= 1:
        motif_offsets.append(char_offsets[0])
    if len(char_offsets) >= 2:
        motif_offsets.append(char_offsets[-1])
    else:
        motif_offsets.append(len(scale_pattern) // 2)
    
    motif_queue = []
    for offset in motif_offsets:
        target_idx = current_idx + offset
        if 0 <= target_idx < len(allowed):
            motif_queue.append(target_idx)

    while t < t_end - 1e-6:
        db = rng.choice(durations_beats, p=duration_probs)
        dur = min(db * spb, t_end - t)
        
        if rng.random() < rest_prob and not motif_queue:
            events.append((t, t + dur, None))
        else:
            if motif_queue:
                ni = motif_queue.pop(0)
            else:
                if rng.random() < step_bias:
                    step = rng.choice([-1, 1])
                    ni = np.clip(current_idx + step, 0, len(allowed) - 1)
                else:
                    min_leap = max(-leap_max_scale_steps, -current_idx)
                    max_leap = min(leap_max_scale_steps, len(allowed) - 1 - current_idx)
                    if min_leap < max_leap:
                        possible_targets = list(range(current_idx + min_leap, current_idx + max_leap + 1))
                        if current_idx in possible_targets:
                            possible_targets.remove(current_idx)
                        if possible_targets:
                            weights = []
                            for target in possible_targets:
                                iv = (allowed[target] - root_midi_class) % 12
                                if iv == 0:
                                    weights.append(3.0)
                                elif iv in char_ivs:
                                    weights.append(2.0)
                                else:
                                    weights.append(1.0)
                            weights = np.array(weights)
                            weights /= weights.sum()
                            ni = rng.choice(possible_targets, p=weights)
                        else:
                            ni = current_idx
                    else:
                        ni = current_idx
            
            current_idx = int(ni)
            events.append((t, t + dur, int(allowed[current_idx])))
        t += dur
    return events


def events_to_f0(
    events,
    sr: int,
    n_samples: int,
    glide_prob: float = 0.25,
    glide_frac: float = 0.35,
    vibrato_hz: float = 5.5,
    vibrato_depth: float = 0.02,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sr
    f0 = np.zeros(n_samples, dtype=float)
    for i, (t0, t1, midi) in enumerate(events):
        s0, s1 = int(round(t0 * sr)), int(round(t1 * sr))
        if s0 >= n_samples:
            break
        s1 = min(s1, n_samples)
        if midi is None:
            continue
        f_curr = midi_to_hz(midi)
        next_midi = events[i + 1][2] if (i + 1) < len(events) else None
        if next_midi is not None and rng.random() < glide_prob:
            f_next = midi_to_hz(next_midi)
            g_len = max(1, int((s1 - s0) * glide_frac))
            if s0 + g_len <= s1:
                f0[s0 : s0 + g_len] = np.linspace(f_curr, f_next, g_len, endpoint=False)
                f0[s0 + g_len : s1] = f_next
            else:
                f0[s0:s1] = np.linspace(f_curr, f_next, s1 - s0, endpoint=False)
        else:
            f0[s0:s1] = f_curr
    mask = f0 > 0
    if np.any(mask):
        f0[mask] *= 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t[mask])
    if n_samples > 2048:
        win = np.hanning(513)
        win /= win.sum()
        f0 = np.convolve(f0, win, mode="same")
    return f0

