from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import numpy as np
import mido

from .midiregion import region_start_ticks


def detect_pitch_bend_range(midi_bytes: bytes) -> float | None:
    """Read the pitch-bend range a MIDI file declares via RPN 0 (if any).

    DAWs that export expressive bass parts (e.g. Logic's Bass Player, which
    exports as MPE) declare the instrument's bend range in-file with the RPN
    sequence CC101=0, CC100=0 (select RPN 0 "pitch bend sensitivity"), then
    CC6 = semitones and optionally CC38 = cents. Returns the declared range
    in semitones, or None when the file doesn't declare one.

    Only data entry while RPN 0 is selected counts — MPE exports also carry
    an RPN 6 zone-configuration message whose CC6 value is a channel count,
    not a bend range.
    """
    midi_file = mido.MidiFile(file=BytesIO(midi_bytes))
    for track in midi_file.tracks:
        rpn: dict[int, tuple[int, int]] = {}  # channel -> (msb, lsb)
        semis: dict[int, int] = {}
        cents: dict[int, int] = {}
        for msg in track:
            if msg.type != "control_change":
                continue
            ch = msg.channel
            msb, lsb = rpn.get(ch, (127, 127))  # 127,127 = RPN null
            if msg.control == 101:
                rpn[ch] = (msg.value, lsb)
            elif msg.control == 100:
                rpn[ch] = (msb, msg.value)
            elif msg.control == 6 and rpn.get(ch) == (0, 0):
                semis[ch] = msg.value
            elif msg.control == 38 and rpn.get(ch) == (0, 0):
                cents[ch] = msg.value
        if semis:
            ch = min(semis)
            return float(semis[ch]) + cents.get(ch, 0) / 100.0
    return None

def inspect_bass_midi_timing(midi_bytes: bytes) -> dict:
    midi_file = mido.MidiFile(file=BytesIO(midi_bytes))
    tempo = 500000
    length_ticks = 0
    ticks_per_beat = midi_file.ticks_per_beat
    
    for track in midi_file.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == 'set_tempo':
                tempo = msg.tempo
        if t > length_ticks:
            length_ticks = t
            
    bpm = mido.tempo2bpm(tempo)
    length_s = mido.tick2second(length_ticks, ticks_per_beat, tempo)
    return {"bpm": bpm, "length_s": length_s}


def parse_midi_bass_events(midi_bytes: bytes, pitch_bend_range: float = 12.0, trim_leading_silence: bool = True) -> Tuple[List[Tuple[float, float, int, float]], List[Tuple[float, float]], float]:
    """
    Parses a MIDI file to extract bass note events and pitch bend.

    Parameters
    ----------
    trim_leading_silence:
        When True (default), remove the export's region-position offset:
        Logic exports MIDI with tick 0 at project bar 1, so a region that
        sat at bar 2 arrives with one bar of padding. The region's true
        start is marked in-file by the smpte_offset/set_tempo meta stamp
        (see midiregion.region_start_ticks) — everything before the stamp
        is padding and is removed; silence after it is musical content
        (e.g. a bass entering at bar 5) and is preserved. When the stamp
        sits at 0, fall back to subtracting time consumed by non-note
        events (controllers/meta) preceding the first note_on, which
        catches exports whose padding rides on a controller's delta while
        still preserving pickups written on the note's own delta. Set
        False to return raw absolute timing unchanged.

    Returns
    -------
    events: List of (start_s, end_s, midi_note, velocity_norm)
    pitch_bends: List of (time_s, bend_semitones)
    bpm: detected BPM
    """
    midi_file = mido.MidiFile(file=BytesIO(midi_bytes))
    tempo = 500000
    ticks_per_beat = midi_file.ticks_per_beat

    # First pass: find tempo
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    bpm = mido.tempo2bpm(tempo)

    # Region-position offset (Logic: tick 0 = project bar 1, meta stamp at
    # region start). None when the file carries no stamp at all.
    region_ticks = region_start_ticks(midi_file)
    region_s = (
        mido.tick2second(region_ticks, ticks_per_beat, tempo)
        if region_ticks else 0.0
    )

    # We'll assume the longest track with notes is the bass track
    best_track_events: List[Tuple[float, float, int, float]] = []
    best_track_bends: List[Tuple[float, float]] = []

    pitch_bend_range_semitones = float(pitch_bend_range)

    for track in midi_file.tracks:
        current_time_s = 0.0
        active_notes: dict[int, tuple[float, float]] = {}  # note -> (start_time, velocity)
        track_events: List[Tuple[float, float, int, float]] = []
        track_bends: List[Tuple[float, float]] = []
        lead_time: float | None = None  # time consumed by non-note events before first note_on

        for msg in track:
            dt_s = mido.tick2second(msg.time, ticks_per_beat, tempo)
            current_time_s += dt_s

            if msg.type == 'note_on' and msg.velocity > 0:
                # Bug 2: capture time of non-note preamble at moment of first note_on,
                # BEFORE this note's own delta, so intentional pickups are preserved.
                if lead_time is None:
                    lead_time = current_time_s - dt_s

                # Bug 1: same pitch retriggered while still held — close the held note
                # first so it gets its real duration instead of being overwritten.
                if msg.note in active_notes:
                    start_t, vel = active_notes.pop(msg.note)
                    if current_time_s > start_t:
                        track_events.append((start_t, current_time_s, msg.note, vel))

                active_notes[msg.note] = (current_time_s, msg.velocity / 127.0)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_t, vel = active_notes.pop(msg.note)
                    if current_time_s > start_t:
                        track_events.append((start_t, current_time_s, msg.note, vel))

            elif msg.type == 'pitchwheel':
                bend_st = (msg.pitch / 8192.0) * pitch_bend_range_semitones
                track_bends.append((current_time_s, bend_st))

        # Bug 1: close any notes still held at end of track (no note_off received).
        for note, (start_t, vel) in active_notes.items():
            if current_time_s > start_t:
                track_events.append((start_t, current_time_s, note, vel))

        # Strip export padding (only when requested).
        if lead_time is None:
            lead_time = 0.0
        if trim_leading_silence:
            if region_s > 0.0:
                # Region stamp marks the true start: everything before it is
                # project-position padding. Clamp because humanization can
                # place the first note a few ticks ahead of the barline.
                track_events = [
                    (max(0.0, s - region_s), max(max(0.0, s - region_s) + 0.01, e - region_s), m, v)
                    for s, e, m, v in track_events
                ]
                track_bends = [(max(0.0, t - region_s), b) for t, b in track_bends]
            elif lead_time > 0.0:
                # No region offset — fall back to stripping non-note lead
                # time (padding carried on a controller/meta delta). A
                # lead_time of 0 (clean MIDI or genuine pickup) leaves
                # everything unchanged.
                track_events = [(s - lead_time, e - lead_time, m, v)
                                for s, e, m, v in track_events]
                track_bends = [(max(0.0, t - lead_time), b) for t, b in track_bends]

        if len(track_events) > len(best_track_events):
            best_track_events = track_events
            best_track_bends = track_bends

    best_track_events.sort(key=lambda x: x[0])
    return best_track_events, best_track_bends, bpm

def scale_bass_events(
    events: List[Tuple[float, float, int, float]],
    bends: List[Tuple[float, float]],
    speed_mult: float
):
    scaled_events = [(s / speed_mult, e / speed_mult, m, v) for s, e, m, v in events]
    scaled_bends = [(t / speed_mult, b) for t, b in bends]
    return scaled_events, scaled_bends

def loop_bass_events(
    events: List[Tuple[float, float, int, float]],
    bends: List[Tuple[float, float]],
    prompt_seconds: float
):
    if not events:
        return events, bends
    
    max_time = max([e for _, e, _, _ in events])
    if max_time <= 0:
        return events, bends
        
    loop_length = max_time
    looped_events = []
    looped_bends = []
    
    n_loops = int(np.ceil(prompt_seconds / loop_length))
    for i in range(n_loops):
        offset = i * loop_length
        for s, e, m, v in events:
            if s + offset < prompt_seconds:
                looped_events.append((s + offset, min(e + offset, prompt_seconds), m, v))
        for t, b in bends:
            if t + offset < prompt_seconds:
                looped_bends.append((t + offset, b))
                
    return looped_events, looped_bends

def bass_events_to_f0(
    events: List[Tuple[float, float, int, float]],
    bends: List[Tuple[float, float]],
    sr: int,
    n_samples: int
) -> np.ndarray:
    t = np.arange(n_samples) / sr
    f0 = np.zeros(n_samples, dtype=float)
    
    # Sort events by start time just in case
    events = sorted(events, key=lambda x: x[0])
    
    # Render base pitch with legato portamento
    for i, (s_time, e_time, midi_note, _) in enumerate(events):
        s0 = int(np.round(s_time * sr))
        s1 = int(np.round(e_time * sr))
        s0 = max(0, min(n_samples - 1, s0))
        s1 = max(0, min(n_samples, s1))
        
        if s1 <= s0:
            continue
            
        f0[s0:s1] = midi_note
        
        # Check for overlaps (legato) to create a slide
        if i > 0:
            _, prev_e_time, prev_midi, _ = events[i-1]
            # If previous note overlaps into this one, or ends exactly as this starts
            if s_time < prev_e_time + 0.01:
                glide_s = min(0.08, (e_time - s_time) * 0.5) # 80ms glide, or max half the note
                g_samples = int(np.round(glide_s * sr))
                if g_samples > 0 and s0 + g_samples < n_samples:
                    # Overwrite the start of this note with a slide from the previous note
                    f0[s0:s0+g_samples] = np.linspace(prev_midi, midi_note, g_samples)
            
    # Apply pitch bend
    if bends and np.any(f0 > 0):
        bend_times = [0.0] + [b[0] for b in bends] + [n_samples / sr]
        bend_vals = [0.0] + [b[1] for b in bends] + [bends[-1][1] if bends else 0.0]
        
        # Interpolate pitch bend curve
        bend_curve = np.interp(t, bend_times, bend_vals)
        
        # Add bend to active notes
        mask = f0 > 0
        f0[mask] = f0[mask] + bend_curve[mask]
        
    # Convert midi to hz
    mask = f0 > 0
    f0[mask] = 440.0 * (2.0 ** ((f0[mask] - 69.0) / 12.0))
    
    # Smooth to avoid zipper noise on bends
    if n_samples > 1024:
        win = np.hanning(129)
        win /= win.sum()
        f0 = np.convolve(f0, win, mode="same")
        
    return f0
