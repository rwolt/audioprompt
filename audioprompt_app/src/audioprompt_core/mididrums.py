from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import mido


# General MIDI drum note mapping (channel 9/10)
DRUM_LANE_MAP = {
    # Kick
    35: "kick",
    36: "kick",
    # Snare
    37: "snare",  # Side Stick
    38: "snare",  # Acoustic Snare
    39: "snare",  # Hand Clap
    40: "snare",  # Electric Snare
    # Low toms (map to snare-ish body for v1)
    41: "snare",
    43: "snare",
    45: "snare",
    47: "snare",
    48: "snare",
    50: "snare",
    # Hi-hats
    42: "hat",  # Closed Hi-Hat
    44: "hat",  # Pedal Hi-Hat
    46: "hat",  # Open Hi-Hat
    # Cymbals → hat-ish for v1 (high freq burst)
    49: "hat",  # Crash Cymbal 1
    51: "hat",  # Ride Cymbal 1
    52: "hat",  # Chinese Cymbal
    55: "hat",  # Splash Cymbal
    57: "hat",  # Crash Cymbal 2
    59: "hat",  # Ride Cymbal 2
    # Tambourine, cowbell → perc (misc)
    54: "perc",
    56: "perc",
    58: "perc",
    62: "perc",
    63: "perc",
    64: "perc",
    70: "perc",
    71: "perc",
    72: "perc",
    73: "perc",
    74: "perc",
    75: "perc",
    76: "perc",
    77: "perc",
    78: "perc",
    79: "perc",
    80: "perc",
    81: "perc",
    82: "perc",
    83: "perc",
    84: "perc",
    85: "perc",
}

# Lanes we actually generate noise for
LANES = ["kick", "snare", "hat", "perc"]


def _as_readable_midi(file_or_bytes: Union[str, Path, bytes, bytearray, BytesIO]) -> Union[str, BytesIO]:
    """Return something mido.MidiFile can read: path str or BytesIO."""
    if isinstance(file_or_bytes, (str, Path)):
        return str(file_or_bytes)
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return BytesIO(file_or_bytes)
    if isinstance(file_or_bytes, BytesIO):
        file_or_bytes.seek(0)
        return file_or_bytes
    raise TypeError(f"Unsupported MIDI input type: {type(file_or_bytes)}")


def _get_bpm(midi_file: mido.MidiFile) -> float:
    """Attempt to determine BPM. Defaults to 120 if no tempo message found."""
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return round(mido.tempo2bpm(msg.tempo), 2)
    return 120.0


def detect_midi_bpm(file_or_bytes: Union[str, Path, bytes, bytearray, BytesIO]) -> float:
    """Return the first tempo found in a MIDI file, defaulting to 120 BPM."""
    readable = _as_readable_midi(file_or_bytes)
    if isinstance(readable, BytesIO):
        midi_file = mido.MidiFile(file=readable)
    else:
        midi_file = mido.MidiFile(readable)
    return _get_bpm(midi_file)


def inspect_midi_timing(file_or_bytes: Union[str, Path, bytes, bytearray, BytesIO]) -> dict:
    """Return lightweight timing metadata for UI previews."""
    readable = _as_readable_midi(file_or_bytes)
    if isinstance(readable, BytesIO):
        midi_file = mido.MidiFile(file=readable)
    else:
        midi_file = mido.MidiFile(readable)
    bpm = _get_bpm(midi_file)
    tempo = mido.bpm2tempo(bpm)
    max_tick = 0
    for track in midi_file.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
        max_tick = max(max_tick, abs_tick)
    length_s = mido.tick2second(max_tick, midi_file.ticks_per_beat, tempo) if max_tick > 0 else 0.0
    return {
        "bpm": float(bpm),
        "length_s": float(length_s),
        "ticks": int(max_tick),
        "ticks_per_beat": int(midi_file.ticks_per_beat),
    }


def parse_midi_drum_events(
    file_or_bytes: Union[str, Path, bytes, bytearray, BytesIO],
) -> Tuple[Dict[str, List[Tuple[float, float, int]]], float]:
    """Parse a MIDI drum track into lane events.

    Returns:
        lanes: Dict mapping lane name -> list of (start_s, duration_s, velocity)
        bpm:   Detected or default BPM

    Note durations are determined by matching note_on with the next matching
    note_off for the same note number on the same channel. If no matching
    note_off is found, the note is treated as a short burst (default 0.01 s).
    """
    readable = _as_readable_midi(file_or_bytes)
    if isinstance(readable, BytesIO):
        midi_file = mido.MidiFile(file=readable)
    else:
        midi_file = mido.MidiFile(readable)
    bpm = _get_bpm(midi_file)

    # Accumulate absolute ticks and events per track
    # We process all tracks because some DAWs export drums across multiple tracks
    note_on_events = []  # (abs_tick, note, velocity, channel)
    note_off_events = []  # (abs_tick, note, channel)

    for track in midi_file.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_on_events.append((abs_tick, msg.note, msg.velocity, msg.channel))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                note_off_events.append((abs_tick, msg.note, msg.channel))

    # Sort by tick, then match note_on to next note_off for same (note, channel)
    note_on_events.sort(key=lambda x: (x[0], x[3], x[1]))
    note_off_events.sort(key=lambda x: (x[0], x[2], x[1]))

    # Match: for each note_on, find next note_off with same note+channel
    matched_notes = []  # (start_tick, end_tick, note, velocity)
    unresolved = list(note_on_events)
    off_pending = list(note_off_events)

    for start_tick, note, vel, ch in unresolved:
        found = False
        for idx, (end_tick, off_note, off_ch) in enumerate(off_pending):
            if off_note == note and off_ch == ch and end_tick >= start_tick:
                matched_notes.append((start_tick, end_tick, note, vel))
                off_pending.pop(idx)
                found = True
                break
        if not found:
            # No matching note_off found, default to short burst
            matched_notes.append((start_tick, start_tick + 1, note, vel))

    # Convert ticks to seconds
    def ticks_to_seconds(tick, tpb, tempo_us):
        return mido.tick2second(tick, tpb, tempo_us)

    # Re-scan for tempo changes
    # Build a tick->tempo map (handle tempo changes mid-track)
    tempo_events = []
    for track in midi_file.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((abs_tick, msg.tempo))
    if not tempo_events:
        tempo_events = [(0, mido.bpm2tempo(bpm))]

    # Sort tempo events by tick
    tempo_events.sort(key=lambda x: x[0])

    def resolve_tempo_at_tick(tick):
        """Find the active tempo at a given tick."""
        current = tempo_events[0][1]
        for t, tmp in tempo_events:
            if t <= tick:
                current = tmp
            else:
                break
        return current

    tpb = midi_file.ticks_per_beat

    lanes: Dict[str, List[Tuple[float, float, int]]] = {lane: [] for lane in LANES}

    for start_tick, end_tick, note, vel in matched_notes:
        lane = DRUM_LANE_MAP.get(note, "perc")  # unknown notes → perc lane
        tempo_us = resolve_tempo_at_tick(start_tick)
        start_s = ticks_to_seconds(start_tick, tpb, tempo_us)
        end_s = ticks_to_seconds(end_tick, tpb, tempo_us)
        duration_s = max(end_s - start_s, 0.01)  # minimum 10ms
        lanes[lane].append((start_s, duration_s, vel))

    # Sort each lane by time
    for lane in lanes:
        lanes[lane].sort(key=lambda x: x[0])

    return lanes, bpm


def scale_lane_events(
    lanes: Dict[str, List[Tuple[float, float, int]]],
    speed_mult: float,
) -> Dict[str, List[Tuple[float, float, int]]]:
    """Scale event start times and durations by a speed multiplier.

    speed_mult > 1.0  => faster (higher effective BPM, shorter times)
    speed_mult < 1.0  => slower (lower effective BPM, longer times)
    speed_mult = 1.0  => no change

    Returns a new dict with scaled events; original is unmodified.
    """
    if speed_mult == 1.0:
        return lanes
    result: Dict[str, List[Tuple[float, float, int]]] = {}
    for lane, events in lanes.items():
        result[lane] = [
            (start_s / speed_mult, duration_s / speed_mult, vel)
            for (start_s, duration_s, vel) in events
        ]
    return result


def loop_lane_events(
    lanes: Dict[str, List[Tuple[float, float, int]]],
    prompt_seconds: float,
) -> Dict[str, List[Tuple[float, float, int]]]:
    """Repeat parsed lane events until they fill the prompt length.

    The loop length is inferred from the last event end. This keeps simple
    exported drum regions useful even when the generated prompt is longer.
    """
    max_end = 0.0
    for events in lanes.values():
        for start_s, duration_s, _ in events:
            max_end = max(max_end, start_s + duration_s)
    if max_end <= 0.0 or prompt_seconds <= max_end:
        return lanes

    result: Dict[str, List[Tuple[float, float, int]]] = {lane: [] for lane in lanes}
    offset = 0.0
    while offset < prompt_seconds:
        for lane, events in lanes.items():
            for start_s, duration_s, vel in events:
                looped_start = start_s + offset
                if looped_start >= prompt_seconds:
                    continue
                result[lane].append((looped_start, duration_s, vel))
        offset += max_end

    for lane in result:
        result[lane].sort(key=lambda x: x[0])
    return result


def summarize_drum_lanes(lanes: Dict[str, List[Tuple[float, float, int]]]) -> dict:
    """Return a human-readable summary of lane contents."""
    summary = {}
    for lane, events in lanes.items():
        if not events:
            summary[lane] = "no events"
            continue
        avg_vel = int(np.mean([e[2] for e in events]))
        summary[lane] = f"{len(events)} events, avg vel {avg_vel}, span {events[-1][0]:.2f}s"
    return summary
