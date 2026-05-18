from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import numpy as np
import mido

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


def parse_midi_bass_events(midi_bytes: bytes) -> Tuple[List[Tuple[float, float, int, float]], List[Tuple[float, float]], float]:
    """
    Parses a MIDI file to extract bass note events and pitch bend.
    Returns:
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
    
    events = []
    pitch_bends = []
    
    # We'll assume the longest track with notes is the bass track
    best_track_events = []
    best_track_bends = []
    
    pitch_bend_range_semitones = 2.0 # Standard GM pitch bend range
    
    for track in midi_file.tracks:
        current_time_s = 0.0
        active_notes = {} # note -> (start_time, velocity)
        track_events = []
        track_bends = []
        
        for msg in track:
            dt_s = mido.tick2second(msg.time, ticks_per_beat, tempo)
            current_time_s += dt_s
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_time_s, msg.velocity / 127.0)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_t, vel = active_notes.pop(msg.note)
                    track_events.append((start_t, current_time_s, msg.note, vel))
            elif msg.type == 'pitchwheel':
                # msg.pitch is -8192 to 8191
                bend_st = (msg.pitch / 8192.0) * pitch_bend_range_semitones
                track_bends.append((current_time_s, bend_st))
                
        if len(track_events) > len(best_track_events):
            best_track_events = track_events
            best_track_bends = track_bends
            
    # Sort events by start time
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
    
    # Render base pitch
    for s_time, e_time, midi_note, _ in events:
        s0 = int(np.round(s_time * sr))
        s1 = int(np.round(e_time * sr))
        s0 = max(0, min(n_samples - 1, s0))
        s1 = max(0, min(n_samples, s1))
        if s1 > s0:
            f0[s0:s1] = midi_note
            
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
