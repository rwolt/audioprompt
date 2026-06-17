"""Regression tests for parse_midi_bass_events.

Build MIDIs in-memory with mido so tests have no external file dependencies.
Timing: ticks_per_beat=480, tempo=500000 (120 BPM) → 480 ticks = 0.5 s.
"""
import sys
from io import BytesIO
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audioprompt_core.midibass import parse_midi_bass_events


# ── helpers ──────────────────────────────────────────────────────────────────

TPB = 480       # ticks per beat
TEMPO = 500000  # µs per beat → 120 BPM; 480 ticks = 0.5 s


def _midi_bytes(messages: list, ticks_per_beat: int = TPB, tempo: int = TEMPO) -> bytes:
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    for msg in messages:
        track.append(msg)
    buf = BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _ticks_to_s(ticks: int) -> float:
    """Convert ticks to seconds at the test tempo."""
    return ticks * TEMPO / (TPB * 1_000_000)


# ── tests ────────────────────────────────────────────────────────────────────

def test_repeated_same_pitch_all_recovered():
    """Bug 1: same pitch retriggers before note-off — both notes must survive."""
    data = _midi_bytes([
        mido.Message("note_on",  note=60, velocity=80, time=0),
        mido.Message("note_on",  note=60, velocity=80, time=TPB),    # retrigger, no note-off
        mido.Message("note_off", note=60, velocity=0,  time=TPB),
    ])
    events, _, _ = parse_midi_bass_events(data)
    assert len(events) == 2, f"expected 2 events, got {len(events)}: {events}"
    for s, e, m, v in events:
        assert e > s, f"zero-length event: ({s}, {e})"
    print(f"  recovered {len(events)} notes: {[(round(s,3), round(e,3)) for s,e,*_ in events]}")


def test_phantom_leading_controller_stripped(trim=True):
    """Bug 2: controller with large delta before first note → first note starts ~0 s."""
    lead_ticks = 2 * TPB   # 1.0 s of phantom silence via controller delta
    data = _midi_bytes([
        mido.Message("control_change", control=7, value=100, time=lead_ticks),
        mido.Message("note_on",  note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0,  time=TPB),
    ])
    events, _, _ = parse_midi_bass_events(data, trim_leading_silence=True)
    assert events, "no events parsed"
    first_start = events[0][0]
    assert first_start < 0.01, f"first note should start ~0 s, got {first_start:.4f} s"
    print(f"  first note start after trim: {first_start:.4f} s")


def test_genuine_pickup_preserved():
    """Bug 2: note_on carrying its own delay (anacrusis) must NOT be stripped."""
    pickup_ticks = TPB   # 0.5 s pickup — encoded on the note itself, no prior events
    data = _midi_bytes([
        mido.Message("note_on",  note=60, velocity=80, time=pickup_ticks),
        mido.Message("note_off", note=60, velocity=0,  time=TPB),
    ])
    events, _, _ = parse_midi_bass_events(data, trim_leading_silence=True)
    assert events, "no events parsed"
    first_start = events[0][0]
    expected = _ticks_to_s(pickup_ticks)
    assert abs(first_start - expected) < 0.005, (
        f"pickup offset should be ~{expected:.3f} s, got {first_start:.4f} s"
    )
    print(f"  pickup preserved at {first_start:.4f} s (expected {expected:.3f} s)")


def test_trim_false_returns_raw_timing():
    """trim_leading_silence=False: raw absolute timing must be returned unchanged."""
    lead_ticks = 2 * TPB   # 1.0 s of phantom silence
    data = _midi_bytes([
        mido.Message("control_change", control=7, value=100, time=lead_ticks),
        mido.Message("note_on",  note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0,  time=TPB),
    ])
    events, _, _ = parse_midi_bass_events(data, trim_leading_silence=False)
    assert events, "no events parsed"
    first_start = events[0][0]
    expected_raw = _ticks_to_s(lead_ticks)
    assert abs(first_start - expected_raw) < 0.005, (
        f"raw timing should be ~{expected_raw:.3f} s, got {first_start:.4f} s"
    )
    print(f"  raw start preserved at {first_start:.4f} s (expected {expected_raw:.3f} s)")


def test_scenario_a_phantom_bar_beat2_entry_rest_survives():
    """Scenario A: phantom bar (controller delta) + beat-2 pickup (note delta).

    Phantom bar: 4 beats of controller silence stripped by trim.
    Beat-2 entry: note_on carries its own 1-beat delta → that rest survives.
    After trim, first note should start at ~0.5 s (1 beat), not 0.
    """
    phantom_ticks = 4 * TPB   # 2.0 s phantom bar via controller
    beat2_ticks   = 1 * TPB   # 0.5 s beat-2 rest encoded on the note itself
    data = _midi_bytes([
        mido.Message("control_change", control=7, value=100, time=phantom_ticks),
        mido.Message("note_on",  note=60, velocity=80, time=beat2_ticks),
        mido.Message("note_off", note=60, velocity=0,  time=2 * TPB),
    ])
    events, _, _ = parse_midi_bass_events(data, trim_leading_silence=True)
    assert events, "no events parsed"
    first_start = events[0][0]
    expected_beat2 = _ticks_to_s(beat2_ticks)   # 0.5 s — the beat-2 rest
    assert abs(first_start - expected_beat2) < 0.005, (
        f"beat-2 rest should survive as ~{expected_beat2:.3f} s, got {first_start:.4f} s"
    )
    print(f"  beat-2 rest survived: first note at {first_start:.4f} s "
          f"(phantom bar stripped, beat-2 offset {expected_beat2:.3f} s preserved)")


# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_repeated_same_pitch_all_recovered,
        test_phantom_leading_controller_stripped,
        test_genuine_pickup_preserved,
        test_trim_false_returns_raw_timing,
        test_scenario_a_phantom_bar_beat2_entry_rest_survives,
    ]
    for fn in tests:
        print(f"\n{fn.__name__}:")
        fn()
    print("\nAll tests passed.")
