"""Regression tests for region-stamp-aware leading-silence trim + RPN bend range.

Logic exports MIDI with tick 0 at project bar 1 and marks the region's true
start with its smpte_offset/set_tempo meta stamp. Trim must remove only the
padding before the stamp (region position), never intentional silence inside
the region. Patterns replicate real Logic exports in audioprompt_recipes.

Build MIDIs in-memory with mido so tests have no external file dependencies.
Timing: ticks_per_beat=480, default tempo 500000 (120 BPM) → 480 ticks = 0.5 s.
"""
import sys
from io import BytesIO
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audioprompt_core.midibass import parse_midi_bass_events, detect_pitch_bend_range
from audioprompt_core.mididrums import parse_midi_drum_events


# ── helpers ──────────────────────────────────────────────────────────────────

TPB = 480
BAR = 4 * TPB          # one 4/4 bar
SEC_PER_TICK = 500000 / (TPB * 1_000_000)  # at the 120 BPM default


def _midi_bytes(messages: list) -> bytes:
    mid = mido.MidiFile(ticks_per_beat=TPB)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for msg in messages:
        track.append(msg)
    buf = BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _smpte(time: int = 0) -> mido.MetaMessage:
    return mido.MetaMessage("smpte_offset", frame_rate=25, hours=1, minutes=0,
                            seconds=0, frames=0, sub_frames=0, time=time)


def _first_drum_start(lanes) -> float:
    return min(s for events in lanes.values() for (s, _, _) in events)


# ── bass ─────────────────────────────────────────────────────────────────────

def test_bass_phantom_bar_marked_by_stamp_is_trimmed():
    """Region at bar 2 (stamp at 1 bar), padding on the note's own delta.

    Replicates dnb_174_root_A_bass.mid: no controllers, the whole bar rides
    on the first note_on delta — the old controller-lead rule missed this.
    """
    data = _midi_bytes([
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.Message("note_on",  note=45, velocity=80, time=BAR - 9),  # humanized early
        _smpte(time=9),                                                # stamp at bar 2
        mido.Message("note_off", note=45, velocity=0,  time=TPB),
    ])
    events, _, _ = parse_midi_bass_events(data, trim_leading_silence=True)
    assert events and events[0][0] < 0.01, f"phantom bar should be trimmed: {events}"
    print(f"  first note at {events[0][0]:.4f} s after stamp trim")


def test_bass_intentional_lead_after_stamp_preserved():
    """Region at bar 1 (stamp at 0), bass enters at bar 5 — 4 bars are content.

    Replicates 4bar-leadin-test.mid: trim must NOT touch in-region silence.
    """
    data = _midi_bytes([
        _smpte(time=0),
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.Message("note_on",  note=45, velocity=80, time=4 * BAR),
        mido.Message("note_off", note=45, velocity=0,  time=TPB),
    ])
    events, _, _ = parse_midi_bass_events(data, trim_leading_silence=True)
    expected = 4 * BAR * SEC_PER_TICK
    assert abs(events[0][0] - expected) < 0.005, (
        f"4-bar lead should be preserved at {expected:.3f} s, got {events[0][0]:.4f} s"
    )
    print(f"  intentional 4-bar lead preserved at {events[0][0]:.3f} s")


def test_bass_controller_lead_fallback_still_trims():
    """Stamp at 0 but padding carried on a controller delta (lofi CC103 case)."""
    data = _midi_bytes([
        _smpte(time=0),
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.Message("control_change", control=103, value=53, time=BAR - 26),
        mido.Message("note_on",  note=45, velocity=80, time=26),
        mido.Message("note_off", note=45, velocity=0,  time=TPB),
    ])
    events, _, _ = parse_midi_bass_events(data, trim_leading_silence=True)
    expected = 26 * SEC_PER_TICK  # only the note's own humanization delta survives
    assert abs(events[0][0] - expected) < 0.005, (
        f"controller lead should be trimmed to {expected:.3f} s, got {events[0][0]:.4f} s"
    )
    print(f"  controller lead trimmed, note delta kept: {events[0][0]:.4f} s")


def test_bass_midsong_tempo_change_is_not_a_stamp():
    """A tempo change after the notes begin must not be mistaken for a region
    marker (no tempo/smpte at tick 0, first set_tempo mid-song)."""
    data = _midi_bytes([
        mido.Message("note_on",  note=45, velocity=80, time=TPB),  # 1-beat pickup
        mido.Message("note_off", note=45, velocity=0,  time=TPB),
        mido.MetaMessage("set_tempo", tempo=400000, time=8 * BAR),
        mido.Message("note_on",  note=47, velocity=80, time=0),
        mido.Message("note_off", note=47, velocity=0,  time=TPB),
    ])
    events, _, _ = parse_midi_bass_events(data, trim_leading_silence=True)
    # Tempo resolution uses the file's (mid-song) tempo for everything, so
    # compare in that clock: pickup must survive as one beat.
    expected = TPB * 400000 / (TPB * 1_000_000)
    assert abs(events[0][0] - expected) < 0.005, (
        f"pickup should survive at ~{expected:.3f} s, got {events[0][0]:.4f} s"
    )
    print(f"  mid-song tempo change ignored, pickup at {events[0][0]:.3f} s")


# ── drums ────────────────────────────────────────────────────────────────────

def test_drums_phantom_bar_marked_by_stamp_is_trimmed():
    """Replicates drum-and-bass-160-drums.mid: stamp at bar 2, first hit 9
    ticks early (humanization) — shift lands it at 0."""
    data = _midi_bytes([
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.Message("note_on",  note=36, velocity=100, time=BAR - 9),
        _smpte(time=9),
        mido.Message("note_off", note=36, velocity=0,   time=TPB),
    ])
    lanes, _ = parse_midi_drum_events(data, trim_leading_silence=True)
    first = _first_drum_start(lanes)
    assert first < 0.01, f"phantom bar should be trimmed, first hit at {first:.4f} s"
    print(f"  first hit at {first:.4f} s after stamp trim")


def test_drums_intentional_lead_after_stamp_preserved():
    """Stamp at 0 → drums resting until bar 5 stay at bar 5 (the 8-bar-loop,
    drums-enter-late use case)."""
    data = _midi_bytes([
        _smpte(time=0),
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.Message("note_on",  note=36, velocity=100, time=4 * BAR),
        mido.Message("note_off", note=36, velocity=0,   time=TPB),
    ])
    lanes, _ = parse_midi_drum_events(data, trim_leading_silence=True)
    first = _first_drum_start(lanes)
    expected = 4 * BAR * SEC_PER_TICK
    assert abs(first - expected) < 0.005, (
        f"4-bar rest should be preserved at {expected:.3f} s, got {first:.4f} s"
    )
    print(f"  intentional 4-bar rest preserved at {first:.3f} s")


def test_drums_no_stamp_falls_back_to_shift_to_zero():
    """Files with no tempo/smpte stamp at all: keep the legacy behavior of
    shifting the earliest hit to zero (can't tell padding from content)."""
    data = _midi_bytes([
        mido.Message("note_on",  note=36, velocity=100, time=BAR),
        mido.Message("note_off", note=36, velocity=0,   time=TPB),
    ])
    lanes, _ = parse_midi_drum_events(data, trim_leading_silence=True)
    first = _first_drum_start(lanes)
    assert first < 0.01, f"legacy shift-to-zero expected, first hit at {first:.4f} s"
    print(f"  no-stamp fallback shifted first hit to {first:.4f} s")


# ── RPN pitch bend range ─────────────────────────────────────────────────────

def test_rpn_bend_range_detected_and_mpe_config_ignored():
    """RPN 0 declares 48 semitones on the note channel; the MPE zone-config
    RPN 6 on the master channel (CC6=15 = channel count) must not be misread.
    Replicates the Logic Bass Player MPE export."""
    data = _midi_bytes([
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        # channel 0: MPE configuration message (RPN 0,6 → 15 member channels)
        mido.Message("control_change", channel=0, control=101, value=0,  time=0),
        mido.Message("control_change", channel=0, control=100, value=6,  time=0),
        mido.Message("control_change", channel=0, control=6,   value=15, time=0),
        # channel 1: pitch bend sensitivity (RPN 0,0 → 48 semitones)
        mido.Message("control_change", channel=1, control=101, value=0,  time=0),
        mido.Message("control_change", channel=1, control=100, value=0,  time=0),
        mido.Message("control_change", channel=1, control=6,   value=48, time=0),
        mido.Message("control_change", channel=1, control=38,  value=0,  time=0),
        mido.Message("note_on",  channel=1, note=45, velocity=80, time=0),
        mido.Message("note_off", channel=1, note=45, velocity=0,  time=TPB),
    ])
    detected = detect_pitch_bend_range(data)
    assert detected == 48.0, f"expected 48.0 semitones, got {detected}"
    print(f"  detected bend range: {detected} semitones (MPE config ignored)")


def test_rpn_absent_returns_none():
    data = _midi_bytes([
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.Message("control_change", control=7, value=100, time=0),
        mido.Message("note_on",  note=45, velocity=80, time=0),
        mido.Message("note_off", note=45, velocity=0,  time=TPB),
    ])
    detected = detect_pitch_bend_range(data)
    assert detected is None, f"expected None, got {detected}"
    print("  no RPN declared → None (slider default applies)")


# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_bass_phantom_bar_marked_by_stamp_is_trimmed,
        test_bass_intentional_lead_after_stamp_preserved,
        test_bass_controller_lead_fallback_still_trims,
        test_bass_midsong_tempo_change_is_not_a_stamp,
        test_drums_phantom_bar_marked_by_stamp_is_trimmed,
        test_drums_intentional_lead_after_stamp_preserved,
        test_drums_no_stamp_falls_back_to_shift_to_zero,
        test_rpn_bend_range_detected_and_mpe_config_ignored,
        test_rpn_absent_returns_none,
    ]
    for fn in tests:
        print(f"\n{fn.__name__}:")
        fn()
    print("\nAll tests passed.")
