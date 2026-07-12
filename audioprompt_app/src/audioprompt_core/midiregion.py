"""Shared helper for locating a DAW export's region-position offset.

Logic Pro (and possibly other DAWs) writes MIDI exports with tick 0 at bar 1
of the *project*, not at the start of the exported region. If the region sat
at bar 2, the file begins with one bar of silence that was never part of the
music — the infamous "phantom bar". Logic marks the region's true start by
placing its ``smpte_offset``/``set_tempo`` meta stamp at the region position
(tick 0 when the region is at bar 1, tick 1920 for bar 2 at 480 tpb, …).

That stamp lets us distinguish export padding (before the stamp — always
junk) from intentional leading silence inside the region (after the stamp —
musical content, e.g. a bass entering at bar 5 of an 8-bar loop).
"""
from __future__ import annotations

import mido


def region_start_ticks(midi_file: mido.MidiFile) -> int | None:
    """Best-effort region-position offset of a MIDI export, in ticks.

    Returns the absolute tick of the region-start meta stamp: the first
    ``smpte_offset`` if present, else the first ``set_tempo``. Returns None
    when the file has neither (no way to tell padding from content).

    The stamp only counts as a region marker when it sits at or before the
    first note (within one beat of tolerance — session-player humanization
    can place the first note a few ticks *before* the barline the stamp sits
    on). A later ``set_tempo`` is a mid-song tempo change, not a marker.
    """
    smpte_tick: int | None = None
    tempo_tick: int | None = None
    first_note_tick: int | None = None
    for track in midi_file.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "smpte_offset" and smpte_tick is None:
                smpte_tick = abs_tick
            elif msg.type == "set_tempo" and tempo_tick is None:
                tempo_tick = abs_tick
            elif msg.type == "note_on" and msg.velocity > 0:
                # Humanization can place the first note a few ticks before
                # the stamp, so the stamp may appear *after* it in message
                # order — keep scanning the track past the first note.
                if first_note_tick is None or abs_tick < first_note_tick:
                    first_note_tick = abs_tick
    stamp = smpte_tick if smpte_tick is not None else tempo_tick
    if stamp is None:
        return None
    if first_note_tick is not None and stamp > first_note_tick + midi_file.ticks_per_beat:
        # Stamp is well past the first note: a mid-song tempo change,
        # not a region marker. Treat as "region starts at 0".
        return 0
    return stamp
