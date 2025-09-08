"""Lightweight package init for audioprompt_core.

Avoid importing submodules at import time to prevent reloader edge cases.
Import from submodules explicitly (e.g., `from audioprompt_core.audio import ...`).
"""

__all__: list[str] = []

# Best-effort re-exports for backward compatibility (non-fatal on failure)
try:
    from .audio import load_audio_mono, apply_fades, wav_bytes, tag_suffix  # type: ignore
    __all__ += ["load_audio_mono", "apply_fades", "wav_bytes", "tag_suffix"]
except Exception:
    pass

try:
    from .prompt import (
        pink_noise,
        imprint_melody_focus,
        rhythmic_gate_from_events,
        apply_hpf,
        apply_mono_lows,
    )  # type: ignore
    __all__ += [
        "pink_noise",
        "imprint_melody_focus",
        "rhythmic_gate_from_events",
        "apply_hpf",
        "apply_mono_lows",
    ]
except Exception:
    pass

# Do not hard-import melody; environments without it should still import package
try:
    from .melody import SCALES, generate_random_melody, events_to_f0  # type: ignore
    __all__ += ["SCALES", "generate_random_melody", "events_to_f0"]
except Exception:
    # Leave names undefined if unavailable; callers should import from submodule
    pass
