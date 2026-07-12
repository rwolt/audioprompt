"""Functional i18n/accessibility tests using Streamlit's AppTest harness.

Runs app.py headlessly and checks: English default rendering, full Japanese
rendering, help-as-text mode producing visible captions, and the combination.

Run: python tests/test_i18n_app.py
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run(lang: str | None = None, help_text: bool = False) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    if lang:
        at.session_state["_ui_lang"] = lang
    if help_text:
        at.session_state["ui_help_as_text"] = True
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
    return at


def test_english_default():
    at = _run()
    subs = [s.value for s in at.subheader]
    assert "Quick Start" in subs and "Outputs" in subs, subs
    print(f"  EN: {len(subs)} subheaders, {len(at.caption)} captions")


def test_japanese_renders_everywhere():
    at = _run(lang="ja")
    subs = [s.value for s in at.subheader]
    for expected in ("クイックスタート", "メロディ", "フォーカス", "出力"):
        assert expected in subs, f"{expected} missing from {subs}"
    assert "プロンプトを生成" in [b.label for b in at.button]
    print(f"  JA: subheaders {subs}")


def test_help_as_text_mode():
    baseline = len(_run().caption)
    at = _run(help_text=True)
    n = len(at.caption)
    assert n > baseline + 10, f"expected help captions: {baseline} -> {n}"
    texts = [c.value for c in at.caption]
    assert any("randomized melody" in c for c in texts), "melody help caption missing"
    print(f"  a11y: captions {baseline} -> {n}")


def test_both_midi_sections_in_help_text_mode():
    """Regression: identically-labeled trim checkboxes in the drum and bass
    sections collided (StreamlitDuplicateElementId) in help-as-text mode,
    where stripping help= made their auto-generated widget IDs equal."""
    at = _run(help_text=True)
    cbs = {c.label: c for c in at.checkbox}
    cbs["Enable drum MIDI imprint"].check()
    cbs["Enable bass MIDI imprint"].check()
    at.run()
    assert not at.exception, f"duplicate-ID regression: {at.exception}"
    trims = [c for c in at.checkbox if c.label == "Trim DAW export padding"]
    assert len(trims) == 2, "expected both trim checkboxes rendered"
    print("  both MIDI sections + a11y: no widget ID collision")


def test_japanese_help_as_text():
    at = _run(lang="ja", help_text=True)
    texts = [c.value for c in at.caption]
    assert any("ランダムメロディ" in c for c in texts), "JA melody help caption missing"
    print("  JA + a11y: Japanese help captions render")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n{name}:")
            fn()
    print("\nAll tests passed.")
