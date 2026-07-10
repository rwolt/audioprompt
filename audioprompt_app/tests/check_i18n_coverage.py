"""Verify JA translation coverage of app.py's UI strings.

Extracts translatable literals from app.py's AST — widget labels, help=
strings, option lists, and t() templates — and asserts each has an entry in
the JA table. Because English literals are the lookup keys, a typo in
ui_strings.py or a reworded string in app.py silently falls back to English;
this check turns that into a visible failure.

Run: python tests/check_i18n_coverage.py
"""
import ast
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from ui_strings import JA  # noqa: E402

WIDGETS = {
    "slider", "checkbox", "selectbox", "radio", "number_input", "file_uploader",
    "button", "download_button", "expander", "subheader", "spinner", "caption",
}

# Intentionally untranslated (identical in Japanese, or musical notation).
IGNORE = {"---"}


def _constants(node) -> list[str]:
    """String constants in a node, descending into IfExp branches and lists."""
    out = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, ast.IfExp):
        out += _constants(node.body) + _constants(node.orelse)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            out += _constants(elt)
    return out


def extract(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # st.<widget>(label, ..., help=..., options=...)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
            and node.func.attr in WIDGETS
        ):
            if node.args:
                found.update(_constants(node.args[0]))
            for kw in node.keywords:
                if kw.arg in ("help", "options"):
                    found.update(_constants(kw.value))
        # t("template", ...) — includes markdown/warning/error/caption templates
        elif isinstance(node.func, ast.Name) and node.func.id == "t":
            if node.args:
                found.update(_constants(node.args[0]))
    return found


def main() -> int:
    strings = extract(APP_DIR / "app.py") - IGNORE
    missing = sorted(s for s in strings if s not in JA)
    print(f"extracted {len(strings)} translatable strings from app.py")
    if missing:
        print(f"\nMISSING from JA table ({len(missing)}):")
        for s in missing:
            print(f"  {s!r}")
        return 1
    print("JA coverage complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
