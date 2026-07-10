"""Lightweight i18n + accessibility layer for the AudioPrompt Streamlit app.

English UI literals are the translation keys (gettext style): ``t(s)`` looks
the string up in the active language table and falls back to the original
English when missing, so untranslated strings degrade gracefully.

Importing this module wraps the common ``st.*`` widget functions once per
process so that, without touching every call site in app.py:

- widget labels and ``help=`` tooltips are translated via ``t()``;
- selectbox/radio option values get a translating ``format_func`` for display
  while the returned values stay untouched (code that compares option values
  keeps working);
- in "help as visible text" mode, each widget's help renders as an
  ``st.caption`` under the control instead of a hover tooltip. Streamlit
  tooltips are hover-only (not keyboard-focusable, no aria-describedby), so
  screen readers and browser translation can't reach them; this mode is the
  accessible/translatable alternative.

Language resolution order: explicit ``?lang=`` URL param > browser locale from
``st.context.locale`` > English. The user's selector choice is written back to
the URL so it survives reruns and makes shareable links.
"""
from __future__ import annotations

import streamlit as st

from ui_strings import BLOCKS, LANG_NAMES, TABLES

_LANG_KEY = "_ui_lang"
_HELP_TEXT_KEY = "ui_help_as_text"  # doubles as the checkbox widget key


def current_lang() -> str:
    return st.session_state.get(_LANG_KEY, "en")


def help_as_text() -> bool:
    return bool(st.session_state.get(_HELP_TEXT_KEY, False))


def t(text, **fmt):
    """Translate an English UI literal; format placeholders if given."""
    if not isinstance(text, str):
        return text
    table = TABLES.get(current_lang())
    out = table.get(text, text) if table else text
    return out.format(**fmt) if fmt else out


def block(name: str) -> str:
    """Fetch a multi-line content block (quick start, footer) for the language."""
    entry = BLOCKS[name]
    return entry.get(current_lang()) or entry["en"]


def init() -> None:
    """Resolve language and help mode once per session."""
    if _LANG_KEY not in st.session_state:
        lang = st.query_params.get("lang")
        if lang != "en" and lang not in TABLES:
            lang = None
        if lang is None:
            try:
                locale = str(st.context.locale or "")
            except Exception:
                locale = ""
            lang = next(
                (code for code in TABLES if locale.lower().startswith(code)), "en"
            )
        st.session_state[_LANG_KEY] = lang
    if _HELP_TEXT_KEY not in st.session_state and st.query_params.get("help") == "text":
        st.session_state[_HELP_TEXT_KEY] = True


def render_controls() -> None:
    """Language selector + help-mode toggle.

    Called before anything else renders so these are the first elements a
    screen reader reaches. The selector label is bilingual because it must be
    findable before a language has been chosen.
    """
    # Visually hidden hint, announced only by screen readers (standard
    # sr-only CSS). Screen readers cannot be detected — by deliberate browser
    # design, to avoid disclosing disability status — so instead this tells
    # them about the help-as-text mode before they reach the toggle.
    st.markdown(
        "<div style=\"position:absolute;width:1px;height:1px;padding:0;margin:-1px;"
        "overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;\">"
        + t(
            "Screen reader tip: check the 'Show help as visible text' checkbox "
            "below to read every control's explanation as regular text instead "
            "of hover-only tooltips."
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    codes = ["en"] + sorted(TABLES)
    col_lang, col_help, _spacer = st.columns([1.1, 1.9, 3.0])
    with col_lang:
        lang = st.selectbox(
            "Language / 言語",
            options=codes,
            index=codes.index(current_lang()) if current_lang() in codes else 0,
            format_func=lambda c: LANG_NAMES.get(c, c),
            key="ui_lang_select",
        )
        if lang != st.session_state.get(_LANG_KEY):
            st.session_state[_LANG_KEY] = lang
        if st.query_params.get("lang") != lang:
            st.query_params["lang"] = lang
    with col_help:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        on = st.checkbox(
            "Show help as visible text",
            key=_HELP_TEXT_KEY,
            help=(
                "Shows each control's explanation as plain text under the "
                "control instead of a hover tooltip. Recommended for screen "
                "readers; also lets the language setting translate every "
                "explanation."
            ),
        )
        if on and st.query_params.get("help") != "text":
            st.query_params["help"] = "text"
        elif not on and st.query_params.get("help") == "text":
            try:
                del st.query_params["help"]
            except Exception:
                pass


# --------------------------- widget translation shim ------------------------ #

def _wrap_widget(fn, translate_options: bool):
    def wrapper(label, *args, **kwargs):
        if isinstance(label, str):
            label = t(label)
        help_text = kwargs.get("help")
        if isinstance(help_text, str):
            help_text = t(help_text)
            kwargs["help"] = help_text
        show_caption = bool(help_text) and help_as_text()
        if show_caption:
            kwargs["help"] = None
        if translate_options and "format_func" not in kwargs:
            kwargs["format_func"] = lambda o: t(str(o))
        out = fn(label, *args, **kwargs)
        if show_caption:
            st.caption(help_text)
        return out

    wrapper._i18n_wrapped = True
    return wrapper


def _install() -> None:
    """Wrap widgets once per process (module import is cached across reruns)."""
    if getattr(st.slider, "_i18n_wrapped", False):
        return
    for name in (
        "slider",
        "checkbox",
        "number_input",
        "file_uploader",
        "button",
        "download_button",
        "expander",
        "subheader",
        "caption",
        "spinner",
    ):
        setattr(st, name, _wrap_widget(getattr(st, name), translate_options=False))
    for name in ("selectbox", "radio"):
        setattr(st, name, _wrap_widget(getattr(st, name), translate_options=True))


_install()
