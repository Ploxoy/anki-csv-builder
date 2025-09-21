"""Streamlit entry point for the Anki CSV Builder app."""
from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    __package__ = "app"  # type: ignore[misc]

# Ensure project root on sys.path when run via ``streamlit run app/app.py``
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.settings import (  # noqa: E402
    AUDIO_INCLUDE_SENTENCE_DEFAULT as CFG_AUDIO_INCLUDE_SENTENCE_DEFAULT,
    AUDIO_INCLUDE_WORD_DEFAULT as CFG_AUDIO_INCLUDE_WORD_DEFAULT,
    AUDIO_SENTENCE_INSTRUCTION_DEFAULT as CFG_SENTENCE_INSTR_DEFAULT,
    AUDIO_TTS_FALLBACK as CFG_AUDIO_TTS_FALLBACK,
    AUDIO_TTS_INSTRUCTIONS as CFG_AUDIO_TTS_INSTRUCTIONS,
    AUDIO_TTS_MODEL as CFG_AUDIO_TTS_MODEL,
    AUDIO_VOICES as CFG_AUDIO_VOICES,
    CSV_DELIMITER as CFG_CSV_DELIM,
    CSV_LINETERMINATOR as CFG_CSV_EOL,
    DEFAULT_MODELS as CFG_DEFAULT_MODELS,
    TEMPERATURE_DEFAULT as CFG_TDEF,
    TEMPERATURE_MAX as CFG_TMAX,
    TEMPERATURE_MIN as CFG_TMIN,
    TEMPERATURE_STEP as CFG_TSTEP,
    _ALLOWED_PREFIXES as CFG_ALLOWED_PREFIXES,
    _BLOCK_SUBSTRINGS as CFG_BLOCK_SUBSTRINGS,
    _PREFERRED_ORDER as CFG_PREFERRED_ORDER,
    ANKI_DECK_ID as CFG_ANKI_DECK_ID,
    ANKI_DECK_NAME as CFG_ANKI_DECK_NAME,
    ANKI_MODEL_ID as CFG_ANKI_MODEL_ID,
    ANKI_MODEL_NAME as CFG_ANKI_MODEL_NAME,
    API_REQUEST_DELAY as CFG_API_DELAY,
    AUDIO_WORD_INSTRUCTION_DEFAULT as CFG_WORD_INSTR_DEFAULT,
    BACK_HTML_TEMPLATE as CFG_BACK_HTML_TEMPLATE,
    CSS_STYLING as CFG_CSS_STYLING,
    DEMO_WORDS as CFG_DEMO_WORDS,
    FRONT_HTML_TEMPLATE as CFG_FRONT_HTML_TEMPLATE,
    L1_LANGS as CFG_L1_LANGS,
    PAGE_LAYOUT as CFG_PAGE_LAYOUT,
    PAGE_TITLE as CFG_PAGE_TITLE,
    PROMPT_PROFILES as CFG_PROMPT_PROFILES,
)
from config.settings import SIGNALWORDS_B1 as CFG_SIGNALWORDS_B1  # noqa: E402
from config.settings import SIGNALWORDS_B2_PLUS as CFG_SIGNALWORDS_B2_PLUS  # noqa: E402
from config.signalword_groups import SIGNALWORD_GROUPS as CFG_SIGNALWORD_GROUPS  # noqa: E402

from . import ui_helpers  # noqa: E402
from .generation_page import AudioConfig, ExportConfig, render_generation_page  # noqa: E402
from .input_ui import render_input_section  # noqa: E402
from .sidebar import render_sidebar  # noqa: E402


st.set_page_config(page_title=CFG_PAGE_TITLE, layout=CFG_PAGE_LAYOUT)

default_voice = CFG_AUDIO_VOICES[0]["id"] if CFG_AUDIO_VOICES else ""
ui_helpers.ensure_session_defaults(
    default_voice=default_voice,
    include_word_default=CFG_AUDIO_INCLUDE_WORD_DEFAULT,
    include_sentence_default=CFG_AUDIO_INCLUDE_SENTENCE_DEFAULT,
    sentence_instruction_default=CFG_SENTENCE_INSTR_DEFAULT,
    word_instruction_default=CFG_WORD_INSTR_DEFAULT,
)

# Apply deferred batch parameters (set before widget creation)
pending_bs = st.session_state.pop("batch_size_pending", None)
if pending_bs is not None:
    st.session_state["batch_size"] = int(pending_bs)
pending_workers = st.session_state.pop("max_workers_pending", None)
if pending_workers is not None:
    st.session_state["max_workers"] = int(pending_workers)
pending_auto = st.session_state.pop("auto_advance_pending", None)
if pending_auto is not None:
    st.session_state["auto_advance"] = bool(pending_auto)

sidebar_config = render_sidebar(
    default_models=CFG_DEFAULT_MODELS,
    preferred_order=CFG_PREFERRED_ORDER,
    block_substrings=CFG_BLOCK_SUBSTRINGS,
    allowed_prefixes=CFG_ALLOWED_PREFIXES,
    prompt_profiles=CFG_PROMPT_PROFILES,
    l1_langs=CFG_L1_LANGS,
    temperature_min=CFG_TMIN,
    temperature_max=CFG_TMAX,
    temperature_default=CFG_TDEF,
    temperature_step=CFG_TSTEP,
)

st.title("📘 Anki CSV/Anki Builder — Dutch Cloze Cards")

with st.expander("ℹ️ Quick help", expanded=False):
    st.markdown(
        """
1. Load data via **Upload** or build it in **Manual editor**.
2. Pick model / CEFR / profile / L1, set optional temperature/token limits.
3. Press **Generate cards** — preview shows the first rows and flags.
4. (Optional) Open **Audio (optional)** to synthesize word/sentence MP3 with a chosen voice & style.
5. Download the ready CSV or Anki deck once everything looks good.
6. In Anki Desktop: File → Import → select `anki_cards.csv` (Notes (Cloze), delimiter `|`) or `dutch_cloze.apkg`, then confirm field mapping and import.
        """
    )

render_input_section(CFG_DEMO_WORDS)

render_generation_page(
    sidebar_config,
    signalword_groups=CFG_SIGNALWORD_GROUPS,
    signalwords_b1=CFG_SIGNALWORDS_B1,
    signalwords_b2_plus=CFG_SIGNALWORDS_B2_PLUS,
    api_delay=CFG_API_DELAY,
    audio_config=AudioConfig(
        voices=CFG_AUDIO_VOICES,
        model=CFG_AUDIO_TTS_MODEL,
        fallback_model=CFG_AUDIO_TTS_FALLBACK,
        instructions=CFG_AUDIO_TTS_INSTRUCTIONS,
    ),
    export_config=ExportConfig(
        csv_delimiter=CFG_CSV_DELIM,
        csv_lineterminator=CFG_CSV_EOL,
        anki_model_id=CFG_ANKI_MODEL_ID,
        anki_deck_id=CFG_ANKI_DECK_ID,
        anki_model_name=CFG_ANKI_MODEL_NAME,
        anki_deck_name=CFG_ANKI_DECK_NAME,
        front_template=CFG_FRONT_HTML_TEMPLATE,
        back_template=CFG_BACK_HTML_TEMPLATE,
        css=CFG_CSS_STYLING,
    ),
)

st.caption(
    "Tips: 1) Better Dutch definitions on input → better examples and glosses. "
    "2) From B1, ~50% of sentences include a signal word. "
    "3) Some models (gpt-5/o3) ignore temperature and will be retried without it."
)
