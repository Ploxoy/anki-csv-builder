"""Streamlit entry point for the Anki CSV Builder app."""
from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

import logging
from pathlib import Path

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

gen_logger = logging.getLogger("core.generation")
gen_logger.setLevel(logging.DEBUG)
gen_logger.propagate = False  # чтобы не дублировать в общий поток

handler = logging.FileHandler(log_dir / "generation.debug.log", mode="w", encoding="utf-8")
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

gen_logger.handlers.clear()
gen_logger.addHandler(handler)


if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    __package__ = "app"  # type: ignore[misc]

# Ensure project root on sys.path when run via ``streamlit run app/app.py``
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.settings import (  # noqa: E402
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
    BASIC_CARD1_BACK_TEMPLATE_PATH as CFG_BASIC_CARD1_BACK_TEMPLATE_PATH,
    BASIC_CARD1_FRONT_TEMPLATE_PATH as CFG_BASIC_CARD1_FRONT_TEMPLATE_PATH,
    BASIC_CARD2_BACK_TEMPLATE_PATH as CFG_BASIC_CARD2_BACK_TEMPLATE_PATH,
    BASIC_CARD2_FRONT_TEMPLATE_PATH as CFG_BASIC_CARD2_FRONT_TEMPLATE_PATH,
    CLOZE_BACK_TEMPLATE_PATH as CFG_CLOZE_BACK_TEMPLATE_PATH,
    CLOZE_CSS_PATH as CFG_CLOZE_CSS_PATH,
    CLOZE_FRONT_TEMPLATE_PATH as CFG_CLOZE_FRONT_TEMPLATE_PATH,
    DEMO_WORDS as CFG_DEMO_WORDS,
    L1_LANGS as CFG_L1_LANGS,
    PAGE_LAYOUT as CFG_PAGE_LAYOUT,
    PAGE_TITLE as CFG_PAGE_TITLE,
    PROMPT_PROFILES as CFG_PROMPT_PROFILES,
    TYPEIN_BACK_TEMPLATE_PATH as CFG_TYPEIN_BACK_TEMPLATE_PATH,
    TYPEIN_FRONT_TEMPLATE_PATH as CFG_TYPEIN_FRONT_TEMPLATE_PATH,
)
from config.settings import SIGNALWORDS_B1 as CFG_SIGNALWORDS_B1  # noqa: E402
from config.settings import SIGNALWORDS_B2_PLUS as CFG_SIGNALWORDS_B2_PLUS  # noqa: E402
from config.settings import AUDIO_PROVIDER_DEFAULT as CFG_AUDIO_PROVIDER_DEFAULT  # noqa: E402
from config.settings import AUDIO_TTS_PROVIDERS as CFG_AUDIO_TTS_PROVIDERS  # noqa: E402
from config.settings import ELEVENLABS_DEFAULT_API_KEY as CFG_ELEVENLABS_KEY  # noqa: E402
from config.signalword_groups import SIGNALWORD_GROUPS as CFG_SIGNALWORD_GROUPS  # noqa: E402

from . import ui_helpers  # noqa: E402
from .generation_page import render_generation_page  # noqa: E402
from .input_ui import render_input_section  # noqa: E402
from .sidebar import render_sidebar  # noqa: E402
from .ui_models import AudioConfig, ExportConfig  # noqa: E402


st.set_page_config(page_title=CFG_PAGE_TITLE, layout=CFG_PAGE_LAYOUT)

ui_helpers.apply_theme()

ui_helpers.ensure_session_defaults(
    providers=CFG_AUDIO_TTS_PROVIDERS,
    default_provider=CFG_AUDIO_PROVIDER_DEFAULT,
    elevenlabs_default_key=CFG_ELEVENLABS_KEY,
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

if not sidebar_config.api_key or not st.session_state.get("input_data"):
    steps = [
        "Add your OpenAI API key in the sidebar (or load it from secrets).",
        "Pick a preset (Starter/Fast/Quality) or adjust model & CEFR manually.",
        "Upload a word list or use the demo/manual editor to populate input.",
        "Hit **Generate → Preview → Export** to run the quick flow.",
        "Review flagged rows in the preview — hover tooltips explain why they were flagged.",
    ]
    st.info("**Getting started**\n- " + "\n- ".join(steps))

render_input_section(CFG_DEMO_WORDS)

render_generation_page(
    sidebar_config,
    signalword_groups=CFG_SIGNALWORD_GROUPS,
    signalwords_b1=CFG_SIGNALWORDS_B1,
    signalwords_b2_plus=CFG_SIGNALWORDS_B2_PLUS,
    api_delay=CFG_API_DELAY,
    audio_config=AudioConfig(
        providers=CFG_AUDIO_TTS_PROVIDERS,
        default_provider=CFG_AUDIO_PROVIDER_DEFAULT,
    ),
    export_config=ExportConfig(
        csv_delimiter=CFG_CSV_DELIM,
        csv_lineterminator=CFG_CSV_EOL,
        anki_model_id=CFG_ANKI_MODEL_ID,
        anki_deck_id=CFG_ANKI_DECK_ID,
        anki_model_name=CFG_ANKI_MODEL_NAME,
        anki_deck_name=CFG_ANKI_DECK_NAME,
        front_template_path=CFG_CLOZE_FRONT_TEMPLATE_PATH,
        back_template_path=CFG_CLOZE_BACK_TEMPLATE_PATH,
        css_path=CFG_CLOZE_CSS_PATH,
        basic_templates={
            "card1_front": CFG_BASIC_CARD1_FRONT_TEMPLATE_PATH,
            "card1_back": CFG_BASIC_CARD1_BACK_TEMPLATE_PATH,
            "card2_front": CFG_BASIC_CARD2_FRONT_TEMPLATE_PATH,
            "card2_back": CFG_BASIC_CARD2_BACK_TEMPLATE_PATH,
        },
        typein_templates={
            "front": CFG_TYPEIN_FRONT_TEMPLATE_PATH,
            "back": CFG_TYPEIN_BACK_TEMPLATE_PATH,
        },
    ),
)

st.caption(
    "Tips: 1) Better Dutch definitions on input → better examples and glosses. "
    "2) From B1, ~50% of sentences include a signal word. "
    "3) Some models (gpt-5/o3) ignore temperature and will be retried without it."
)
