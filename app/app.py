# app/app.py
# ------------------------------------
# Streamlit app to generate Dutch Cloze Anki cards from text input.
# - Uses OpenAI Responses API (official SDK) with strict JSON output expectations.
# - Prompt lives in prompts.py; tweak content there.
# - Configurable via config.py; sensible fallbacks if missing.
# - Includes CSV export and Anki .apkg export (via genanki).
# - Enforces cloze double-braces locally (auto-repair) to tolerate smaller models.

import os
import sys
import time
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# Allow importing project modules when launched as ``streamlit run app/app.py``
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# anki_csv_builder.py (near other core imports)
from core.llm_clients import create_client
from core.generation import GenerationSettings, generate_card
from core.export_csv import generate_csv
from core.export_anki import build_anki_package, HAS_GENANKI
from core.parsing import parse_input
from core.sanitize_validate import is_probably_dutch_word
from core.audio import ensure_audio_for_cards, sentence_for_tts

# Config (import from settings)
from config.signalword_groups import SIGNALWORD_GROUPS as CFG_SIGNALWORD_GROUPS
from config.settings import (
    DEFAULT_MODELS as CFG_DEFAULT_MODELS,
    _PREFERRED_ORDER as CFG_PREFERRED_ORDER,
    _BLOCK_SUBSTRINGS as CFG_BLOCK_SUBSTRINGS,
    _ALLOWED_PREFIXES as CFG_ALLOWED_PREFIXES,
    SIGNALWORDS_B1 as CFG_SIGNALWORDS_B1,
    SIGNALWORDS_B2_PLUS as CFG_SIGNALWORDS_B2_PLUS,
    PROMPT_PROFILES as CFG_PROMPT_PROFILES,
    L1_LANGS as CFG_L1_LANGS,
    DEMO_WORDS as CFG_DEMO_WORDS,
    PAGE_TITLE as CFG_PAGE_TITLE,
    PAGE_LAYOUT as CFG_PAGE_LAYOUT,
    TEMPERATURE_MIN as CFG_TMIN,
    TEMPERATURE_MAX as CFG_TMAX,
    TEMPERATURE_DEFAULT as CFG_TDEF,
    TEMPERATURE_STEP as CFG_TSTEP,
    PREVIEW_LIMIT as CFG_PREVIEW_LIMIT,
    API_REQUEST_DELAY as CFG_API_DELAY,
    ANKI_MODEL_ID as CFG_ANKI_MODEL_ID,
    ANKI_DECK_ID as CFG_ANKI_DECK_ID,
    ANKI_MODEL_NAME as CFG_ANKI_MODEL_NAME,
    ANKI_DECK_NAME as CFG_ANKI_DECK_NAME,
    FRONT_HTML_TEMPLATE as CFG_FRONT_HTML_TEMPLATE,
    BACK_HTML_TEMPLATE as CFG_BACK_HTML_TEMPLATE,
    CSS_STYLING as CFG_CSS_STYLING,
    CSV_DELIMITER as CFG_CSV_DELIM,
    CSV_LINETERMINATOR as CFG_CSV_EOL,
    AUDIO_TTS_MODEL as CFG_AUDIO_TTS_MODEL,
    AUDIO_TTS_FALLBACK as CFG_AUDIO_TTS_FALLBACK,
    AUDIO_TTS_INSTRUCTIONS as CFG_AUDIO_TTS_INSTRUCTIONS,
    AUDIO_SENTENCE_INSTRUCTION_DEFAULT as CFG_SENTENCE_INSTR_DEFAULT,
    AUDIO_WORD_INSTRUCTION_DEFAULT as CFG_WORD_INSTR_DEFAULT,
    AUDIO_VOICES as CFG_AUDIO_VOICES,
    AUDIO_INCLUDE_WORD_DEFAULT as CFG_AUDIO_INCLUDE_WORD_DEFAULT,
    AUDIO_INCLUDE_SENTENCE_DEFAULT as CFG_AUDIO_INCLUDE_SENTENCE_DEFAULT,
)
st.set_page_config(page_title=CFG_PAGE_TITLE, layout=CFG_PAGE_LAYOUT)

# Model discovery and filtering
DEFAULT_MODELS: List[str] = CFG_DEFAULT_MODELS
_PREFERRED_ORDER = CFG_PREFERRED_ORDER
_BLOCK_SUBSTRINGS = CFG_BLOCK_SUBSTRINGS
_ALLOWED_PREFIXES = CFG_ALLOWED_PREFIXES


_LEVEL_ORDER = ["A1","A2","B1","B2","C1","C2"]

def _levels_up_to(level: str) -> list[str]:
    """Inclusive list of levels from A1 up to 'level' (for pooling)."""
    try:
        idx = _LEVEL_ORDER.index(level)
    except ValueError:
        idx = 2  # default B1
    return _LEVEL_ORDER[:idx+1]



def _init_sig_usage():
    """Session-persistent usage counter for signal words (per run)."""
    if "sig_usage" not in st.session_state:
        st.session_state.sig_usage = {}  # word -> count
    if "sig_last" not in st.session_state:
        st.session_state.sig_last = None


def _init_response_format_cache():
    if "no_response_format_models" not in st.session_state:
        st.session_state.no_response_format_models = set()
    if "no_response_format_notified" not in st.session_state:
        st.session_state.no_response_format_notified = set()


def _clean_manual_rows(rows: List[Dict]) -> List[Dict[str, str]]:
    """Normalize manual editor rows: trim fields, drop incomplete entries."""
    cleaned: List[Dict[str, str]] = []
    for raw in rows:
        woord = str(raw.get("woord", "") or "").strip()
        def_nl = str(raw.get("def_nl", "") or "").strip()
        translation = str(raw.get("translation", "") or "").strip()
        if not (woord or def_nl or translation):
            continue  # fully empty row
        if not woord:
            continue  # skip rows without target word
        cleaned.append({
            "woord": woord,
            "def_nl": def_nl,
            "translation": translation,
        })
    return cleaned



def _sort_key(model_id: str) -> tuple:
    for k, rank in _PREFERRED_ORDER.items():
        if model_id.startswith(k):
            return (rank, model_id)
    return (999, model_id)

def get_model_options(api_key: str | None) -> List[str]:
    """Fetch available models and filter out non-text ones; fallback to defaults on errors."""
    if not api_key:
        return DEFAULT_MODELS
    try:
        #client = OpenAI(api_key=api_key)
        client = create_client(API_KEY)
        if client is None:
                st.error("OpenAI SDK not available (core.llm_clients.create_client returned None). Please install the OpenAI SDK.")
                return
        models = client.models.list()
        ids = []
        for m in getattr(models, "data", []) or []:
            mid = getattr(m, "id", "")
            if any(mid.startswith(p) for p in _ALLOWED_PREFIXES) and not any(b in mid for b in _BLOCK_SUBSTRINGS):
                ids.append(mid)
        if not ids:
            return DEFAULT_MODELS
        return sorted(set(ids), key=_sort_key)
    except Exception:
        return DEFAULT_MODELS

# CEFR & signaalwoorden (from config)
SIGNALWORDS_B1: List[str] = CFG_SIGNALWORDS_B1
SIGNALWORDS_B2_PLUS: List[str] = CFG_SIGNALWORDS_B2_PLUS

L1_LANGS = CFG_L1_LANGS
PROMPT_PROFILES = CFG_PROMPT_PROFILES

# ----- Sidebar: API, model, params -----
st.sidebar.header("🔐 API Settings")

def _get_secret(name: str):
    """Try streamlit secrets, then env var; return None if missing."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)

API_KEY = (
    _get_secret("OPENAI_API_KEY")
    or st.sidebar.text_input("OpenAI API Key", type="password")
)

# SDK version hint (optional)
try:
    import openai as _openai  # type: ignore
    st.sidebar.caption(f"OpenAI SDK: v{_openai.__version__}")
except Exception:
    pass

model_options = get_model_options(API_KEY)

# -- changed: prefer "gpt-4.1-mini" as the default selection if available --
_default_model_preferred = "gpt-4.1-mini"
try:
    _default_index = model_options.index(_default_model_preferred)
except ValueError:
    _default_index = 0

model = st.sidebar.selectbox(
    "Model",
    model_options,
    index=_default_index,
    help="Best quality — gpt-5 (if available); balanced — gpt-4.1; faster/cheaper — gpt-4o / gpt-5-mini.",
)

profile = st.sidebar.selectbox(
    "Prompt profile",
    list(PROMPT_PROFILES.keys()),
    index=list(PROMPT_PROFILES.keys()).index("strict") if "strict" in PROMPT_PROFILES else 0,
)

level = st.sidebar.selectbox("CEFR", ["A1", "A2", "B1", "B2", "C1", "C2"], index=2)

L1_code = st.sidebar.selectbox("Your language (L1)", list(L1_LANGS.keys()), index=0)
L1_meta = L1_LANGS[L1_code]

TMIN, TMAX, TDEF, TSTEP = CFG_TMIN, CFG_TMAX, CFG_TDEF, CFG_TSTEP
temperature = st.sidebar.slider("Temperature", TMIN, TMAX, TDEF, TSTEP)

# Add a checkbox to control token limiting
limit_tokens = st.sidebar.checkbox(
    "Limit output tokens",
    value=True,
    help="Check to limit the number of output tokens. Uncheck to allow unlimited tokens."
)

# Add a checkbox to control raw response display
display_raw_response = st.sidebar.checkbox(
    "Display raw responses",
    value=False,
    help="Check to display raw responses from the OpenAI API."
)

# CSV & Anki export options
csv_with_header = st.sidebar.checkbox(
    "CSV: include header row",
    value=True,
    help="Uncheck if your Anki import treats the first row as a record."
)
csv_anki_header = st.sidebar.checkbox(
    "CSV: use Anki field names",
    value=True,
    help="Header row will be: L2_word|L2_cloze|L1_sentence|L2_collocations|L2_definition|L1_gloss|L1_hint"
)
# Sidebar: option to force generation for flagged entries
force_flagged = st.sidebar.checkbox(
    "Force generate for flagged entries",
    value=False,
    help="If off, rows flagged as suspicious by a quick heuristic will be skipped from generation."
)
st.session_state["force_flagged"] = force_flagged


_guid_label = st.sidebar.selectbox(
    "Anki GUID policy",
    ["stable (update/skip existing)", "unique per export (import as new)"],
    index=0,
    help=("stable: the same notes are recognized as existing/updatable\n"
          "unique: each export has new GUIDs — Anki treats them as new notes."),
)
st.session_state["csv_with_header"] = csv_with_header
st.session_state["anki_guid_policy"] = "unique" if _guid_label.startswith("unique") else "stable"

# Persist selections in session
st.session_state["prompt_profile"] = profile
st.session_state["level"] = level
st.session_state["L1_code"] = L1_code

# ----- App title -----
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

# ----- Demo & clear -----
DEMO_WORDS = CFG_DEMO_WORDS

if "input_data" not in st.session_state:
    st.session_state.input_data: List[Dict] = [] # type: ignore
if "results" not in st.session_state:
    st.session_state.results: List[Dict] = [] # type: ignore
if "manual_rows" not in st.session_state:
    st.session_state.manual_rows = [{"woord": "", "def_nl": "", "translation": ""}]
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "audio_media" not in st.session_state:
    st.session_state.audio_media = {}
if "audio_summary" not in st.session_state:
    st.session_state.audio_summary = None
if "audio_voice" not in st.session_state:
    default_voice = CFG_AUDIO_VOICES[0]["id"] if CFG_AUDIO_VOICES else ""
    st.session_state.audio_voice = default_voice
if "audio_include_word" not in st.session_state:
    st.session_state.audio_include_word = CFG_AUDIO_INCLUDE_WORD_DEFAULT
if "audio_include_sentence" not in st.session_state:
    st.session_state.audio_include_sentence = CFG_AUDIO_INCLUDE_SENTENCE_DEFAULT
if "audio_sentence_instruction" not in st.session_state:
    st.session_state.audio_sentence_instruction = CFG_SENTENCE_INSTR_DEFAULT
if "audio_word_instruction" not in st.session_state:
    st.session_state.audio_word_instruction = CFG_WORD_INSTR_DEFAULT
if "audio_panel_expanded" not in st.session_state:
    st.session_state.audio_panel_expanded = bool(st.session_state.get("results"))

col_demo, col_clear = st.columns([1, 1])
with col_demo:
    if st.button("Try demo", type="secondary"):
        st.session_state.input_data = [dict(row) for row in DEMO_WORDS]
        st.session_state.manual_rows = [
            {"woord": row.get("woord", ""), "def_nl": row.get("def_nl", ""), "translation": row.get("translation", "") or ""}
            for row in DEMO_WORDS
        ]
        st.toast("✅ Demo set (6 words) loaded", icon="✅")
with col_clear:
    if st.button("Clear", type="secondary"):
        st.session_state.input_data = []
        st.session_state.results = []
        st.session_state.manual_rows = [{"woord": "", "def_nl": "", "translation": ""}]
        st.session_state.audio_media = {}
        st.session_state.audio_summary = None

# ----- Input tabs: upload vs manual editor -----
tab_upload, tab_manual = st.tabs(["📄 Upload", "✍️ Manual editor"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload .txt / .md / .tsv / .csv",
        type=["txt", "md", "tsv", "csv"],
        accept_multiple_files=False,
        key="file_uploader",
    )

    if uploaded_file is not None:
        try:
            file_text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            file_text = uploaded_file.read().decode("utf-16")
        st.session_state.input_data = parse_input(file_text)
        st.session_state.manual_rows = [
            {"woord": row.get("woord", ""), "def_nl": row.get("def_nl", ""), "translation": row.get("translation", "") or ""}
            for row in st.session_state.input_data
        ]
        st.session_state.results = []
        st.toast("📥 Input replaced with uploaded file", icon="📄")

with tab_manual:
    manual_cols = st.columns([1, 1, 1])
    with manual_cols[0]:
        if st.button("Reset list", key="manual_reset"):
            st.session_state.manual_rows = [{"woord": "", "def_nl": "", "translation": ""}]
    with manual_cols[1]:
        if st.button("Load demo", key="manual_seed_demo"):
            st.session_state.manual_rows = [
                {"woord": row.get("woord", ""), "def_nl": row.get("def_nl", ""), "translation": row.get("translation", "") or ""}
                for row in DEMO_WORDS
            ]
    with manual_cols[2]:
        st.caption("Добавьте слова, определения и при необходимости перевод. Строки можно редактировать и удалять.")

    manual_rows = list(st.session_state.manual_rows or [])
    def _is_empty(row):
        return not (
            (row.get("woord") or "").strip()
            or (row.get("def_nl") or "").strip()
            or (row.get("translation") or "").strip()
        )
    if not manual_rows or not _is_empty(manual_rows[-1]):
        manual_rows.append({"woord": "", "def_nl": "", "translation": ""})

    manual_df = pd.DataFrame(manual_rows).reindex(columns=["woord", "def_nl", "translation"])

    edited_df = st.data_editor(
        manual_df,
        key="manual_editor",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "woord": st.column_config.TextColumn("woord", help="Целевое нидерландское слово"),
            "def_nl": st.column_config.TextColumn("def_nl", help="Опционально: определение или контекст"),
            "translation": st.column_config.TextColumn("translation", help="Опционально: перевод на L1"),
        },
    )

    st.session_state.manual_rows = edited_df.fillna("").to_dict("records")
    manual_clean = _clean_manual_rows(st.session_state.manual_rows)

    apply_col, info_col = st.columns([1, 1])
    with apply_col:
        if st.button("Use manual list", type="primary", key="manual_apply"):
            if manual_clean:
                st.session_state.input_data = manual_clean
                st.session_state.results = []
                st.toast(f"✍️ Loaded {len(manual_clean)} manual rows", icon="✍️")
            else:
                st.warning("Нужно заполнить хотя бы одну строку с полем 'woord'.")
    with info_col:
        st.caption(f"Активные строки: {len(manual_clean)}")

# Preview input
# Preview input — annotate parsed rows with quick Dutch-word heuristic and show flags
if st.session_state.input_data:
    # Annotate parsed rows with a quick Dutch-word heuristic
    for row in st.session_state.input_data:
        ok, reason = is_probably_dutch_word(row.get("woord", ""))
        row["_flag_ok"] = bool(ok)
        row["_flag_reason"] = reason or ""

    # Show warning if any rows are flagged
    flagged = [r for r in st.session_state.input_data if not r.get("_flag_ok", True)]
    if flagged:
        st.warning(
            f"{len(flagged)} rows flagged as suspicious by a quick heuristic. "
            "Use 'Force generate for flagged entries' in the sidebar to ignore flags."
        )

    st.subheader("🔍 Parsed rows")
    # Create a small DataFrame copy to show flags
    preview_in = pd.DataFrame(st.session_state.input_data)
    cols = [c for c in ["woord", "def_nl", "ru_short", "_flag_ok", "_flag_reason"] if c in preview_in.columns]
    st.dataframe(preview_in[cols], use_container_width=True)
else:
    st.info("Upload a file or click **Try demo**")

# ----- Helpers: sanitize, temperature, signaalwoord policy -----
def _should_pass_temperature(model_id: str) -> bool:
    """Some models (gpt-5/o3) do not accept temperature."""
    no_temp = st.session_state.get("no_temp_models", set())
    if model_id in no_temp:
        return False
    if model_id.startswith(("gpt-5", "o3")):
        return False
    return True
# ----- Generate section -----
if st.session_state.input_data:
    if st.button("Generate cards", type="primary"):
        if not API_KEY:
            st.error("Provide OPENAI_API_KEY via Secrets, environment variable, or the input field.")
        else:
            client = OpenAI(api_key=API_KEY)
            # Stable run id for this batch (used in 'unique' GUID policy)
            st.session_state.anki_run_id = st.session_state.get("anki_run_id") or str(int(time.time()))
            st.session_state.results = []
            st.session_state.model_id = model
            st.session_state.audio_media = {}
            st.session_state.audio_summary = None
            total = len(st.session_state.input_data)
            progress = st.progress(0)
            max_tokens = 3000 if limit_tokens else None
            effective_temp = temperature if _should_pass_temperature(model) else None
            _init_sig_usage()
            _init_response_format_cache()
            rng = random.Random()
            for idx, row in enumerate(st.session_state.input_data):
                try:
                    # Skip flagged rows unless user explicitly forces generation
                    if not st.session_state.get("force_flagged", False) and not row.get("_flag_ok", True):
                        # Add a small result entry indicating skip so user sees it in preview
                        st.session_state.results.append({
                            "L2_word": row.get("woord", ""),
                            "L2_cloze": "",
                            "L1_sentence": "",
                            "L2_collocations": "",
                            "L2_definition": "",
                            "L1_gloss": "",
                            "L1_hint": "",
                            "AudioSentence": "",
                            "AudioWord": "",
                            "error": "flagged_precheck",
                            "meta": {"flag_reason": row.get("_flag_reason", "")}
                        })
                        continue

                    card_seed = rng.randint(0, 2**31 - 1)
                    settings = GenerationSettings(
                        model=model,
                        L1_code=L1_code,
                        L1_name=L1_meta["name"],
                        level=level,
                        profile=profile,
                        temperature=effective_temp,
                        max_output_tokens=max_tokens,
                        allow_response_format=model not in st.session_state.get("no_response_format_models", set()),
                        signalword_seed=card_seed,
                    )
                    gen_result = generate_card(
                        client=client,
                        row=row,
                        settings=settings,
                        signalword_groups=CFG_SIGNALWORD_GROUPS,
                        signalwords_b1=CFG_SIGNALWORDS_B1,
                        signalwords_b2_plus=CFG_SIGNALWORDS_B2_PLUS,
                        signal_usage=st.session_state.get("sig_usage"),
                        signal_last=st.session_state.get("sig_last"),
                    )
                    st.session_state.sig_usage = gen_result.signal_usage
                    st.session_state.sig_last = gen_result.signal_last
                    st.session_state.results.append(gen_result.card)
                    meta = gen_result.card.get("meta", {})
                    if meta.get("response_format_removed"):
                        cache = set(st.session_state.get("no_response_format_models", set()))
                        notified = set(st.session_state.get("no_response_format_notified", set()))
                        if model not in cache:
                            cache.add(model)
                            st.session_state.no_response_format_models = cache
                        if model not in notified:
                            notified.add(model)
                            st.session_state.no_response_format_notified = notified
                            detail = meta.get("response_format_error")
                            message = (
                                f"Model {model} ignored response_format; falling back to text parsing for this session."
                            )
                            if detail:
                                message += f"\nReason: {detail}"
                            st.info(message, icon="ℹ️")
                except Exception as e:
                    st.error(f"Error for word '{row.get('woord','?')}': {e}")
                finally:
                    progress.progress(int((idx + 1) / max(total, 1) * 100))
                    if CFG_API_DELAY > 0:
                        time.sleep(CFG_API_DELAY)

# ----- Preview & downloads -----
if st.session_state.results:
    preview_container = st.container()

    st.divider()
    audio_summary = st.session_state.get("audio_summary")
    audio_panel_expanded = st.session_state.get("audio_panel_expanded", False)
    if audio_summary:
        audio_panel_expanded = True
        st.session_state.audio_panel_expanded = True

    with st.expander("🔊 Audio (optional)", expanded=audio_panel_expanded):
        st.session_state.audio_panel_expanded = True
        if not CFG_AUDIO_VOICES:
            st.info("Голоса TTS пока не настроены.")
        else:
            voice_index = 0
            current_voice = st.session_state.get("audio_voice", CFG_AUDIO_VOICES[0]["id"])
            for idx, option in enumerate(CFG_AUDIO_VOICES):
                if option["id"] == current_voice:
                    voice_index = idx
                    break
            voice_choice = st.selectbox(
                "Голос",
                options=CFG_AUDIO_VOICES,
                index=voice_index,
                format_func=lambda opt: opt["label"],
                key="audio_voice_select",
            )
            selected_voice = voice_choice["id"]
            st.session_state.audio_voice = selected_voice

            include_word = st.checkbox(
                "Include word audio",
                value=st.session_state.get("audio_include_word", CFG_AUDIO_INCLUDE_WORD_DEFAULT),
                key="audio_include_word",
            )
            include_sentence = st.checkbox(
                "Include sentence audio",
                value=st.session_state.get("audio_include_sentence", CFG_AUDIO_INCLUDE_SENTENCE_DEFAULT),
                key="audio_include_sentence",
            )

            def _instruction_label(key: str) -> str:
                if key.startswith("Dutch_sentence_"):
                    suffix = key.split("Dutch_sentence_", 1)[1].replace("_", " ")
                    return f"Sentence · {suffix.capitalize()}"
                if key.startswith("Dutch_word_"):
                    suffix = key.split("Dutch_word_", 1)[1].replace("_", " ")
                    return f"Word · {suffix.capitalize()}"
                return key

            sentence_options = sorted(
                [k for k in CFG_AUDIO_TTS_INSTRUCTIONS if k.startswith("Dutch_sentence_")]
            )
            word_options = sorted(
                [k for k in CFG_AUDIO_TTS_INSTRUCTIONS if k.startswith("Dutch_word_")]
            )

            sentence_choice = st.selectbox(
                "Sentence style",
                options=sentence_options,
                format_func=_instruction_label,
                key="audio_sentence_instruction",
            )
            sentence_caption = st.empty()
            sentence_caption.caption(
                CFG_AUDIO_TTS_INSTRUCTIONS.get(sentence_choice, "") or " "
            )

            word_choice = st.selectbox(
                "Word style",
                options=word_options,
                format_func=_instruction_label,
                key="audio_word_instruction",
            )
            word_caption = st.empty()
            word_caption.caption(
                CFG_AUDIO_TTS_INSTRUCTIONS.get(word_choice, "") or " "
            )

            cards = st.session_state.results
            unique_words = set()
            unique_sentences = set()
            for card in cards:
                woord_text = (card.get("L2_word") or "").strip()
                if woord_text:
                    unique_words.add(woord_text)
                sentence_text = sentence_for_tts(card.get("L2_cloze", ""))
                if sentence_text:
                    unique_sentences.add(sentence_text)

            requests_estimate = 0
            if include_word:
                requests_estimate += len(unique_words)
            if include_sentence:
                requests_estimate += len(unique_sentences)

            st.caption(
                "Estimated requests: "
                f"{requests_estimate} (unique words — {len(unique_words)}, sentences — {len(unique_sentences)})."
            )

            if audio_summary:
                cache_hits = audio_summary.get("cache_hits", 0)
                fallback_hits = audio_summary.get("fallback_switches", 0)
                msg = (
                    f"Done: words — {audio_summary.get('word_success', 0)}, "
                    f"sentences — {audio_summary.get('sentence_success', 0)}."
                )
                if cache_hits:
                    msg += f" Cache hits: {cache_hits}."
                if fallback_hits:
                    msg += f" Fallback used: {fallback_hits}×."
                st.success(msg)
                errors = audio_summary.get("errors") or []
                if errors:
                    preview_err = "; ".join(errors[:3])
                    if len(errors) > 3:
                        preview_err += " …"
                    st.warning(f"Audio issues: {preview_err}")
                styles = []
                sent_key = audio_summary.get("sentence_instruction_key") or ""
                word_key = audio_summary.get("word_instruction_key") or ""
                if sent_key:
                    styles.append(f"sentence: {_instruction_label(sent_key)}")
                if word_key:
                    styles.append(f"word: {_instruction_label(word_key)}")
                if styles:
                    st.caption("Styles → " + "; ".join(styles))

            button_disabled = requests_estimate == 0
            generate_audio = st.button(
                "🔊 Generate audio",
                type="primary",
                disabled=button_disabled,
                key="generate_audio_button",
            )

            if generate_audio:
                if button_disabled:
                    st.info("No text to synthesize — enable word or sentence above.")
                elif not API_KEY:
                    st.error("OPENAI_API_KEY is required for audio synthesis.")
                else:
                    client = OpenAI(api_key=API_KEY)
                    progress = st.progress(0)

                    def _progress(done: int, total: int) -> None:
                        if total <= 0:
                            progress.progress(1.0)
                        else:
                            progress.progress(min(1.0, done / total))

                    try:
                        instruction_keys = {
                            "sentence": sentence_choice,
                            "word": word_choice,
                        }
                        instruction_texts = {
                            "sentence": CFG_AUDIO_TTS_INSTRUCTIONS.get(
                                sentence_choice,
                                "",
                            ),
                            "word": CFG_AUDIO_TTS_INSTRUCTIONS.get(
                                word_choice,
                                "",
                            ),
                        }

                        media_map, summary_obj = ensure_audio_for_cards(
                            st.session_state.results,
                            client=client,
                            model=CFG_AUDIO_TTS_MODEL,
                            fallback_model=CFG_AUDIO_TTS_FALLBACK,
                            voice=selected_voice,
                            include_word=include_word,
                            include_sentence=include_sentence,
                            cache=st.session_state.audio_cache,
                            progress_cb=_progress,
                            instructions=instruction_texts,
                            instruction_keys=instruction_keys,
                        )
                        progress.progress(1.0)
                        st.session_state.audio_media = media_map
                        st.session_state.audio_summary = asdict(summary_obj)
                        st.session_state.audio_voice = selected_voice

                        success_msg = (
                            f"Done: words — {summary_obj.word_success}, "
                            f"sentences — {summary_obj.sentence_success}."
                        )
                        if summary_obj.cache_hits:
                            success_msg += f" Cache hits: {summary_obj.cache_hits}."
                        if summary_obj.fallback_switches:
                            success_msg += f" Fallback used: {summary_obj.fallback_switches}×."
                        styles_now = []
                        if summary_obj.sentence_instruction_key:
                            styles_now.append(
                                f"sentence: {_instruction_label(summary_obj.sentence_instruction_key)}"
                            )
                        if summary_obj.word_instruction_key:
                            styles_now.append(
                                f"word: {_instruction_label(summary_obj.word_instruction_key)}"
                            )
                        if styles_now:
                            success_msg += " | Styles → " + "; ".join(styles_now)
                        st.success(success_msg)
                        if summary_obj.errors:
                            err_preview = "; ".join(summary_obj.errors[:3])
                            if len(summary_obj.errors) > 3:
                                err_preview += " …"
                            st.warning(f"Audio issues: {err_preview}")
                        # Reassign results to force dataframe refresh with audio fields
                        st.session_state.results = [dict(card) for card in st.session_state.results]
                    except Exception as exc:  # pragma: no cover - network interaction
                        st.error(f"Audio synthesis failed: {exc}")

        if st.button("Hide audio options", key="hide_audio_panel"):
            st.session_state.audio_panel_expanded = False

    with preview_container:
        st.subheader(f"📋 Preview (first {CFG_PREVIEW_LIMIT})")
        preview_df = pd.DataFrame(st.session_state.results)[:CFG_PREVIEW_LIMIT]
        st.dataframe(preview_df, use_container_width=True)

    csv_extras = {
        "level": st.session_state.get("level", level),
        "profile": st.session_state.get("prompt_profile", profile),
        "model": st.session_state.get("model_id", model),
        "L1": st.session_state.get("L1_code", L1_code),
    }
    csv_data = generate_csv(
        st.session_state.results,
        L1_meta,
        delimiter=CFG_CSV_DELIM,
        line_terminator=CFG_CSV_EOL,
        include_header=st.session_state.get("csv_with_header", True),
        include_extras=True,
        anki_field_header=csv_anki_header,
        extras_meta=csv_extras,
    )

    st.session_state.last_csv_data = csv_data
    st.download_button(
        label="📥 Download anki_cards.csv",
        data=csv_data,
        file_name="anki_cards.csv",
        mime="text/csv",
        key="download_csv",
    )

    if HAS_GENANKI:
        try:
            front_html = CFG_FRONT_HTML_TEMPLATE.replace("{L1_LABEL}", L1_meta["label"])
            tags_meta = {
                "level": st.session_state.get("level", level),
                "profile": st.session_state.get("prompt_profile", profile),
                "model": st.session_state.get("model_id", model),
                "L1": st.session_state.get("L1_code", L1_code),
            }
            anki_bytes = build_anki_package(
                st.session_state.results,
                l1_label=L1_meta["label"],
                guid_policy=st.session_state.get("anki_guid_policy", "stable"),
                run_id=st.session_state.get("anki_run_id", str(int(time.time()))),
                model_id=CFG_ANKI_MODEL_ID,
                model_name=CFG_ANKI_MODEL_NAME,
                deck_id=CFG_ANKI_DECK_ID,
                deck_name=CFG_ANKI_DECK_NAME,
                front_template=front_html,
                back_template=CFG_BACK_HTML_TEMPLATE,
                css=CFG_CSS_STYLING,
                tags_meta=tags_meta,
                media_files=st.session_state.get("audio_media"),
            )
            st.session_state.last_anki_package = anki_bytes
            st.download_button(
                label="🧩 Download Anki deck (.apkg)",
                data=anki_bytes,
                file_name="dutch_cloze.apkg",
                mime="application/octet-stream",
                key="download_apkg",
            )
        except Exception as e:
            st.error(f"Failed to build .apkg: {e}")
    else:
        st.info("To enable .apkg export, add 'genanki' to requirements.txt and restart the app.")

# ----- Footer -----
st.caption(
    "Tips: 1) Better Dutch definitions on input → better examples and glosses. "
    "2) From B1, ~50% of sentences include a signal word. "
    "3) Some models (gpt-5/o3) ignore temperature and will be retried without it."
)
