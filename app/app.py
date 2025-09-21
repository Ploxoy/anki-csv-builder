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
from core.llm_clients import create_client, send_responses_request, responses_accepts_param
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


def _probe_response_format_support(client: OpenAI, model: str) -> None:
    """Quickly probe whether the selected model supports response_format.

    On failure, store the model in session cache so subsequent requests skip schema
    and avoid noisy warnings. Does nothing if already cached.
    """
    if not model or model in st.session_state.get("no_response_format_models", set()):
        return
    # If SDK does not expose the parameter at all, short-circuit
    if not responses_accepts_param(client, "text"):
        cache = set(st.session_state.get("no_response_format_models", set()))
        cache.add(model)
        st.session_state.no_response_format_models = cache
        return
    try:
        probe_schema = {
            "name": "Probe",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
            },
        }
        # Minimal instructions and input
        instructions = "Return strictly: {\"ok\": true}"
        input_text = "probe"
        # Use small token limit to reduce cost
        resp_format = probe_schema
        _, meta = send_responses_request(
            client=client,
            model=model,
            instructions=instructions,
            input_text=input_text,
            response_format=resp_format,
            max_output_tokens=64,
            temperature=None,
            retries=0,
            warn=False,
        )
        if meta.get("response_format_removed"):
            cache = set(st.session_state.get("no_response_format_models", set()))
            cache.add(model)
            st.session_state.no_response_format_models = cache
            notified = set(st.session_state.get("no_response_format_notified", set()))
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
    except Exception:
        # Silently ignore probe errors to avoid blocking generation
        pass


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

# Apply pending batch params (set before widget instantiation)
pending_bs = st.session_state.pop("batch_size_pending", None)
if pending_bs is not None:
    st.session_state["batch_size"] = int(pending_bs)
pending_workers = st.session_state.pop("max_workers_pending", None)
if pending_workers is not None:
    st.session_state["max_workers"] = int(pending_workers)
pending_auto = st.session_state.pop("auto_advance_pending", None)
if pending_auto is not None:
    st.session_state["auto_advance"] = bool(pending_auto)

# Batch processing controls (to avoid long single runs on Streamlit Cloud)
st.sidebar.subheader("Batch processing")
st.sidebar.number_input(
    "Batch size",
    min_value=1,
    max_value=50,
    value=st.session_state.get("batch_size", 5),
    step=1,
    help="How many rows to process per batch.",
    key="batch_size",
)
st.sidebar.checkbox(
    "Auto-advance batches",
    value=st.session_state.get("auto_advance", True),
    help="Continue to the next batch automatically until finished.",
    key="auto_advance",
)

# Optional parallelism within a batch
st.sidebar.slider(
    "Max workers per batch",
    min_value=1,
    max_value=8,
    value=st.session_state.get("max_workers", 3),
    step=1,
    help="Parallel requests inside a batch. Keep modest (3–4) to avoid rate limits.",
    key="max_workers",
)

# Advanced: schema handling controls
with st.sidebar.expander("Advanced (Responses schema)"):
    force_schema = st.checkbox(
        "Force JSON schema (ignore cache)",
        value=False,
        help=(
            "Attempt to send response_format=json_schema even if the model was previously marked as unsupported. "
            "If the SDK/model rejects it, we will fall back automatically."
        ),
        key="force_schema_checkbox",
    )
    # SDK capability hint
    try:
        _client_for_probe = create_client(API_KEY)
        if _client_for_probe is not None and not responses_accepts_param(_client_for_probe, "text"):
            st.caption("SDK check: Responses.create has no 'text' parameter — schema (text.format) will be disabled.")
    except Exception:
        pass
    if st.button("Reset schema support cache"):
        st.session_state.no_response_format_models = set()
        st.session_state.no_response_format_notified = set()
        st.success("Schema support cache has been reset for this session.")
    if st.button("Re-probe schema support for selected model"):
        _init_response_format_cache()
        client = create_client(API_KEY)
        if client is None:
            st.warning("OpenAI SDK not available; cannot probe.")
        else:
            _probe_response_format_support(client, model)
            st.info("Probe completed. Check debug panel or try generation.")

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


def _toast(message: str, *, icon: str | None = None, variant: str = "info") -> None:
    """Use st.toast when available; otherwise fall back to standard status blocks."""
    toast_fn = getattr(st, "toast", None)
    if callable(toast_fn):  # Streamlit ≥1.25
        kwargs: Dict[str, str] = {}
        if icon:
            kwargs["icon"] = icon
        toast_fn(message, **kwargs)
        return

    # Fallback for older Streamlit versions without st.toast
    fallback_msg = f"{icon} {message}" if icon else message
    if variant == "success":
        st.success(fallback_msg)
    elif variant == "warning":
        st.warning(fallback_msg)
    else:
        st.info(fallback_msg)

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

def _recommend_batch_params(total: int) -> tuple[int, int]:
    """Return (batch_size, max_workers) based on dataset size.

    Heuristic:
    - Target ~20 items per batch, up to 8 batches total for big lists.
    - Allow up to 10 parallel workers (as per empirical limit), but not more than batch size.
    """
    import math
    if total <= 0:
        return (5, 3)
    if total <= 10:
        return (total, min(10, total))
    # Aim for ~20 per batch, clamp number of batches to <=8
    target_batches = max(2, min(8, math.ceil(total / 20)))
    bs = max(1, min(total, math.ceil(total / target_batches)))
    workers = min(10, max(2, min(bs, 10)))
    return (bs, workers)


def _compute_list_token(rows: list[dict]) -> str:
    try:
        first = (rows[0].get("woord", "") if rows else "").strip()
        last = (rows[-1].get("woord", "") if rows else "").strip()
        return f"len={len(rows)}|first={first}|last={last}"
    except Exception:
        return f"len={len(rows)}"


def _apply_recommended_batch_params(total: int, *, token: str | None = None) -> None:
    # Do not re-apply if already applied for this exact dataset token
    if token and st.session_state.get("recommend_applied_token") == token:
        return
    bs, workers = _recommend_batch_params(total)
    # Defer actual widget state update to the next rerun (before widget creation)
    st.session_state["batch_size_pending"] = int(bs)
    st.session_state["max_workers_pending"] = int(workers)
    if total > 1:
        st.session_state["auto_advance_pending"] = True
    if token:
        st.session_state["recommend_applied_token"] = token
    _toast(f"Recommended batch: size {bs}, workers {workers}", icon="⚙️")
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

col_demo, col_clear = st.columns([1, 1])
with col_demo:
    if st.button("Try demo", type="secondary"):
        st.session_state.input_data = [dict(row) for row in DEMO_WORDS]
        st.session_state.manual_rows = [
            {"woord": row.get("woord", ""), "def_nl": row.get("def_nl", ""), "translation": row.get("translation", "") or ""}
            for row in DEMO_WORDS
        ]
        st.session_state.upload_token = None
        _apply_recommended_batch_params(
            len(st.session_state.input_data),
            token=_compute_list_token(st.session_state.input_data),
        )
        _toast("Demo set (6 words) loaded", icon="✅", variant="success")
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
        upload_token = f"{getattr(uploaded_file,'name','')}|{getattr(uploaded_file,'size','')}"
        if st.session_state.get("upload_token") != upload_token:
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
            st.session_state.upload_token = upload_token
            _apply_recommended_batch_params(
                len(st.session_state.input_data),
                token=_compute_list_token(st.session_state.input_data),
            )
            _toast("Input replaced with uploaded file", icon="📄")

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
        width="stretch",
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
                st.session_state.upload_token = None
                _apply_recommended_batch_params(
                    len(manual_clean),
                    token=_compute_list_token(manual_clean),
                )
                _toast(f"Loaded {len(manual_clean)} manual rows", icon="✍️", variant="success")
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
    st.dataframe(preview_in[cols], width="stretch")
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
# ----- Generate section (batch mode) -----
if st.session_state.input_data:
    # Persistent run state
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "run_active" not in st.session_state:
        st.session_state.run_active = False
    if "auto_continue" not in st.session_state:
        st.session_state.auto_continue = False

    total = len(st.session_state.input_data)
    processed = len(st.session_state.get("results", []))
    # Run summary (across batches)
    run_stats = st.session_state.get("run_stats") or {
        "batches": 0,
        "items": 0,
        "elapsed": 0.0,
        "errors": 0,
        "transient": 0,
        "start_ts": None,
    }
    st.session_state.run_stats = run_stats
    summary = st.empty()
    if run_stats["start_ts"]:
        total_elapsed = max(0.001, time.time() - run_stats["start_ts"])
        rate = (run_stats["items"]) / total_elapsed
        valid_now = sum(1 for c in st.session_state.get("results", []) if not c.get("error"))
        summary.caption(
            f"Run: batches {run_stats['batches']} • processed {processed}/{total} • valid {valid_now} • "
            f"elapsed {total_elapsed:.1f}s • {rate:.2f}/s • errors {run_stats['errors']} (transient {run_stats['transient']})"
        )
    else:
        valid_now = sum(1 for c in st.session_state.get("results", []) if not c.get("error"))
        summary.caption(f"Run: processed {processed}/{total} • valid {valid_now}")

    overall_caption = st.empty()
    overall = st.progress(0)
    overall.progress(min(1.0, processed / max(total, 1)))
    overall_caption.caption(f"Overall: {processed}/{total} processed")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    start_run = c1.button("Start run", type="primary")
    next_batch = c2.button("Next batch")
    stop_run = c3.button("Stop run")
    rerun_errored = c4.button("Re‑run errored only")

    def _process_batch() -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        client = OpenAI(api_key=API_KEY)
        max_tokens = 3000 if limit_tokens else None
        effective_temp = temperature if _should_pass_temperature(model) else None
        _init_sig_usage()
        _init_response_format_cache()
        if not st.session_state.get("anki_run_id"):
            st.session_state.anki_run_id = str(int(time.time()))
        st.session_state.model_id = model

        start = int(st.session_state.current_index or 0)
        end = min(start + int(st.session_state.get("batch_size", 5)), total)
        if start >= total:
            return
        indices = list(range(start, end))
        input_snapshot = list(st.session_state.input_data)

        # Snapshot of params and state to pass into workers
        no_rf_models = set(st.session_state.get("no_response_format_models", set()))
        force_schema = st.session_state.get("force_schema_checkbox", False)
        force_flagged = st.session_state.get("force_flagged", False)

        def _make_settings(seed: int) -> GenerationSettings:
            return GenerationSettings(
                model=model,
                L1_code=L1_code,
                L1_name=L1_meta["name"],
                level=level,
                profile=profile,
                temperature=effective_temp,
                max_output_tokens=max_tokens,
                allow_response_format=(model not in no_rf_models or force_schema),
                signalword_seed=seed,
            )

        def _worker(idx: int, row: dict) -> tuple[int, dict]:
            if not force_flagged and not row.get("_flag_ok", True):
                return idx, {
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
                    "meta": {"flag_reason": row.get("_flag_reason", ""), "input_index": idx},
                }
            try:
                seed = random.randint(0, 2**31 - 1)
                settings = _make_settings(seed)
                gen_result = generate_card(
                    client=client,
                    row=row,
                    settings=settings,
                    signalword_groups=CFG_SIGNALWORD_GROUPS,
                    signalwords_b1=CFG_SIGNALWORDS_B1,
                    signalwords_b2_plus=CFG_SIGNALWORDS_B2_PLUS,
                    signal_usage=None,
                    signal_last=None,
                )
                card = gen_result.card
                meta = card.get("meta", {}) or {}
                meta["input_index"] = idx
                card["meta"] = meta
                return idx, card
            except Exception as e:  # pragma: no cover
                return idx, {
                    "L2_word": row.get("woord", ""),
                    "L2_cloze": "",
                    "L1_sentence": "",
                    "L2_collocations": "",
                    "L2_definition": row.get("def_nl", ""),
                    "L1_gloss": row.get("translation", ""),
                    "L1_hint": "",
                    "AudioSentence": "",
                    "AudioWord": "",
                    "error": f"exception: {e}",
                    "meta": {"input_index": idx},
                }

        workers = int(st.session_state.get("max_workers", 3))
        batch_header = st.empty()
        batch_header.caption(
            f"Batch {start+1}–{end} of {total} • size {len(indices)} • workers {workers}"
        )
        batch_prog = st.progress(0)
        batch_status = st.empty()
        batch_start_ts = time.time()
        results_map: dict[int, dict] = {}
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, idx, input_snapshot[idx]): idx for idx in indices}
            for fut in as_completed(futs):
                idx, card = fut.result()
                results_map[idx] = card
                completed += 1
                batch_prog.progress(min(1.0, completed / max(len(indices), 1)))
                # Status line with approximate active/queued and speed
                elapsed = max(0.001, time.time() - batch_start_ts)
                active = max(0, min(workers, len(indices) - completed))
                queued = max(0, len(indices) - completed - active)
                rate = completed / elapsed
                batch_status.caption(
                    f"Done {completed}/{len(indices)} • Active ~{active} • Queued ~{queued} • {elapsed:.1f}s • {rate:.2f}/s"
                )
                # Update overall progress early (tasks processed, not only saved cards)
                done_tasks = (start + completed)
                overall.progress(min(1.0, done_tasks / max(total, 1)))
                overall_caption.caption(f"Overall: {done_tasks}/{total} processed")
                if CFG_API_DELAY > 0:
                    time.sleep(CFG_API_DELAY)

        # Merge in input order and update signalword usage sequentially
        usage = dict(st.session_state.get("sig_usage", {}))
        last = st.session_state.get("sig_last")
        batch_errors = 0
        batch_transient = 0
        for idx in indices:
            card = results_map.get(idx)
            if card is None:
                continue
            st.session_state.results.append(card)
            meta = card.get("meta", {}) or {}
            # Handle schema unsupported notification (once per model)
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
                        f"Model {model} ignored schema (text.format); falling back to text parsing for this session."
                    )
                    if detail:
                        message += f"\nReason: {detail}"
                    st.info(message, icon="ℹ️")
            found = meta.get("signalword_found")
            if found:
                usage[found] = usage.get(found, 0) + 1
                last = found
            err_text = (card.get("error") or "").lower()
            if err_text:
                batch_errors += 1
                if any(code in err_text for code in ("429", "rate", "timeout", "502", "503")):
                    batch_transient += 1

        st.session_state.sig_usage = usage
        st.session_state.sig_last = last
        st.session_state.current_index = end
        overall_count = len(st.session_state.results)
        overall.progress(min(1.0, overall_count / max(total, 1)))
        overall_caption.caption(f"Overall: {overall_count}/{total} processed")
        batch_elapsed = max(0.001, time.time() - batch_start_ts)
        batch_status.caption(f"Batch finished in {batch_elapsed:.1f}s • {len(indices)/batch_elapsed:.2f}/s")

        # Update run summary
        if not st.session_state.run_stats.get("start_ts"):
            st.session_state.run_stats["start_ts"] = batch_start_ts
        st.session_state.run_stats["batches"] += 1
        st.session_state.run_stats["items"] += len(indices)
        st.session_state.run_stats["elapsed"] += batch_elapsed
        st.session_state.run_stats["errors"] += batch_errors
        st.session_state.run_stats["transient"] += batch_transient
        # Refresh top summary line
        total_elapsed = max(0.001, time.time() - st.session_state.run_stats["start_ts"])
        rate = (st.session_state.run_stats["items"]) / total_elapsed
        summary.caption(
            f"Run: batches {st.session_state.run_stats['batches']} • processed {overall_count}/{total} • "
            f"elapsed {total_elapsed:.1f}s • {rate:.2f}/s • errors {st.session_state.run_stats['errors']} "
            f"(transient {st.session_state.run_stats['transient']})"
        )

        # Auto-adapt workers for next batch on transient spikes
        if batch_transient >= 2 and st.session_state.get("max_workers", 3) > 1:
            st.session_state.max_workers = int(st.session_state.get("max_workers", 3)) - 1
            st.info(
                f"Transient errors detected ({batch_transient}); reducing max workers to {st.session_state.max_workers} for next batch.",
                icon="⚠️",
            )

    def _rerun_errored_only() -> None:
        # Collect indices of errored cards
        err_indices = []
        for card in st.session_state.get("results", []):
            if card.get("error"):
                meta = card.get("meta", {}) or {}
                idx = meta.get("input_index")
                if isinstance(idx, int):
                    err_indices.append(idx)
        err_indices = sorted(set(err_indices))
        if not err_indices:
            st.info("No errored cards to re-run.")
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed
        client = OpenAI(api_key=API_KEY)
        max_tokens = 3000 if limit_tokens else None
        effective_temp = temperature if _should_pass_temperature(model) else None
        _init_response_format_cache()
        no_rf_models = set(st.session_state.get("no_response_format_models", set()))
        force_schema = st.session_state.get("force_schema_checkbox", False)
        force_flagged = st.session_state.get("force_flagged", False)
        input_snapshot = list(st.session_state.input_data)

        def _make_settings(seed: int) -> GenerationSettings:
            return GenerationSettings(
                model=model,
                L1_code=L1_code,
                L1_name=L1_meta["name"],
                level=level,
                profile=profile,
                temperature=effective_temp,
                max_output_tokens=max_tokens,
                allow_response_format=(model not in no_rf_models or force_schema),
                signalword_seed=seed,
            )

        def _worker(idx: int, row: dict) -> tuple[int, dict]:
            try:
                if not force_flagged and not row.get("_flag_ok", True):
                    return idx, {"error": "flagged_precheck", "meta": {"input_index": idx},
                                 "L2_word": row.get("woord", ""), "L2_cloze": "", "L1_sentence": "",
                                 "L2_collocations": "", "L2_definition": row.get("def_nl", ""),
                                 "L1_gloss": row.get("translation", ""), "L1_hint": "",
                                 "AudioSentence": "", "AudioWord": ""}
                seed = random.randint(0, 2**31 - 1)
                settings = _make_settings(seed)
                gen_result = generate_card(
                    client=client,
                    row=row,
                    settings=settings,
                    signalword_groups=CFG_SIGNALWORD_GROUPS,
                    signalwords_b1=CFG_SIGNALWORDS_B1,
                    signalwords_b2_plus=CFG_SIGNALWORDS_B2_PLUS,
                    signal_usage=None,
                    signal_last=None,
                )
                card = gen_result.card
                meta = card.get("meta", {}) or {}
                meta["input_index"] = idx
                card["meta"] = meta
                return idx, card
            except Exception as e:
                return idx, {"error": f"exception: {e}", "meta": {"input_index": idx},
                             "L2_word": row.get("woord", ""), "L2_cloze": "", "L1_sentence": "",
                             "L2_collocations": "", "L2_definition": row.get("def_nl", ""),
                             "L1_gloss": row.get("translation", ""), "L1_hint": "",
                             "AudioSentence": "", "AudioWord": ""}

        st.info(f"Re‑running {len(err_indices)} errored items…")
        bar = st.progress(0)
        results_map: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=int(st.session_state.get("max_workers", 3))) as ex:
            futs = {ex.submit(_worker, idx, input_snapshot[idx]): idx for idx in err_indices}
            done = 0
            for fut in as_completed(futs):
                idx, card = fut.result()
                results_map[idx] = card
                done += 1
                bar.progress(min(1.0, done / max(len(err_indices), 1)))

        # Replace in existing results by input_index
        new_results: list[dict] = []
        for card in st.session_state.get("results", []):
            meta = card.get("meta", {}) or {}
            idx = meta.get("input_index")
            if isinstance(idx, int) and idx in results_map:
                new_results.append(results_map[idx])
            else:
                new_results.append(card)
        st.session_state.results = new_results

        # Recompute usage/last (sequential scan)
        usage: dict[str, int] = {}
        last = None
        for card in st.session_state.results:
            meta = card.get("meta", {}) or {}
            found = meta.get("signalword_found")
            if found:
                usage[found] = usage.get(found, 0) + 1
                last = found
        st.session_state.sig_usage = usage
        st.session_state.sig_last = last
        st.success("Errored items re‑run completed.")

    # Controls logic
    if start_run:
        if not API_KEY:
            st.error("Provide OPENAI_API_KEY via Secrets, environment variable, or the input field.")
        else:
            st.session_state.results = []
            st.session_state.audio_media = {}
            st.session_state.audio_summary = None
            st.session_state.current_index = 0
            st.session_state.run_stats = {"batches": 0, "items": 0, "elapsed": 0.0, "errors": 0, "transient": 0, "start_ts": None}
            st.session_state.run_active = True
            _probe_response_format_support(create_client(API_KEY), model)
            _process_batch()
            if st.session_state.get("auto_advance") and st.session_state.current_index < total:
                st.session_state.auto_continue = True
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

    elif next_batch:
        if not st.session_state.get("run_active"):
            st.session_state.run_active = True
        _process_batch()
        if st.session_state.get("auto_advance") and st.session_state.current_index < total:
            st.session_state.auto_continue = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    elif stop_run:
        st.session_state.run_active = False
        st.session_state.auto_continue = False

    elif st.session_state.get("run_active") and st.session_state.get("auto_advance") and st.session_state.get("auto_continue"):
        # Auto-advance path: process next batch automatically
        _process_batch()
        if st.session_state.current_index < total:
            st.session_state.auto_continue = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        else:
            st.session_state.run_active = False
            st.session_state.auto_continue = False

    if rerun_errored:
        _rerun_errored_only()

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
        st.subheader("📋 Preview (all)")
        # Counters and filters
        total_rows = len(st.session_state.results)
        total_errors = sum(1 for c in st.session_state.results if c.get("error"))
        total_valid = total_rows - total_errors
        st.caption(f"Preview: valid {total_valid} • errors {total_errors} • rows {total_rows}")
        filt_col1, filt_col2 = st.columns([1, 1])
        with filt_col1:
            st.checkbox(
                "Show only errors",
                value=st.session_state.get("preview_only_errors", False),
                key="preview_only_errors",
            )
        with filt_col2:
            next_err_click = st.button("Next error")

        # Navigate to next error (show focused card below controls)
        if next_err_click and total_errors:
            err_indices = [i for i, c in enumerate(st.session_state.results) if c.get("error")]
            ptr = int(st.session_state.get("err_ptr", -1))
            candidates = [i for i in err_indices if i > ptr]
            target = candidates[0] if candidates else err_indices[0]
            st.session_state.err_ptr = target
            target_card = st.session_state.results[target]
            st.warning(
                f"Next error at row {target+1}/{total_rows}: "
                f"{target_card.get('L2_word','')} — {target_card.get('error','')}"
            )
            def _style_errors_df(df):
                def _row_style(row):
                    has_err = bool(str(row.get("error", "")).strip())
                    color = "background-color: #ffe6e6" if has_err else ""
                    return [color for _ in row.index]
                return df.style.apply(_row_style, axis=1)
            st.dataframe(_style_errors_df(pd.DataFrame([target_card])), width="stretch")

        # Build dataframe for main preview (optionally filtered)
        source_rows = (
            [c for c in st.session_state.results if c.get("error")] if st.session_state.get("preview_only_errors") else st.session_state.results
        )
        preview_df = pd.DataFrame(source_rows)
        if not preview_df.empty:
            if "error" not in preview_df.columns:
                preview_df["error"] = ""
            # Extract error_stage from nested meta
            stages: list[str | None] = []
            for row in source_rows:
                m = row.get("meta") if isinstance(row, dict) else None
                stages.append((m or {}).get("error_stage") if isinstance(m, dict) else None)
            preview_df["error_stage"] = stages
        def _style_errors_df(df):
            def _row_style(row):
                has_err = bool(str(row.get("error", "")).strip())
                color = "background-color: #ffe6e6" if has_err else ""
                return [color for _ in row.index]
            return df.style.apply(_row_style, axis=1)
        st.dataframe(_style_errors_df(preview_df) if not preview_df.empty else preview_df, width="stretch")

    csv_extras = {
        "level": st.session_state.get("level", level),
        "profile": st.session_state.get("prompt_profile", profile),
        "model": st.session_state.get("model_id", model),
        "L1": st.session_state.get("L1_code", L1_code),
    }
    include_errored = st.sidebar.checkbox("Include errored cards in exports", value=False)
    export_cards = st.session_state.results if include_errored else [c for c in st.session_state.results if not c.get("error")]
    exportable_count = len(export_cards)
    saved_count = len(st.session_state.results)
    st.caption(
        f"Exportable cards: {exportable_count} / saved results: {saved_count} / total input: {len(st.session_state.input_data)}"
    )

    csv_data = generate_csv(
        export_cards,
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
                export_cards,
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

    # Debug: show last request parameters captured by generation
    last_meta = (st.session_state.results or [{}])[-1].get("meta", {}) if st.session_state.results else {}
    req_dbg = last_meta.get("request") if isinstance(last_meta, dict) else None
    with st.expander("🐞 Debug: last model request", expanded=False):
        if req_dbg:
            st.json(req_dbg)
            # Extra flags
            st.caption(
                f"response_format_removed={last_meta.get('response_format_removed')} | "
                f"temperature_removed={last_meta.get('temperature_removed')}"
            )
            # SDK version
            try:
                import openai as _openai  # type: ignore

                st.caption(f"openai SDK version: {_openai.__version__}")
            except Exception:
                pass
        else:
            st.caption("No recent request captured yet.")

# ----- Footer -----
st.caption(
    "Tips: 1) Better Dutch definitions on input → better examples and glosses. "
    "2) From B1, ~50% of sentences include a signal word. "
    "3) Some models (gpt-5/o3) ignore temperature and will be retried without it."
)
