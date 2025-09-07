# anki_csv_builder.py
# ------------------------------------
# Streamlit app to generate Dutch Cloze Anki cards from text input.
# - Uses OpenAI Responses API (official SDK) with strict JSON output expectations.
# - Prompt lives in prompts.py; tweak content there.
# - Configurable via config.py; sensible fallbacks if missing.
# - Includes CSV export and Anki .apkg export (via genanki).
# - Enforces cloze double-braces locally (auto-repair) to tolerate smaller models.

import os
import re
import io
import csv
import json
import time
import hashlib
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple
from openai import OpenAI

# Optional: Anki export (genanki)
try:
    import genanki  # type: ignore
    HAS_GENANKI = True
except Exception:
    HAS_GENANKI = False

# Config (import with safe fallbacks)
try:
    from config import SIGNALWORD_GROUPS as CFG_SIGNALWORD_GROUPS
except Exception:
    CFG_SIGNALWORD_GROUPS = {}

try:
    from config import (  # type: ignore
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
    )
except Exception:
    # Minimal fallback config
    CFG_DEFAULT_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3-mini"]
    CFG_PREFERRED_ORDER = {"gpt-5": 0, "gpt-5-mini": 1, "gpt-5-nano": 2, "gpt-4.1": 3, "gpt-4o": 4, "gpt-4o-mini": 5, "o3": 6, "o3-mini": 7}
    CFG_BLOCK_SUBSTRINGS = ("audio", "realtime", "embed", "embedding", "whisper", "asr", "transcribe", "speech", "tts", "moderation", "search", "vision", "vision-preview", "distill", "distilled", "batch", "preview")
    CFG_ALLOWED_PREFIXES = ("gpt-5", "gpt-4.1", "gpt-4o", "o3")
    CFG_SIGNALWORDS_B1 = ["omdat", "maar", "dus", "want", "terwijl", "daarom", "daardoor", "toch"]
    CFG_SIGNALWORDS_B2_PLUS = ["hoewel", "zodat", "doordat", "bovendien", "echter", "bijvoorbeeld", "tenzij", "ondanks", "desondanks", "daarentegen", "aangezien", "zodra", "voordat", "nadat", "enerzijds ... anderzijds", "niet alleen ... maar ook", "opdat"]
    CFG_PROMPT_PROFILES = {"strict": "Be literal and concise; avoid figurative language; keep the simplest structure that satisfies CEFR.",
                           "balanced": "Natural and clear; minor synonymy allowed if it improves fluency.",
                           "exam": "Neutral-formal register; precise; avoid colloquialisms.",
                           "creative": "Allow mild figurativeness if it keeps clarity and CEFR constraints."}
    CFG_L1_LANGS = {"RU": {"label": "RU", "name": "Russian", "csv_translation": "Перевод", "csv_gloss": "Перевод слова"},
                    "EN": {"label": "EN", "name": "English", "csv_translation": "Translation", "csv_gloss": "Word gloss"},
                    "ES": {"label": "ES", "name": "Spanish", "csv_translation": "Traducción", "csv_gloss": "Glosa"},
                    "DE": {"label": "DE", "name": "German", "csv_translation": "Übersetzung", "csv_gloss": "Kurzgloss"}}
    CFG_DEMO_WORDS = [
        {"woord": "aanraken", "def_nl": "iets met je hand of een ander deel van je lichaam voelen"},
        {"woord": "begrijpen", "def_nl": "snappen wat iets betekent of inhoudt"},
        {"woord": "gillen", "def_nl": "hard en hoog schreeuwen"},
        {"woord": "kloppen", "def_nl": "met regelmaat bonzen of tikken"},
        {"woord": "toestaan", "def_nl": "goedkeuren of laten gebeuren"},
        {"woord": "opruimen", "def_nl": "iets netjes maken door het op zijn plaats te leggen"},
    ]
    CFG_PAGE_TITLE = "Anki CSV Builder — Cloze (NL)"
    CFG_PAGE_LAYOUT = "wide"
    CFG_TMIN, CFG_TMAX, CFG_TDEF, CFG_TSTEP = 0.2, 0.8, 0.4, 0.1
    CFG_PREVIEW_LIMIT = 20
    CFG_API_DELAY = 0.0
    CFG_ANKI_MODEL_ID = 1607392319
    CFG_ANKI_DECK_ID = 1970010101
    CFG_ANKI_MODEL_NAME = "Dutch Cloze (L2/L1)"
    CFG_ANKI_DECK_NAME = "Dutch • Cloze"
    CFG_FRONT_HTML_TEMPLATE = """<div class="card-inner">{{cloze:L2_cloze}}</div>"""
    CFG_BACK_HTML_TEMPLATE = """<div class="card-inner">{{cloze:L2_cloze}}<div class="answer">{{L1_sentence}}</div></div>"""
    CFG_CSS_STYLING = ""
    CFG_CSV_DELIM = '|'
    CFG_CSV_EOL = '\n'

# Import the prompt builder
from prompts import compose_instructions_en, PROMPT_PROFILES as PR_PROMPT_PROFILES  
from core.sanitize_validate import (
    sanitize,
    normalize_cloze_braces,
    force_wrap_first_match,
    try_separable_verb_wrap,
    validate_card,
)
from core.parsing import parse_input

# Streamlit page config
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

def _build_signal_pool(level: str) -> dict[str, list[str]]:
    """
    Build per-group pool using SIGNALWORD_GROUPS up to the current level.
    Example: for B2 include A1+A2+B1+B2 entries.
    """
    pool: dict[str, list[str]] = {}
    lvls = set(_levels_up_to(level))
    for grp, by_lvl in (CFG_SIGNALWORD_GROUPS or {}).items():
        items: list[str] = []
        for lv, arr in by_lvl.items():
            if lv in lvls:
                items.extend(arr)
        if items:
            pool[grp] = items
    return pool

def _init_sig_usage():
    """Session-persistent usage counter for signal words (per run)."""
    if "sig_usage" not in st.session_state:
        st.session_state.sig_usage = {}  # word -> count
    if "sig_last" not in st.session_state:
        st.session_state.sig_last = None

def _choose_signalwords(level: str, n: int = 3, force_balance: bool = False) -> list[str]:
    """
    Pick a small shuffled set of candidates with light balancing:
    - pool includes groups up to 'level'
    - prefer words with smallest usage count
    - avoid immediate repetition (do not include last used)
    - ensure group diversity if possible
    """
    _init_sig_usage()
    pool = _build_signal_pool(level)
    if not pool:
        return []

    # Flatten with (word, group) pairs
    pairs: list[tuple[str,str]] = []
    for g, arr in pool.items():
        for w in arr:
            pairs.append((w, g))

    # Sort by usage count (ascending), then by group name to stabilize
    used = st.session_state.sig_usage
    last = st.session_state.sig_last
    pairs.sort(key=lambda wg: (used.get(wg[0], 0), wg[1], wg[0]))

    result: list[str] = []
    seen_groups: set[str] = set()

    for w, g in pairs:
        if w == last:
            continue  # avoid immediate repetition
        # If force_balance, try to spread across groups first
        if force_balance and g in seen_groups:
            continue
        result.append(w)
        seen_groups.add(g)
        if len(result) >= n:
            break

    # Fallback if we couldn't reach n because of strict spread
    if len(result) < n:
        for w, g in pairs:
            if w == last or w in result:
                continue
            result.append(w)
            if len(result) >= n:
                break

    return result

def _note_signalword_used(sentence: str, allowed: list[str]) -> None:
    """If any allowed signal word appears in the final sentence, increment its usage and remember last."""
    if not sentence or not allowed:
        return
    low = sentence.lower()
    for w in allowed:
        # rough containment; avoids strict tokenization but works well for our use
        if w.lower() in low:
            st.session_state.sig_usage[w] = st.session_state.sig_usage.get(w, 0) + 1
            st.session_state.sig_last = w
            break



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
        client = OpenAI(api_key=api_key)
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
PROMPT_PROFILES = PR_PROMPT_PROFILES

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

# ----- Demo & clear -----
DEMO_WORDS = CFG_DEMO_WORDS

if "input_data" not in st.session_state:
    st.session_state.input_data: List[Dict] = [] # type: ignore
if "results" not in st.session_state:
    st.session_state.results: List[Dict] = [] # type: ignore

col_demo, col_clear = st.columns([1, 1])
with col_demo:
    if st.button("Try demo", type="secondary"):
        st.session_state.input_data = DEMO_WORDS
        st.toast("✅ Demo set (6 words) loaded", icon="✅")
with col_clear:
    if st.button("Clear", type="secondary"):
        st.session_state.input_data = []
        st.session_state.results = []

# ----- Upload -----
uploaded_file = st.file_uploader("Upload .txt / .md", type=["txt", "md"], accept_multiple_files=False)

# ----- Parsing helpers -----


# Read upload
if uploaded_file is not None:
    try:
        file_text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        file_text = uploaded_file.read().decode("utf-16")
    st.session_state.input_data = parse_input(file_text)

# Preview input
if st.session_state.input_data:
    st.subheader("🔍 Parsed rows")
    st.dataframe(pd.DataFrame(st.session_state.input_data), use_container_width=True)
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

def _det_include_signalword(woord: str, level: str) -> bool:
    """Deterministic ~50% inclusion for B1+ by hashing word+level."""
    if level in ("B1", "B2", "C1", "C2"):
        seed = int(hashlib.sha256(f"{woord}|{level}".encode()).hexdigest(), 16) % 100
        return seed < 50
    return False

# ----- JSON extraction and validation -----
RE_CODE_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
RE_FIRST_OBJECT = re.compile(r"\{[\s\S]*\}")

REQUIRED_KEYS = {"L2_word","L2_cloze","L1_sentence","L2_collocations","L2_definition","L1_gloss"}

def _try_parse_candidate(s: str) -> Dict:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and REQUIRED_KEYS.issubset(obj.keys()):
            return obj
    except Exception:
        pass
    return {}

def _brace_scan_pick(text: str) -> Dict:
    """
    Scan text and try every {...} span with brace-balancing.
    Return the first dict that parses and has all required keys.
    """
    opens = []
    for i, ch in enumerate(text):
        if ch == "{":
            opens.append(i)
        elif ch == "}" and opens:
            start = opens.pop()
            chunk = text[start:i+1]
            obj = _try_parse_candidate(chunk)
            if obj:
                return obj
    return {}

def extract_json_block(text: str) -> Dict:
    """
    Robustly extract the JSON object from model output.
    Supports fenced code blocks, plain text with multiple objects, and balanced-brace scanning.
    """
    if not text:
        return {}

    # 1) Try fenced code block ```json ... ```
    m = RE_CODE_FENCE.search(text)
    if m:
        cand = m.group(1).strip()
        obj = _try_parse_candidate(cand)
        if obj:
            return obj
        # Some models still add prose around; try brace-scan inside fence
        obj = _brace_scan_pick(cand)
        if obj:
            return obj

    # 2) Try to find any {...} in the whole text (first pass: greedy regex then validate)
    m2 = RE_FIRST_OBJECT.search(text)
    if m2:
        cand2 = m2.group(0)
        obj = _try_parse_candidate(cand2)
        if obj:
            return obj

    # 3) Final attempt: balanced-brace scan over the full text
    obj = _brace_scan_pick(text)
    if obj:
        return obj

    return {}


# --- new helper: безопасно получить текст ответа ---
def _get_response_text(resp) -> str:
    if not resp:
        return ""
    # Try direct attribute
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # Newer SDK objects: resp.output[0].content[0].text.value
    try:
        parts = []
        for out in getattr(resp, "output", []) or []:
            for item in getattr(out, "content", []) or []:
                # gpt-5/o3 return objects with .text.value
                txt_obj = getattr(item, "text", None)
                if txt_obj and hasattr(txt_obj, "value"):
                    parts.append(txt_obj.value)
                elif isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], dict) and "value" in item["text"]:
                        parts.append(item["text"]["value"])
                    elif "text" in item:
                        parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(item.text)
        if parts:
            return "".join(parts)
    except Exception:
        pass
    return str(resp)
def _get_response_parsed(resp) -> Dict:
    """Extract parsed JSON from Responses API (gpt-5/o3 structured outputs)."""
    if not resp:
        return {}
    # Fast path (SDK convenience):
    try:
        p = getattr(resp, "output_parsed", None)
        if isinstance(p, dict) and p:
            return p
    except Exception:
        pass
    # Walk the new structure: resp.output[*].content[*].parsed
    try:
        for out in (getattr(resp, "output", None) or []):
            content = getattr(out, "content", None) or []
            for item in content:
                # object attribute
                if hasattr(item, "parsed"):
                    p = getattr(item, "parsed")
                    if isinstance(p, dict) and p:
                        return p
                # dict-ish fallback
                if isinstance(item, dict) and isinstance(item.get("parsed"), dict) and item["parsed"]:
                    return item["parsed"]
    except Exception:
        pass
    return {}


def validate_card(card: Dict) -> List[str]:
    """Field presence, cloze presence, collocations len=3, gloss len<=2, no pipes."""
    problems = []
    required = ["L2_word", "L2_cloze", "L1_sentence", "L2_collocations", "L2_definition", "L1_gloss"]
    for k in required:
        v = card.get(k, "")
        if not isinstance(v, str) or not v.strip():
            problems.append(f"Field '{k}' is empty")
        if "|" in str(v):
            problems.append(f"Field '{k}' contains '|'")
    if "{{c1::" not in card.get("L2_cloze", ""):
        problems.append("Missing {{c1::…}} in L2_cloze")
    col_raw = card.get("L2_collocations", "")
    items = [s.strip() for s in re.split(r";\s*|\n+", col_raw) if s.strip()]
    if len(items) != 3:
        problems.append("L2_collocations must contain exactly 3 items")
    if len(card.get("L1_gloss", "").split()) > 2:
        problems.append("L1_gloss must be 1–2 words")
    return problems



# ----- OpenAI call -----
def call_openai_card(client: OpenAI, row: Dict, model: str, temperature: float,
                     L1_code: str, level: str, profile: str) -> Dict:
    """Call OpenAI for one word, with strict instructions and local cloze auto-fix."""

    # Build instructions text from prompts.py
    instructions = compose_instructions_en(L1_code, level, profile)

    # Decide whether to include a signal word
    include_sig = _det_include_signalword(row.get("woord", ""), level)
    # For B2+ we enforce stronger balancing; for B1 we just suggest
    force_balance = level in ("B2","C1","C2")
    sig_list = _choose_signalwords(level, n=3, force_balance=force_balance)

    # Payload to send (input data)
    payload = {
        "L2_word": row.get("woord", "").strip(),
        "given_L2_definition": row.get("def_nl", "").strip(),
        "preferred_L1_gloss": row.get("ru_short", "").strip(),
        "L1": L1_LANGS[L1_code]["name"],
        "CEFR": level,
        "INCLUDE_SIGNALWORD": include_sig,
        "ALLOWED_SIGNALWORDS": sig_list,
    }

    # Explicit JSON template to guide model output
    json_template = (
        '{'
        '"L2_word": "<Dutch lemma>", '
        '"L2_cloze": "ONE Dutch sentence with {{c1::...}} (and {{c2::...}} only if separable)", '
        '"L1_sentence": "<exact translation into ' + L1_LANGS[L1_code]["name"] + '>", '
        '"L2_collocations": "colloc1; colloc2; colloc3", '
        '"L2_definition": "<short Dutch definition>", '
        '"L1_gloss": "<1-2 ' + L1_LANGS[L1_code]["name"] + ' words>"'
        '}'
    )
    json_schema = {
    "name": "AnkiClozeCard",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["L2_word","L2_cloze","L1_sentence","L2_collocations","L2_definition","L1_gloss"],
        "properties": {
            "L2_word":        {"type":"string","minLength":1},
            "L2_cloze":       {"type":"string","minLength":1},
            "L1_sentence":    {"type":"string","minLength":1},
            "L2_collocations":{"type":"string","minLength":1},
            "L2_definition":  {"type":"string","minLength":1},
            "L1_gloss":       {"type":"string","minLength":1},
        },
    },
}

    kwargs = dict(
        model=model,
        instructions=instructions,
        input=(
            "Input JSON:\n" + json.dumps(payload, ensure_ascii=False) +
            "\nReply with STRICT JSON ONLY. It must match this template exactly (same keys, one-line JSON):\n" +
            json_template
        ),
    )

    if limit_tokens:
        kwargs["max_output_tokens"] = 3000
    if _should_pass_temperature(model):
        kwargs["temperature"] = temperature

    # --- First request ---
    try:
        resp = client.responses.create(**kwargs)
        parsed = getattr(resp, "output_parsed", None)
        if not parsed:
            parsed = extract_json_block(_get_response_text(resp))
    except Exception as e:
        msg = str(e)
        st.error(f"OpenAI API Error: {msg}")  # Enhanced error logging
        st.error(f"Request kwargs: {kwargs}") # Log the request parameters
        try:
            st.error(f"Response text: {_get_response_text(resp)}") # Log the response text if available
        except:
            pass

        if "Unsupported parameter: 'temperature'" in msg or "param': 'temperature'" in msg:
            no_temp = st.session_state.get("no_temp_models", set())
            no_temp.add(model)
            st.session_state.no_temp_models = no_temp
            kwargs.pop("temperature", None)
            resp = client.responses.create(**kwargs)
        else:
            raise

    parsed = _get_response_parsed(resp)  # предпочитаем structured outputs
    if not parsed:
        raw_response = _get_response_text(resp)
        if display_raw_response:
            st.write(f"Raw GPT-5 Response: {raw_response}")  # Log raw response
        parsed = extract_json_block(raw_response)  # фолбэк на текст


    # Sanitize and collect fields
    card = {
        "L2_word": sanitize(parsed.get("L2_word", payload["L2_word"])),
        "L2_cloze": sanitize(parsed.get("L2_cloze", "")),
        "L1_sentence": sanitize(parsed.get("L1_sentence", "")),
        "L2_collocations": sanitize(parsed.get("L2_collocations", "")),
        "L2_definition": sanitize(parsed.get("L2_definition", payload.get("given_L2_definition", ""))),
        "L1_gloss": sanitize(parsed.get("L1_gloss", payload.get("preferred_L1_gloss", ""))),
        "L1_hint": "",  # reserved for future use
    }

    # --- Local cloze auto-fixes BEFORE validation ---
    clz = normalize_cloze_braces(card["L2_cloze"])
    if "{{c1::" not in clz:
        clz = force_wrap_first_match(card["L2_word"], clz)
    clz = try_separable_verb_wrap(card["L2_word"], clz)
    card["L2_cloze"] = clz

    # --- Validation + one repair-pass if needed ---
    problems = validate_card(card)
    if problems:
        repair_prompt = instructions + "\n\nREPAIR: The previous JSON has issues: " + "; ".join(problems) + ". " \
            + "Fix ONLY the problematic fields and return STRICT JSON again."

        repair_kwargs = dict(
            model=model,
            instructions=repair_prompt,
            input=("Previous JSON:\n" + json.dumps(card, ensure_ascii=False)),
        )
        if limit_tokens:
            repair_kwargs["max_output_tokens"] = 3000
        if _should_pass_temperature(model):
            repair_kwargs["temperature"] = temperature

        try:
            resp2 = client.responses.create(**repair_kwargs)
            parsed2 = _get_response_parsed(resp2) or extract_json_block(_get_response_text(resp2))
            if not parsed:
                raw_preview = (getattr(resp2, "output_text", "") or "").strip()
                if raw_preview:
                    st.warning("Model returned non-parsable output; showing first 300 chars for debugging.")
                    st.code(raw_preview[:300] + ("…" if len(raw_preview) > 300 else ""), language="text")
        except Exception as e:
            msg = str(e)
            st.error(f"Repair request OpenAI API Error: {msg}") # Enhanced error logging for repair
            st.error(f"Repair request kwargs: {repair_kwargs}") # Log repair request parameters
            try:
                raw_response2 = _get_response_text(resp2)
                if display_raw_response:
                    st.error(f"Repair response text: {raw_response2}") # Log repair response text
            except:
                pass
        parsed2 = extract_json_block(_get_response_text(resp2))
        if parsed2:
            card.update({
                "L2_word": sanitize(parsed2.get("L2_word", card["L2_word"])),
                "L2_cloze": sanitize(parsed2.get("L2_cloze", card["L2_cloze"])),
                "L1_sentence": sanitize(parsed2.get("L1_sentence", card["L1_sentence"])),
                "L2_collocations": sanitize(parsed2.get("L2_collocations", card["L2_collocations"])),
                "L2_definition": sanitize(parsed2.get("L2_definition", card["L2_definition"])),
                "L1_gloss": sanitize(parsed2.get("L1_gloss", card["L1_gloss"])),
            })
    
    _note_signalword_used(card.get("L2_cloze",""), sig_list)

    return card



# ----- CSV generation -----
def generate_csv(
    results: List[Dict],
    L1_code: str,
    include_header: bool = True,
    include_extras: bool = False,
    anki_field_header: bool = True,
) -> str:
    """Build CSV with configured delimiter; optional localized or Anki-field header."""
    meta = L1_LANGS[L1_code]
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, delimiter=CFG_CSV_DELIM, lineterminator=CFG_CSV_EOL)

    if include_header:
        if anki_field_header:
            header = ["L2_word","L2_cloze","L1_sentence","L2_collocations","L2_definition","L1_gloss","L1_hint"]
        else:
            header = [
                "NL-слово",
                "Предложение NL (с cloze)",
                f"{meta['csv_translation']} {meta['label']}",
                "Коллокации (NL)",
                "Определение NL",
                f"{meta['csv_gloss']} {meta['label']}",
                "Подсказка (L1)",
            ]
        if include_extras:
            header += ["CEFR", "Profile", "Model", "L1"]
        writer.writerow(header)

    for r in results:
        row = [
            r.get('L2_word',''),
            r.get('L2_cloze',''),
            r.get('L1_sentence',''),
            r.get('L2_collocations',''),
            r.get('L2_definition',''),
            r.get('L1_gloss',''),
            r.get('L1_hint',''),
        ]
        if include_extras:
            row += [
                st.session_state.get('level', ''),
                st.session_state.get('prompt_profile', ''),
                st.session_state.get('model_id', ''),
                st.session_state.get('L1_code', ''),
            ]
        writer.writerow(row)

    return csv_buffer.getvalue()


# ----- Anki .apkg export -----
def _compute_guid(c: Dict, policy: str, run_id: str) -> str:
    """Stable GUID (content-based), or unique-per-export if policy=='unique'."""
    base = f"{c.get('L2_word','')}|{c.get('L2_cloze','')}"
    if policy == "unique":
        base = base + "|" + run_id
    try:
        return genanki.guid_for(base)  # type: ignore
    except Exception:
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]

def build_anki_package(cards: List[Dict], L1_label: str, guid_policy: str, run_id: str) -> bytes:
    """Create a .apkg deck 'Dutch • Cloze' with our custom Cloze model."""
    if not HAS_GENANKI:
        raise RuntimeError("genanki is not installed. Add 'genanki' to requirements.txt and redeploy.")

    front = CFG_FRONT_HTML_TEMPLATE.replace("{L1_LABEL}", L1_label)
    back = CFG_BACK_HTML_TEMPLATE

    model = genanki.Model(
        CFG_ANKI_MODEL_ID,
        CFG_ANKI_MODEL_NAME,
        fields=[
            {"name": "L2_word"},
            {"name": "L2_cloze"},
            {"name": "L1_sentence"},
            {"name": "L2_collocations"},
            {"name": "L2_definition"},
            {"name": "L1_gloss"},
            {"name": "L1_hint"},
        ],
        templates=[{"name": "Cloze", "qfmt": front, "afmt": back}],
        css=CFG_CSS_STYLING,
        model_type=genanki.Model.CLOZE,
    )

    deck = genanki.Deck(CFG_ANKI_DECK_ID, CFG_ANKI_DECK_NAME)

    for c in cards:
        note = genanki.Note(
            model=model,
            fields=[
                c.get("L2_word", ""),
                c.get("L2_cloze", ""),
                c.get("L1_sentence", ""),
                c.get("L2_collocations", ""),
                c.get("L2_definition", ""),
                c.get("L1_gloss", ""),
                c.get("L1_hint", ""),
            ],
            guid=_compute_guid(c, guid_policy, run_id),
            tags=list({
                f"CEFR::{st.session_state.get('level','')}",
                f"profile::{st.session_state.get('prompt_profile','')}",
                f"model::{st.session_state.get('model_id', '')}",
                f"L1::{st.session_state.get('L1_code','RU')}",
            })
        )
        deck.add_note(note)

    pkg = genanki.Package(deck)
    bio = io.BytesIO()
    pkg.write_to_file(bio)
    return bio.getvalue()

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
            total = len(st.session_state.input_data)
            progress = st.progress(0)
            for idx, row in enumerate(st.session_state.input_data):
                try:
                    card = call_openai_card(
                        client, row, model=model, temperature=temperature,
                        L1_code=L1_code, level=level, profile=profile
                    )
                    st.session_state.results.append(card)
                except Exception as e:
                    st.error(f"Error for word '{row.get('woord','?')}': {e}")
                finally:
                    progress.progress(int((idx + 1) / max(total, 1) * 100))
                    if CFG_API_DELAY > 0:
                        time.sleep(CFG_API_DELAY)

# ----- Preview & downloads -----
if st.session_state.results:
    st.subheader("📋 Preview (first 20)")
    preview_df = pd.DataFrame(st.session_state.results)[:CFG_PREVIEW_LIMIT]
    st.dataframe(preview_df, use_container_width=True)

    csv_data = generate_csv(
        st.session_state.results,
        L1_code,
        include_header=st.session_state.get("csv_with_header", True),
        include_extras=True,
        anki_field_header=csv_anki_header,  
    )

    st.download_button(
        label="📥 Download anki_cards.csv",
        data=csv_data,
        file_name="anki_cards.csv",
        mime="text/csv",
    )

    if HAS_GENANKI:
        try:
            anki_bytes = build_anki_package(
                st.session_state.results,
                L1_label=L1_meta["label"],
                guid_policy=st.session_state.get("anki_guid_policy", "stable"),
                run_id=st.session_state.get("anki_run_id", str(int(time.time())))
            )
            st.download_button(
                label="🧩 Download Anki deck (.apkg)",
                data=anki_bytes,
                file_name="dutch_cloze.apkg",
                mime="application/octet-stream",
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
