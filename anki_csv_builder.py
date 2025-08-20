import streamlit as st
import pandas as pd
import re
import csv
import io
import json
import time
import hashlib
from typing import List, Dict, Tuple
from openai import OpenAI
from prompts import compose_instructions_en, PROMPT_PROFILES as PR_PROMPT_PROFILES

# ==========================
# Optional: Anki export (genanki)
# ==========================
try:
    import genanki  # type: ignore
    HAS_GENANKI = True
except Exception:
    HAS_GENANKI = False

# ==========================
# Config (import with safe fallbacks)
# ==========================
try:
    # Ваш файл с параметрами. Можно свободно редактировать списки/локализацию здесь.
    from config import (
        DEFAULT_MODELS as CFG_DEFAULT_MODELS,
        _PREFERRED_ORDER as CFG_PREFERRED_ORDER,
        _BLOCK_SUBSTRINGS as CFG_BLOCK_SUBSTRINGS,
        _ALLOWED_PREFIXES as CFG_ALLOWED_PREFIXES,
        SIGNALWORDS_B1 as CFG_SIGNALWORDS_B1,
        SIGNALWORDS_B2_PLUS as CFG_SIGNALWORDS_B2_PLUS,
        PROMPT_PROFILES as CFG_PROMPT_PROFILES,
        L1_LANGS as CFG_L1_LANGS,
        CSV_HEADERS_LOCALIZATION as CFG_CSV_HEADERS_LOCALIZATION,
        CSV_HEADERS_FIXED as CFG_CSV_HEADERS_FIXED,
        PAGE_TITLE as CFG_PAGE_TITLE,
        PAGE_LAYOUT as CFG_PAGE_LAYOUT,
        TEMPERATURE_MIN as CFG_TMIN,
        TEMPERATURE_MAX as CFG_TMAX,
        TEMPERATURE_DEFAULT as CFG_TDEF,
        TEMPERATURE_STEP as CFG_TSTEP,
        PREVIEW_LIMIT as CFG_PREVIEW_LIMIT,
        API_REQUEST_DELAY as CFG_API_DELAY,
        FRONT_HTML_TEMPLATE as CFG_FRONT_HTML_TEMPLATE,
        BACK_HTML_TEMPLATE as CFG_BACK_HTML_TEMPLATE,
        CSS_STYLING as CFG_CSS_STYLING,
        MESSAGES as CFG_MESSAGES,
        ANKI_MODEL_ID as CFG_ANKI_MODEL_ID,
        ANKI_DECK_ID as CFG_ANKI_DECK_ID,
        ANKI_MODEL_NAME as CFG_ANKI_MODEL_NAME,
        ANKI_DECK_NAME as CFG_ANKI_DECK_NAME,
        DEMO_WORDS as CFG_DEMO_WORDS,
    )
except Exception:
    # Fallback значения, если config.py временно отсутствует/неполный
    CFG_DEFAULT_MODELS = [
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3-mini"
    ]
    CFG_PREFERRED_ORDER = {
        "gpt-5": 0, "gpt-5-mini": 1, "gpt-5-nano": 2, "gpt-4.1": 3, "gpt-4o": 4, "gpt-4o-mini": 5, "o3": 6, "o3-mini": 7
    }
    CFG_BLOCK_SUBSTRINGS = (
        "audio", "realtime", "embed", "embedding", "whisper", "asr", "transcribe",
        "speech", "tts", "moderation", "search", "vision", "vision-preview", "distill", "distilled",
        "batch", "preview"
    )
    CFG_ALLOWED_PREFIXES = ("gpt-5", "gpt-4.1", "gpt-4o", "o3")
    CFG_SIGNALWORDS_B1 = ["omdat", "maar", "dus", "want", "terwijl", "daarom", "daardoor", "toch"]
    CFG_SIGNALWORDS_B2_PLUS = [
        "hoewel", "zodat", "doordat", "bovendien", "echter", "bijvoorbeeld", "tenzij", "ondanks",
        "desondanks", "daarentegen", "aangezien", "zodra", "voordat", "nadat", "enerzijds ... anderzijds",
        "niet alleen ... maar ook", "opdat"
    ]
    CFG_PROMPT_PROFILES = {
        "strict": "Be literal and concise; avoid figurative language; keep the simplest structure that satisfies CEFR.",
        "balanced": "Natural and clear; minor synonymy allowed if it improves fluency.",
        "exam": "Neutral-formal register; precise; avoid colloquialisms.",
        "creative": "Allow mild figurativeness if it keeps clarity and CEFR constraints.",
    }
    CFG_L1_LANGS = {
        "RU": {"label": "RU", "name": "Russian", "csv_translation": "Перевод", "csv_gloss": "Перевод слова"},
        "EN": {"label": "EN", "name": "English", "csv_translation": "Translation", "csv_gloss": "Word gloss"},
        "ES": {"label": "ES", "name": "Spanish", "csv_translation": "Traducción", "csv_gloss": "Glosa"},
        "DE": {"label": "DE", "name": "German", "csv_translation": "Übersetzung", "csv_gloss": "Kurzgloss"},
    }
    CFG_CSV_HEADERS_LOCALIZATION = CFG_L1_LANGS
    CFG_PAGE_TITLE = "Anki CSV Builder — Cloze (NL)"
    CFG_PAGE_LAYOUT = "wide"
    CFG_TMIN, CFG_TMAX, CFG_TDEF, CFG_TSTEP = 0.2, 0.8, 0.4, 0.1
    CFG_PREVIEW_LIMIT = 20
    CFG_API_DELAY = 0.0
    # Fallback для шаблонов, если нет config.py
    CFG_FRONT_HTML_TEMPLATE = """
<div class="card-inner">
  {{cloze:L2_cloze}}
  <div class="hints">
    {{#L1_gloss}}
    <details class="hint">
      <summary>{L1_LABEL}</summary>
      <div class="hint-body">{{L1_gloss}}</div>
    </details>
    {{/L1_gloss}}

    {{#L2_definition}}
    <details class="hint">
      <summary>NL</summary>
      <div class="hint-body">{{L2_definition}}</div>
    </details>
    {{/L2_definition}}
  </div>
</div>
""".strip()
    CFG_BACK_HTML_TEMPLATE = """
<div class="card-inner">
  {{cloze:L2_cloze}}
  <div class="answer">
    {{#L1_sentence}}
    <div class="section ru">{{L1_sentence}}</div>
    {{/L1_sentence}}

    {{#L2_collocations}}
    <div class="section">
      <ul class="colloc" id="colloc-list"></ul>
      <script id="colloc-raw" type="text/plain">{{L2_collocations}}</script>
      <script>
        (function () {
          var rawEl = document.getElementById('colloc-raw');
          if (!rawEl) return;
          var raw = rawEl.textContent || "";
          var items = raw.split(/;\s*|\n+/).map(function (s) { return s.trim(); }).filter(Boolean);
          var ul = document.getElementById('colloc-list');
          if (!ul) return;
          for (var i = 0; i < items.length; i++) {
            var li = document.createElement('li');
            li.textContent = items[i];
            ul.appendChild(li);
          }
        })();
      </script>
    </div>
    {{/L2_collocations}}

    {{#L2_definition}}
    <div class="section def">{{L2_definition}}</div>
    {{/L2_definition}}

    {{#L2_word}}
    <div class="section lemma">
      <span class="lemma-nl">{{L2_word}}</span> — <span class="lemma-ru">{{L1_gloss}}</span>
    </div>
    {{/L2_word}}
  </div>
</div>
""".strip()
    CFG_CSS_STYLING = """
/* ===== масштабирование и верстка ===== */
:root{
  --fs-base: clamp(18px, 1.2vw + 1.1vh, 28px);
  --fs-sm: calc(var(--fs-base) * .9);
  --fs-lg: calc(var(--fs-base) * 1.12);
  --hl-col:#1976d2;
  --hl-bg:rgba(25,118,210,.14);
}
html, body { height:100%; }
.card{ font-size: var(--fs-base); line-height: 1.55; margin:0; min-height:100vh; display:flex; justify-content:center; align-items:flex-start; background: transparent; }
.card-inner{ width: min(92vw, 80ch); padding: 2.5vh 3vw; }
.answer { margin-top:.75em; }
.section + .section { margin-top:.55em; padding-top:.45em; border-top:1px solid rgba(0,0,0,.14); }
@media (prefers-color-scheme: dark){ .section + .section { border-top-color: rgba(255,255,255,.22); } }
.ru { font-weight:600; font-size: var(--fs-lg); }
.def { font-style: italic; opacity:.9; font-size: var(--fs-sm); }
.lemma { font-weight:600; }
.lemma-nl{ color:var(--hl-col); font-variant: small-caps; letter-spacing:.02em; }
.lemma-ru{ opacity:.9; }
.colloc{ margin:.1em 0 0 1.1em; padding:0; }
.colloc li{ margin:.12em 0; }
.cloze{ color:var(--hl-col); font-weight:700; }
mark.hl{ background:var(--hl-bg); color:inherit; padding:0 .12em; border-radius:.18em; }
.def-hint { margin-top:.5em; }
.def-hint b { opacity:.8; margin-right:.35em; }
.def-toggle{ list-style:none; cursor:pointer; display:inline-block; }
.def-toggle::-webkit-details-marker{ display:none; }
.def-toggle::before{ content: attr(data-closed); text-decoration: underline dotted; }
.def-details[open] .def-toggle::before{ content: attr(data-open); text-decoration:none; opacity:.75; }
img{ max-width:100%; height:auto; }
@media (max-width: 420px){ .card-inner{ width: 94vw; padding: 2vh 3vw; } }
.hints{ margin-top:.6em; display:flex; gap:1em 1.2em; flex-wrap:wrap; align-items:flex-start; }
.hint summary{ cursor:pointer; text-decoration: underline dotted; list-style:none; display:inline-block; }
.hint summary::-webkit-details-marker{ display:none; }
.hint[open] summary{ opacity:.75; text-decoration:none; }
.hint-body{ margin-top:.25em; font-size: var(--fs-sm); }
""".strip()
    # Fallback: фиксированные заголовки CSV (NL-колонки)
    CFG_CSV_HEADERS_FIXED = {
        "nl_word": "NL-слово",
        "nl_sentence_cloze": "Предложение NL (с cloze)",
        "collocations_nl": "Коллокации (NL)",
        "definition_nl": "Определение NL",
    }
    # Fallback: тексты UI/сообщений
    CFG_MESSAGES = {
        "app_title": "📘 Anki CSV/Anki Builder — Dutch Cloze Cards",
        "sidebar_api_header": "🔐 API Settings",
        "api_key_label": "OpenAI API Key",
        "model_label": "Model",
        "model_help": "Лучшее качество — gpt-5; баланс — gpt-4.1; быстрее — gpt-4o / gpt-5-mini.",
        "profile_label": "Prompt profile",
        "cefr_label": "CEFR",
        "l1_label": "Your language (L1)",
        "temp_label": "Temperature",
        "csv_header_checkbox": "CSV: включить строку заголовка",
        "csv_header_help": "Снимите галочку, если Anki импортирует первую строку как запись.",
        "anki_guid_policy_label": "Anki GUID policy",
        "anki_guid_policy_options": [
            "stable (update/skip existing)",
            "unique per export (import as new)"
        ],
        "anki_guid_policy_help": (
            "stable: те же заметки распознаются как уже существующие/обновляемые\n"
            "unique: каждый экспорт получает новый GUID — Anki считает их новыми заметками."
        ),
        "uploader_label": "Upload .txt / .md",
        "recognized_rows_title": "🔍 Распознанные строки",
        "upload_hint": "Загрузите файл или нажмите **Try demo**",
        "try_demo_button": "Try demo",
        "clear_button": "Очистить",
        "generate_button": "Сгенерировать карточки",
        "no_api_key": "Укажи OPENAI_API_KEY в Secrets или в поле слева.",
        "preview_title_fmt": "📋 Предпросмотр карточек (первые {limit})",
        "csv_download_label": "📥 Скачать anki_cards.csv",
        "apkg_download_label": "🧩 Скачать колоду Anki (.apkg)",
        "apkg_install_hint": "Для экспорта в .apkg добавь в requirements.txt строку 'genanki' и перезагрузи приложение.",
        "error_card_processing_fmt": "Ошибка при обработке слова '{woord}': {error}",
        "error_apkg_build_fmt": "Не удалось собрать .апkg: {error}",
        "demo_loaded": "🔁 Демо-набор из 6 слов подставлен",
        "footer_tips": (
            "Лайфхаки: 1) Чем лучше NL-дефиниции на входе, тем точнее пример и глосс. "
            "2) На уровнях B1+ примерно половина предложений будет со signaalwoorden. "
            "3) Для некоторых моделей (gpt-5/o3) температура не поддерживается и будет игнорироваться."
        ),
    }
    # Fallback: идентификаторы/имена Anki
    CFG_ANKI_MODEL_ID = 1607392319
    CFG_ANKI_DECK_ID = 1970010101
    CFG_ANKI_MODEL_NAME = "Dutch Cloze (L2/L1)"
    CFG_ANKI_DECK_NAME = "Dutch • Cloze"
    # Fallback: демо-данные
    CFG_DEMO_WORDS = [
        {"woord": "aanraken", "def_nl": "iets met je hand of een ander deel van je lichaam voelen"},
        {"woord": "begrijpen", "def_nl": "snappen wat iets betekent of inhoudt"},
        {"woord": "gillen", "def_nl": "hard en hoog schreeuwen"},
        {"woord": "kloppen", "def_nl": "met regelmaat bonzen of tikken"},
        {"woord": "toestaan", "def_nl": "goedkeuren of laten gebeuren"},
        {"woord": "opruimen", "def_nl": "iets netjes maken door het op zijn plaats te leggen"},
    ]

# ==========================
# Streamlit page config
# ==========================
st.set_page_config(page_title=CFG_PAGE_TITLE, layout=CFG_PAGE_LAYOUT)

# ==========================
# Models: defaults + dynamic fetch
# ==========================
DEFAULT_MODELS: List[str] = CFG_DEFAULT_MODELS
_PREFERRED_ORDER = CFG_PREFERRED_ORDER
_BLOCK_SUBSTRINGS = CFG_BLOCK_SUBSTRINGS
_ALLOWED_PREFIXES = CFG_ALLOWED_PREFIXES


def _sort_key(model_id: str) -> tuple:
    for k, rank in _PREFERRED_ORDER.items():
        if model_id.startswith(k):
            return (rank, model_id)
    return (999, model_id)


def get_model_options(api_key: str | None) -> List[str]:
    """Получить доступные модели из API, отфильтрованные под текстовую генерацию."""
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

# ==========================
# Signaalwoorden и профили
# ==========================
SIGNALWORDS_B1: List[str] = CFG_SIGNALWORDS_B1
SIGNALWORDS_B2_PLUS: List[str] = CFG_SIGNALWORDS_B2_PLUS
PROMPT_PROFILES = PR_PROMPT_PROFILES

L1_LANGS = CFG_L1_LANGS  # code -> {label, name, csv_translation, csv_gloss}

# ==========================
# Sidebar (API, model, params)
# ==========================
st.sidebar.header(CFG_MESSAGES.get("sidebar_api_header", "🔐 API Settings"))
API_KEY = (
    st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets
    else st.sidebar.text_input(CFG_MESSAGES.get("api_key_label", "OpenAI API Key"), type="password")
)

# Версия SDK (подсказка)
try:
    import openai as _openai
    st.sidebar.caption(f"OpenAI SDK: v{_openai.__version__}")
except Exception:
    pass

# Модель (динамический список с фильтром)
model_options = get_model_options(API_KEY)
model = st.sidebar.selectbox(
    CFG_MESSAGES.get("model_label", "Model"),
    model_options,
    index=0,
    help=CFG_MESSAGES.get("model_help", ""),
)

# Профиль промпта
profile = st.sidebar.selectbox(
    CFG_MESSAGES.get("profile_label", "Prompt profile"),
    list(PROMPT_PROFILES.keys()),
    index=list(PROMPT_PROFILES.keys()).index("strict") if "strict" in PROMPT_PROFILES else 0,
)

# CEFR уровень
level = st.sidebar.selectbox(CFG_MESSAGES.get("cefr_label", "CEFR"), ["A1", "A2", "B1", "B2", "C1", "C2"], index=2)

# L1 язык пользователя (переводы/глоссы)
L1_code = st.sidebar.selectbox(CFG_MESSAGES.get("l1_label", "Your language (L1)"), list(L1_LANGS.keys()), index=0)
L1_meta = L1_LANGS[L1_code]

# Температура (некоторые модели не принимают)
TMIN, TMAX, TDEF, TSTEP = CFG_TMIN, CFG_TMAX, CFG_TDEF, CFG_TSTEP
temperature = st.sidebar.slider(CFG_MESSAGES.get("temp_label", "Temperature"), TMIN, TMAX, TDEF, TSTEP)

# CSV/Anki export options
csv_with_header = st.sidebar.checkbox(
    CFG_MESSAGES.get("csv_header_checkbox", "CSV: включить строку заголовка"),
    value=True,
    help=CFG_MESSAGES.get("csv_header_help", "")
)
_guid_label = st.sidebar.selectbox(
    CFG_MESSAGES.get("anki_guid_policy_label", "Anki GUID policy"),
    CFG_MESSAGES.get("anki_guid_policy_options", ["stable (update/skip existing)", "unique per export (import as new)"]),
    index=0,
    help=CFG_MESSAGES.get("anki_guid_policy_help", ""),
)

st.session_state["csv_with_header"] = csv_with_header
st.session_state["anki_guid_policy"] = "unique" if _guid_label.startswith("unique") else "stable"

# Сохраняем выбор для доступа внутри функций
st.session_state["prompt_profile"] = profile
st.session_state["level"] = level
st.session_state["L1_code"] = L1_code

# ==========================
# Demo / upload / state
# ==========================
if "input_data" not in st.session_state:
    st.session_state.input_data: List[Dict] = []
if "results" not in st.session_state:
    st.session_state.results: List[Dict] = []

st.title(CFG_MESSAGES.get("app_title", "📘 Anki CSV/Anki Builder — Dutch Cloze Cards"))

# Демо из конфига
DEMO_WORDS = CFG_DEMO_WORDS

col_demo, col_clear = st.columns([1,1])
with col_demo:
    if st.button(CFG_MESSAGES.get("try_demo_button", "Try demo"), type="secondary"):
        st.session_state.input_data = DEMO_WORDS
        st.toast(CFG_MESSAGES.get("demo_loaded", "🔁 Demo loaded"), icon="✅")
with col_clear:
    if st.button(CFG_MESSAGES.get("clear_button", "Очистить"), type="secondary"):
        st.session_state.input_data = []
        st.session_state.results = []

# Upload
uploaded_file = st.file_uploader(CFG_MESSAGES.get("uploader_label", "Upload .txt / .md"), type=["txt", "md"], accept_multiple_files=False)

# ==========================
# Parsing входных форматов
# ==========================

def parse_input(text: str) -> List[Dict]:
    rows: List[Dict] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        # 1) Markdown-таблица: | **woord** | definitie NL | RU |
        if line.startswith("|") and "**" in line:
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3:
                woord = re.sub(r"\*", "", parts[0]).strip()
                def_nl = parts[1].strip()
                ru_short = parts[2].strip()
                entry = {"woord": woord}
                if def_nl:
                    entry["def_nl"] = def_nl
                if ru_short:
                    entry["ru_short"] = ru_short
                rows.append(entry)
            continue
        # 4) TSV: woord \t def_nl
        if "\t" in line:
            tparts = [p.strip() for p in line.split("\t")]
            if len(tparts) == 2:
                rows.append({"woord": tparts[0], "def_nl": tparts[1]})
                continue
        # 2) Построчный: woord — def NL — RU  |  woord — def NL
        if " — " in line:
            parts = [p.strip() for p in line.split(" — ")]
            if len(parts) == 3:
                rows.append({"woord": parts[0], "def_nl": parts[1], "ru_short": parts[2]})
                continue
            if len(parts) == 2:
                rows.append({"woord": parts[0], "def_nl": parts[1]})
                continue
        # 3) Просто слово
        rows.append({"woord": line})
    return rows

if uploaded_file is not None:
    try:
        file_text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        file_text = uploaded_file.read().decode("utf-16")
    st.session_state.input_data = parse_input(file_text)

# Preview входных данных
if st.session_state.input_data:
    st.subheader(CFG_MESSAGES.get("recognized_rows_title", "🔍 Распознанные строки"))
    st.dataframe(pd.DataFrame(st.session_state.input_data), use_container_width=True)
else:
    st.info(CFG_MESSAGES.get("upload_hint", "Загрузите файл или нажмите **Try demo**"))

# ==========================
# Helpers: sanitize, temperature policy, prompt compose
# ==========================

def sanitize(value: str) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "∣").strip()


def _should_pass_temperature(model_id: str) -> bool:
    no_temp = st.session_state.get("no_temp_models", set())
    if model_id in no_temp:
        return False
    if model_id.startswith(("gpt-5", "o3")):
        return False
    return True


def _det_include_signalword(woord: str, level: str) -> bool:
    if level in ("B1", "B2", "C1", "C2"):
        seed = int(hashlib.sha256(f"{woord}|{level}".encode()).hexdigest(), 16) % 100
        return seed < 50  # ≈50%
    return False


# compose_instructions_en импортируется из prompts.py


# ==========================
# OpenAI call + validation/repair
# ==========================

def extract_json_block(text: str) -> Dict:
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def validate_card(card: Dict) -> List[str]:
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
    # collocations: exactly 3 items by '; '
    col_raw = card.get("L2_collocations", "")
    items = [s.strip() for s in re.split(r";\s*|\n+", col_raw) if s.strip()]
    if len(items) != 3:
        problems.append("L2_collocations must contain exactly 3 items")
    # L1_gloss: max 2 words
    if len(card.get("L1_gloss", "").split()) > 2:
        problems.append("L1_gloss must be 1–2 words")
    return problems


def call_openai_card(client: OpenAI, row: Dict, model: str, temperature: float, L1_code: str, level: str, profile: str) -> Dict:
    # Compose instructions (EN)
    instructions = compose_instructions_en(L1_code, level, profile)

    # Include signal word policy for current card (deterministic 50% for B1+)
    include_sig = _det_include_signalword(row.get("woord", ""), level)
    sig_list = SIGNALWORDS_B1 if level == "B1" else (SIGNALWORDS_B2_PLUS if level in ("B2", "C1", "C2") else [])

    payload = {
        "L2_word": row.get("woord", "").strip(),
        "given_L2_definition": row.get("def_nl", "").strip(),
        "preferred_L1_gloss": row.get("ru_short", "").strip(),  # backward compat
        "L1": L1_LANGS[L1_code]["name"],
        "CEFR": level,
        "INCLUDE_SIGNALWORD": include_sig,
        "ALLOWED_SIGNALWORDS": sig_list,
    }

    # Формируем kwargs с учётом поддержки temperature
    kwargs = dict(
        model=model,
        instructions=instructions,
        input=(
            "Input data (JSON):\n" + json.dumps(payload, ensure_ascii=False) +
            "\nReturn STRICT JSON with fields: L2_word, L2_cloze, L1_sentence, L2_collocations, L2_definition, L1_gloss."
        ),
    )
    if _should_pass_temperature(model):
        kwargs["temperature"] = temperature

    try:
        resp = client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e)
        if "Unsupported parameter: 'temperature'" in msg or "param': 'temperature'" in msg:
            no_temp = st.session_state.get("no_temp_models", set())
            no_temp.add(model)
            st.session_state.no_temp_models = no_temp
            kwargs.pop("temperature", None)
            resp = client.responses.create(**kwargs)
        else:
            raise

    parsed = extract_json_block(getattr(resp, "output_text", ""))

    # Санитизация
    card = {
        "L2_word": sanitize(parsed.get("L2_word", payload["L2_word"])),
        "L2_cloze": sanitize(parsed.get("L2_cloze", "")),
        "L1_sentence": sanitize(parsed.get("L1_sentence", "")),
        "L2_collocations": sanitize(parsed.get("L2_collocations", "")),
        "L2_definition": sanitize(parsed.get("L2_definition", payload.get("given_L2_definition", ""))),
        "L1_gloss": sanitize(parsed.get("L1_gloss", payload.get("preferred_L1_gloss", ""))),
    }

    # Валидация + один repair-pass при необходимости
    problems = validate_card(card)
    if problems:
        repair_prompt = instructions + "\n\nREPAIR: The previous JSON has issues: " + "; ".join(problems) + ". " \
            + "Fix ONLY the problematic fields and return STRICT JSON again."
        repair_kwargs = dict(
            model=model,
            instructions=repair_prompt,
            input=("Previous JSON:\n" + json.dumps(card, ensure_ascii=False)),
        )
        if _should_pass_temperature(model):
            repair_kwargs["temperature"] = temperature
        try:
            resp2 = client.responses.create(**repair_kwargs)
        except Exception as e:
            msg = str(e)
            if "Unsupported parameter: 'temperature'" in msg or "param': 'temperature'" in msg:
                no_temp = st.session_state.get("no_temp_models", set())
                no_temp.add(model)
                st.session_state.no_temp_models = no_temp
                repair_kwargs.pop("temperature", None)
                resp2 = client.responses.create(**repair_kwargs)
            else:
                raise
        parsed2 = extract_json_block(getattr(resp2, "output_text", ""))
        if parsed2:
            card = {
                "L2_word": sanitize(parsed2.get("L2_word", card["L2_word"])),
                "L2_cloze": sanitize(parsed2.get("L2_cloze", card["L2_cloze"])),
                "L1_sentence": sanitize(parsed2.get("L1_sentence", card["L1_sentence"])),
                "L2_collocations": sanitize(parsed2.get("L2_collocations", card["L2_collocations"])),
                "L2_definition": sanitize(parsed2.get("L2_definition", card["L2_definition"])),
                "L1_gloss": sanitize(parsed2.get("L1_gloss", card["L1_gloss"])),
            }

    return card

# ==========================
# CSV генерация (динамическая шапка под L1)
# ==========================

def generate_csv(results: List[Dict], L1_code: str, include_header: bool = True) -> str:
    meta = L1_LANGS[L1_code]
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, delimiter='|', lineterminator='\n')

    if include_header:
        writer.writerow([
            CFG_CSV_HEADERS_FIXED.get("nl_word", "NL-слово"),
            CFG_CSV_HEADERS_FIXED.get("nl_sentence_cloze", "Предложение NL (с cloze)"),
            f"{meta['csv_translation']} {meta['label']}",
            CFG_CSV_HEADERS_FIXED.get("collocations_nl", "Коллокации (NL)"),
            CFG_CSV_HEADERS_FIXED.get("definition_nl", "Определение NL"),
            f"{meta['csv_gloss']} {meta['label']}",
        ])
    for r in results:
        writer.writerow([
            r.get('L2_word',''),
            r.get('L2_cloze',''),
            r.get('L1_sentence',''),
            r.get('L2_collocations',''),
            r.get('L2_definition',''),
            r.get('L1_gloss',''),
        ])
    return csv_buffer.getvalue()

# ==========================
# Anki .apkg export (genanki)
# ==========================

ANKI_MODEL_ID = CFG_ANKI_MODEL_ID
ANKI_DECK_ID = CFG_ANKI_DECK_ID

FRONT_HTML_TEMPLATE = CFG_FRONT_HTML_TEMPLATE
BACK_HTML_TEMPLATE = CFG_BACK_HTML_TEMPLATE
CSS_STYLING = CFG_CSS_STYLING


def _compute_guid(c: Dict, policy: str, run_id: str) -> str:
    base = f"{c.get('L2_word','')}|{c.get('L2_cloze','')}"
    if policy == "unique":
        base = base + "|" + run_id
    try:
        import genanki as _g
        return _g.guid_for(base)
    except Exception:
        # Fallback: короткий SHA1
        return hashlib.sha1(base.encode('utf-8')).hexdigest()[:10]

def build_anki_package(cards: List[Dict], L1_label: str, guid_policy: str, run_id: str) -> bytes:
    if not HAS_GENANKI:
        raise RuntimeError("genanki is not installed. Add 'genanki' to requirements.txt and redeploy.")

    # Подставляем заголовок хинта
    front = FRONT_HTML_TEMPLATE.replace("{L1_LABEL}", L1_label)
    back = BACK_HTML_TEMPLATE

    model = genanki.Model(
        ANKI_MODEL_ID,
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
        templates=[
            {
                "name": "Cloze",
                "qfmt": front,
                "afmt": back,
            }
        ],
        css=CSS_STYLING,
        model_type=genanki.Model.CLOZE,
    )

    deck = genanki.Deck(ANKI_DECK_ID, CFG_ANKI_DECK_NAME)

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
    # Если позже будут медиа — pkg.media_files = [...]
    bio = io.BytesIO()
    pkg.write_to_file(bio)
    return bio.getvalue()

# ==========================
# Generate section
# ==========================
if st.session_state.input_data:
    if st.button(CFG_MESSAGES.get("generate_button", "Сгенерировать карточки"), type="primary"):
        if not API_KEY:
            st.error(CFG_MESSAGES.get("no_api_key", "Укажи OPENAI_API_KEY в Secrets или в поле слева."))
        else:
            client = OpenAI(api_key=API_KEY)
            # Запоминаем run_id для GUID'ов в этом прогоне
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
                    st.error(CFG_MESSAGES.get("error_card_processing_fmt", "Ошибка при обработке слова '{woord}': {error}").format(woord=row.get('woord','?'), error=e))
                finally:
                    progress.progress(int((idx + 1) / max(total,1) * 100))
                    if CFG_API_DELAY > 0:
                        time.sleep(CFG_API_DELAY)

# ==========================
# Preview & downloads
# ==========================
if st.session_state.results:
    st.subheader(CFG_MESSAGES.get("preview_title_fmt", "📋 Предпросмотр карточек (первые {limit})").format(limit=CFG_PREVIEW_LIMIT))
    preview_df = pd.DataFrame(st.session_state.results)[:CFG_PREVIEW_LIMIT]
    st.dataframe(preview_df, use_container_width=True)

    # CSV download
    csv_data = generate_csv(st.session_state.results, L1_code, include_header=st.session_state.get('csv_with_header', True))
    st.download_button(
        label=CFG_MESSAGES.get("csv_download_label", "📥 Скачать anki_cards.csv"),
        data=csv_data,
        file_name="anki_cards.csv",
        mime="text/csv",
    )

    # APKG download
    if HAS_GENANKI:
        try:
            anki_bytes = build_anki_package(
                st.session_state.results,
                L1_label=L1_meta["label"],
                guid_policy=st.session_state.get("anki_guid_policy", "stable"),
                run_id=st.session_state.get("anki_run_id", str(int(time.time())))
            )
            st.download_button(
                label=CFG_MESSAGES.get("apkg_download_label", "🧩 Скачать колоду Anki (.apkg)"),
                data=anki_bytes,
                file_name="dutch_cloze.apkg",
                mime="application/octet-stream",
            )
        except Exception as e:
            st.error(CFG_MESSAGES.get("error_apkg_build_fmt", "Не удалось собрать .apkg: {error}").format(error=e))
    else:
        st.info(CFG_MESSAGES.get("apkg_install_hint", "Для экспорта в .apkg добавь в requirements.txt строку 'genanki' и перезагрузи приложение."))

# ==========================
# Footer
# ==========================
st.caption(CFG_MESSAGES.get("footer_tips", ""))
