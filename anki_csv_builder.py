import streamlit as st
import pandas as pd
import re
import csv
import io
import json
import time
from typing import List, Dict
from openai import OpenAI
from config import (
    DEFAULT_MODELS, get_preferred_order, get_block_substrings, get_allowed_prefixes,
    PROMPT_SYSTEM, DEMO_WORDS, PAGE_TITLE, PAGE_LAYOUT,
    TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DEFAULT, TEMPERATURE_STEP,
    CSV_DELIMITER, CSV_LINETERMINATOR, PREVIEW_LIMIT, API_REQUEST_DELAY,
    CSV_HEADERS, MESSAGES
)



def _sort_key(model_id: str) -> tuple:
    preferred_order = get_preferred_order()
    for k, rank in preferred_order.items():
        if model_id.startswith(k):
            return (rank, model_id)
    return (999, model_id)

def get_model_options(api_key: str | None) -> List[str]:
    if not api_key:
        return DEFAULT_MODELS
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        ids = []
        allowed_prefixes = get_allowed_prefixes()
        block_substrings = get_block_substrings()
        for m in getattr(models, "data", []) or []:
            mid = getattr(m, "id", "")
            if any(mid.startswith(p) for p in allowed_prefixes) \
               and not any(b in mid for b in block_substrings):
                ids.append(mid)
        if not ids:
            return DEFAULT_MODELS
        return sorted(set(ids), key=_sort_key)
    except Exception:
        return DEFAULT_MODELS


# Системный промпт импортируется из config.py


st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
# ==========================
# Вспомогательные функции
# ==========================

def sanitize_field(value: str) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "∣").strip()

def _should_pass_temperature(model_id: str) -> bool:
    """
    Эвристика: не передавать temperature в семейства, где параметр не поддерживается.
    Плюс учитываем то, что уже выучили на рантайме (session_state.no_temp_models).
    """
    no_temp = st.session_state.get("no_temp_models", set())
    if model_id in no_temp:
        return False
    # известные семейства без поддержки temperature в Responses API
    if model_id.startswith(("gpt-5", "o3")):
        return False
    return True



# ==========================
# Session state (prevents reset on rerun)
# ==========================
if "input_data" not in st.session_state:
    st.session_state.input_data: List[Dict] = []
if "results" not in st.session_state:
    st.session_state.results: List[Dict] = []

# --- SIDEBAR ---
st.sidebar.header("🔐 API Settings")
api_key = (
    st.secrets.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in st.secrets
    else st.sidebar.text_input("OpenAI API Key", type="password")
)

# динамически получаем доступные модели (если ключа нет — вернётся дефолтный список)
options = get_model_options(api_key)
model = st.sidebar.selectbox(
    "Model",
    options,
    index=0,
    help=MESSAGES["help_temperature"]
)
if not _should_pass_temperature(model):
    st.sidebar.caption(MESSAGES["temperature_unavailable"])


# (необязательно) точный ID снапшота модели
custom = st.sidebar.text_input("Custom model id (optional)", placeholder=MESSAGES["placeholder_custom_model"])
if custom.strip():
    model = custom.strip()


temperature = st.sidebar.slider("Temperature", TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DEFAULT, TEMPERATURE_STEP)



stream_output = st.sidebar.checkbox("Stream output (beta)", value=False,
    help=MESSAGES["help_stream"])

# Демо-данные импортируются из config.py

st.title("📘 Anki CSV Builder — Cloze Cards from Dutch Words")

# ==========================
# Parsers
# ==========================

def parse_input(text: str) -> List[Dict]:
    rows: List[Dict] = []
    lines = text.strip().splitlines()

    for raw in lines:
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

        # 4) TSV: woord 	 def_nl  (2 колонки, без шапки)
        if "	" in line:
            tparts = [p.strip() for p in line.split("	")]
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

# ==========================
# File upload + demo button
# ==========================
uploaded_file = st.file_uploader("Upload een bestand (.txt / .md)", type=["txt", "md"], accept_multiple_files=False)

col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("Try demo", type="secondary"):
        st.session_state.input_data = DEMO_WORDS
        st.toast(MESSAGES["demo_loaded"], icon="✅")
with col_b:
    if st.button("Очистить", type="secondary"):
        st.session_state.input_data = []
        st.session_state.results = []

if uploaded_file is not None:
    try:
        file_text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)    
        file_text = uploaded_file.read().decode("utf-16")
    st.session_state.input_data = parse_input(file_text)

# Предпросмотр входных данных
if st.session_state.input_data:
    st.subheader("🔍 Распознанные строки")
    st.dataframe(pd.DataFrame(st.session_state.input_data), use_container_width=True)
else:
    st.info("Загрузите файл или нажмите **Try demo**")

# ==========================
# Helpers
# ==========================



def call_openai_card(client: OpenAI, row: Dict, model: str, temperature: float) -> Dict:
    """Совместимо с SDK >=1.0: без response_format. Просим СТРОГИЙ JSON и парсим output_text."""
    system_instructions = PROMPT_SYSTEM

    user_payload = {
        "woord": row.get("woord", "").strip(),
        "def_nl": row.get("def_nl", "").strip(),
        "ru_short": row.get("ru_short", "").strip(),
    }

    # --- формируем kwargs c учётом поддержки temperature ---
    kwargs = dict(
        model=model,
        instructions=system_instructions,
        input=(
            "Пользовательские данные:\n"
            + json.dumps(user_payload, ensure_ascii=False)
            + "\nВерни СТРОГО JSON с полями: woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short."
        ),
    )
    if _should_pass_temperature(model):
        kwargs["temperature"] = temperature

    try:
        final = client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e)
        # если модель не принимает temperature — запомним и повторим без него
        if "Unsupported parameter: 'temperature'" in msg or "param': 'temperature'" in msg:
            no_temp = st.session_state.get("no_temp_models", set())
            no_temp.add(model)
            st.session_state.no_temp_models = no_temp
            kwargs.pop("temperature", None)
            final = client.responses.create(**kwargs)
        else:
            raise


    # Достаём текст и вырезаем JSON
    text = getattr(final, "output_text", "") or ""
    m = re.search(r"\{[\s\S]*\}", text)
    parsed = json.loads(m.group(0)) if m else {}

    # Санитизация '|' → '∣'
    def sf(x: str) -> str:
        return (x or "").replace("|", "∣").strip()

    return {
        "woord": sf(parsed.get("woord", user_payload["woord"])),
        "cloze_sentence": sf(parsed.get("cloze_sentence", "")),
        "ru_sentence": sf(parsed.get("ru_sentence", "")),
        "collocaties": sf(parsed.get("collocaties", "")),
        "def_nl": sf(parsed.get("def_nl", user_payload.get("def_nl", ""))),
        "ru_short": sf(parsed.get("ru_short", user_payload.get("ru_short", ""))),
    }


# ==========================
# Generate section
# ==========================
if st.session_state.input_data:
    if st.button("Сгенерировать CSV", type="primary"):
        if not api_key:
            st.error(MESSAGES["no_api_key"])
        else:
            client = OpenAI(api_key=api_key)
            st.session_state.results = []
            progress = st.progress(0)
            total = len(st.session_state.input_data)

            for idx, row in enumerate(st.session_state.input_data):
                if idx > 0:  # не ждем перед первым запросом
                    time.sleep(API_REQUEST_DELAY)
                try:
                    card = call_openai_card(client, row, model=model, temperature=temperature)
                    st.session_state.results.append(card)
                except Exception as e:
                    st.error(f"Ошибка при обработке слова '{row.get('woord','?')}': {e}")
                finally:
                    progress.progress(int((idx + 1) / max(total,1) * 100))

# ==========================
# Preview & download
# ==========================
if st.session_state.results:
    st.subheader(f"📋 Предпросмотр карточек (первые {PREVIEW_LIMIT})")
    preview_df = pd.DataFrame(st.session_state.results)[:PREVIEW_LIMIT]
    st.dataframe(preview_df, use_container_width=True)

  # CSV буфер
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, delimiter=CSV_DELIMITER, lineterminator=CSV_LINETERMINATOR)
    writer.writerow(CSV_HEADERS)
    for r in st.session_state.results:
        writer.writerow([
            r.get('woord',''),
            r.get('cloze_sentence',''),
            r.get('ru_sentence',''),
            r.get('collocaties',''),
            r.get('def_nl',''),
            r.get('ru_short',''),
        ])

    st.download_button(
        label="📥 Скачать anki_cards.csv",
        data=csv_buffer.getvalue(),
        file_name="anki_cards.csv",
        mime="text/csv",
    )

# ==========================
# Footer help
# ==========================
st.caption(MESSAGES["footer_tips"])
