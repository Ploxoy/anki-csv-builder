import streamlit as st
import pandas as pd
import re
import csv
import io
import json
from typing import List, Dict
from openai import OpenAI

st.set_page_config(page_title="Anki CSV Builder", layout="wide")
# ==========================
# Вспомогательные функции
# ==========================

def sanitize_field(value: str) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "∣").strip()


# ==========================
# Session state (prevents reset on rerun)
# ==========================
if "input_data" not in st.session_state:
    st.session_state.input_data: List[Dict] = []
if "results" not in st.session_state:
    st.session_state.results: List[Dict] = []

# ==========================
# Sidebar: API, model, params
# ==========================
st.sidebar.header("🔐 API Settings")
api_key = (
    st.secrets.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in st.secrets
    else st.sidebar.text_input("OpenAI API Key", type="password")
)

model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4.1"],
    index=0,
)

temperature = st.sidebar.slider("Temperature", 0.2, 0.8, 0.4, 0.1)
stream_output = st.sidebar.checkbox("Stream output (beta)", value=False,
    help="Streaming в Responses API: финальный JSON будет доступен после завершения стрима")

# ==========================
# Demo data
# ==========================
demo_words = [
    {"woord": "aanraken", "def_nl": "iets met je hand of een ander deel van je lichaam voelen"},
    {"woord": "begrijpen", "def_nl": "snappen wat iets betekent of inhoudt"},
    {"woord": "gillen", "def_nl": "hard en hoog schreeuwen"},
    {"woord": "kloppen", "def_nl": "met regelmaat bonzen of tikken"},
    {"woord": "toestaan", "def_nl": "goedkeuren of laten gebeuren"},
    {"woord": "opruimen", "def_nl": "iets netjes maken door het op zijn plaats te leggen"},
]

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
        st.session_state.input_data = demo_words
        st.toast("🔁 Используется демо-набор из 6 слов", icon="✅")
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

def sanitize_field(value: str) -> str:
    if value is None:
        return ""
    # Не допускаем символ '|' в полях CSV
    return str(value).replace("|", "∣").strip()

def call_openai_card(client: OpenAI, row: Dict, model: str, temperature: float) -> Dict:
    """Совместимо с SDK >=1.0: без response_format. Просим СТРОГИЙ JSON и парсим output_text."""
    system_instructions = (
        "Ты — опытный лексикограф NL→RU и автор учебных материалов. "
        "Сгенерируй СТРОГО JSON-объект карточки Anki со структурой: "
        "{woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short}. "
        "Правила: 1) NL-предложение короткое, естественное; целевое слово в {{c1::…}}; "
        "для разделимых глаголов: {{c1::stam}} … {{c2::partikel}}. "
        "2) ru_sentence — точный перевод NL-предложения на русский. "
        "3) collocaties — РОВНО 3 частотные связки, разделитель '; '. Никаких выдуманных сочетаний. "
        "4) def_nl — короткая NL-дефиниция. 5) ru_short — 1–2 русских слова. "
        "6) Символ '|' в текстах не использовать. Верни только JSON, без пояснений."
    )

    user_payload = {
        "woord": row.get("woord", "").strip(),
        "def_nl": row.get("def_nl", "").strip(),
        "ru_short": row.get("ru_short", "").strip(),
    }

    # Responses API по официальному примеру: instructions + input
    final = client.responses.create(
        model=model,
        instructions=system_instructions,
        input=(
            "Пользовательские данные:\n"
            + json.dumps(user_payload, ensure_ascii=False)
            + "\nВерни СТРОГО JSON с полями: "
              "woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short."
        ),
        temperature=temperature,
    )

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
            st.error("Укажи OPENAI_API_KEY в Secrets или в поле слева.")
        else:
            client = OpenAI(api_key=api_key)
            st.session_state.results = []
            progress = st.progress(0)
            total = len(st.session_state.input_data)

            for idx, row in enumerate(st.session_state.input_data):
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
    st.subheader("📋 Предпросмотр карточек (первые 20)")
    preview_df = pd.DataFrame(st.session_state.results)[:20]
    st.dataframe(preview_df, use_container_width=True)

  # CSV буфер
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, delimiter='|', lineterminator='\n')
    writer.writerow([
        "NL-слово",
        "Предложение NL (с cloze)",
        "Перевод RU",
        "Коллокации",
        "Определение NL",
        "Перевод слова RU",
    ])
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
st.caption(
    "Советы: 1) добавляй качественные NL-дефиниции во вход — это улучшит примеры; "
    "2) если видишь странные коллокации — перезапусти генерацию для конкретного слова, "
    "3) символ '|' в текстах заменяется на '∣'."
)
