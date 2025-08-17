import streamlit as st
import pandas as pd
import re
import csv
import io
import json
from typing import List, Dict
from openai import OpenAI
# ==========================
# Модели: дефолтный список + динамическая подгрузка из API
# ==========================
from typing import List

DEFAULT_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
]

_PREFERRED_ORDER = {  # чем меньше число — тем выше в списке
    "gpt-5": 0,
    "gpt-5-mini": 1,
    "gpt-5-nano": 2,
    "gpt-4.1": 3,
    "gpt-4o": 4,
    "gpt-4o-mini": 5,
    "o3": 6,
    "o3-mini": 7,
}

def _sort_key(model_id: str) -> tuple:
    for k, rank in _PREFERRED_ORDER.items():
        if model_id.startswith(k):
            return (rank, model_id)
    return (999, model_id)

def get_model_options(api_key: str | None) -> List[str]:
    """Вернёт список доступных ID моделей. Сначала пытаемся спросить у API, иначе — дефолт."""
    prefixes = ("gpt-5", "gpt-4.1", "gpt-4o", "o3")
    if not api_key:
        return DEFAULT_MODELS
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        ids = []
        for m in getattr(models, "data", []) or []:
            mid = getattr(m, "id", "")
            if any(mid.startswith(p) for p in prefixes):
                ids.append(mid)
        if not ids:
            return DEFAULT_MODELS
        return sorted(set(ids), key=_sort_key)
    except Exception:
        return DEFAULT_MODELS

# ===== Усиленный системный промпт для устойчивого заполнения всех полей =====
PROMPT_SYSTEM = (
    "Ты — опытный лексикограф NL→RU и автор учебных материалов. "
    "Сгенерируй СТРОГО JSON-объект карточки Anki со структурой: "
    "{woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short}.\n"
    "ОБЩИЕ ПРАВИЛА (ОЧЕНЬ ВАЖНО):\n"
    "• Верни ТОЛЬКО JSON БЕЗ пояснений и форматирования.\n"
    "• НИ ОДНО поле не пустое. Запрещены пустые строки.\n"
    "• Символ '|' в текстах запрещён.\n"
    "• Если дано def_nl — строго следуй ему; не меняй базовое значение слова.\n"
    "• Сохраняй часть речи: ru_short должен соответствовать части речи слова "
    "(глагол→инфинитив; существительное→существительное; прилагательное→прилагательное).\n"
    "• ru_sentence — ТОЧНЫЙ перевод NL-предложения, без перефраза.\n"
    "• cloze_sentence — одно короткое естественное NL-предложение (8–14 слов, настоящее время, "
    "без имён/цифр/кавычек); целевое слово внутри {{c1::…}}.\n"
    "  Если слово — разделимый глагол: {{c1::stam}} … {{c2::partikel}}. Иначе только {{c1::…}}.\n"
    "• collocaties — РОВНО 3 частотные связки, разделитель '; ' (точка с запятой и пробел).\n"
    "  Каждая связка — 2–3 слова с целевым словом в естественной форме. Нельзя: бессмысленные пары "
    "(например, 'een grote caissière'), редкие/книжные, имена собственные.\n"
    "• Избегай редкой лексики; используй A2–B1 вокруг целевого слова.\n\n"
    "ФОРМАТ ВЫВОДА: один JSON-объект с ключами: woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short.\n\n"
    "ПРИМЕРЫ (стиль, НЕ копируй слова):\n"
    "// Существительное\n"
    "{\"woord\": \"boodschap\", \"cloze_sentence\": \"Hij doet elke dag de {{c1::boodschap}}.\", "
    "\"ru_sentence\": \"Он делает покупки каждый день.\", "
    "\"collocaties\": \"boodschappen doen; een boodschap doorgeven; een duidelijke boodschap\", "
    "\"def_nl\": \"iets wat je wilt zeggen of inkopen die je doet\", \"ru_short\": \"покупка; послание\"}\n"
    "// Разделимый глагол\n"
    "{\"woord\": \"opruimen\", \"cloze_sentence\": \"Na het eten {{c1::ruimt}} hij de tafel {{c2::op}}.\", "
    "\"ru_sentence\": \"После еды он убирает со стола.\", "
    "\"collocaties\": \"de kamer opruimen; speelgoed opruimen; netjes opruimen\", "
    "\"def_nl\": \"iets op zijn plaats leggen zodat het netjes is\", \"ru_short\": \"убирать\"}\n"
    "// Прилагательное\n"
    "{\"woord\": \"streng\", \"cloze_sentence\": \"De docent is vandaag {{c1::streng}}.\", "
    "\"ru_sentence\": \"Преподаватель сегодня строгий.\", "
    "\"collocaties\": \"strenge regels; een strenge docent; streng optreden\", "
    "\"def_nl\": \"met veel eisen en weinig toelating\", \"ru_short\": \"строгий\"}"
)


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
    help="Лучшее качество — gpt-5 (если доступен); баланс — gpt-4.1; быстрее/дешевле — gpt-4o / gpt-5-mini."
)

# (необязательно) точный ID снапшота модели
custom = st.sidebar.text_input("Custom model id (optional)", placeholder="например, gpt-5-2025-08-07")
if custom.strip():
    model = custom.strip()

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



def call_openai_card(client: OpenAI, row: Dict, model: str, temperature: float) -> Dict:
    """Совместимо с SDK >=1.0: без response_format. Просим СТРОГИЙ JSON и парсим output_text."""
    system_instructions = PROMPT_SYSTEM

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
