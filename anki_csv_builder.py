import streamlit as st
import pandas as pd
import re
import csv
import io
from openai import OpenAI
from typing import List, Dict

st.set_page_config(page_title="Anki CSV Builder", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("🔐 API Settings")
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.sidebar.text_input("OpenAI API Key", type="password")

model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4.1"], index=0)
temperature = st.sidebar.slider("Temperature", 0.2, 0.8, 0.4, 0.1)
stream_output = st.sidebar.checkbox("Stream output", value=False)

# --- DEMO DATA ---
demo_words = [
    {"woord": "aanraken", "def_nl": "iets met je hand of een ander deel van je lichaam voelen"},
    {"woord": "begrijpen", "def_nl": "snappen wat iets betekent of inhoudt"},
    {"woord": "gillen", "def_nl": "hard en hoog schreeuwen"},
    {"woord": "kloppen", "def_nl": "met regelmaat bonzen of tikken"},
    {"woord": "toestaan", "def_nl": "goedkeuren of laten gebeuren"}
]

st.title("📘 Anki CSV Builder — Cloze Cards from Dutch Words")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload een bestand (.txt / .md)", type=["txt", "md"])

def parse_input(text: str) -> List[Dict]:
    rows = []
    lines = text.strip().splitlines()
    md_table = re.compile(r"\|\s*\*\*(.*?)\*\*\s*\|")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("|") and "**" in line:
            parts = [p.strip().strip("*") for p in line.strip("|").split("|")]
            if len(parts) >= 3:
                rows.append({"woord": parts[0], "def_nl": parts[1], "ru_short": parts[2]})
        elif "\t" in line:
            parts = line.split("\t")
            if len(parts) == 2:
                rows.append({"woord": parts[0].strip(), "def_nl": parts[1].strip()})
        elif " — " in line:
            parts = line.split(" — ")
            if len(parts) == 3:
                rows.append({"woord": parts[0], "def_nl": parts[1], "ru_short": parts[2]})
            elif len(parts) == 2:
                rows.append({"woord": parts[0], "def_nl": parts[1]})
        else:
            rows.append({"woord": line})
    return rows

# --- INPUT PROCESSING ---
input_data = []
if uploaded_file:
    file_text = uploaded_file.read().decode("utf-8")
    input_data = parse_input(file_text)
    st.success(f"✅ Распознано {len(input_data)} слов из файла")
elif st.button("Try demo"):
    input_data = demo_words
    st.success("🔁 Используется демо-набор из 5 слов")

if input_data:
    st.subheader("🔍 Распознанные строки")
    st.dataframe(pd.DataFrame(input_data))

    if st.button("Сгенерировать CSV"):
        client = OpenAI(api_key=api_key)

        result_rows = []
        progress = st.progress(0.0)
        total = len(input_data)

        for idx, row in enumerate(input_data):
            prompt = {
                "instruction": "Генерируй карточку Anki по образцу (cloze, RU, collocaties, def NL, RU short)",
                "input": row,
                "schema": {
                    "type": "object",
                    "properties": {
                        "woord": {"type": "string"},
                        "cloze_sentence": {"type": "string"},
                        "ru_sentence": {"type": "string"},
                        "collocaties": {"type": "string"},
                        "def_nl": {"type": "string"},
                        "ru_short": {"type": "string"}
                    },
                    "required": ["woord", "cloze_sentence", "ru_sentence", "collocaties", "def_nl", "ru_short"]
                }
            }

            try:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    stream=stream_output
                )
                result_rows.append(response.output)
            except Exception as e:
                st.error(f"Ошибка при обработке слова {row['woord']}: {str(e)}")
                continue

            progress.progress((idx + 1) / total)

        # --- OUTPUT ---
        st.subheader("📋 Предпросмотр карточек")
        preview_df = pd.DataFrame(result_rows)[:20]
        st.dataframe(preview_df)

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer, delimiter='|')
        writer.writerow(["NL-слово", "Предложение NL (с cloze)", "Перевод RU", "Коллокации", "Определение NL", "Перевод слова RU"])
        for r in result_rows:
            writer.writerow([r['woord'], r['cloze_sentence'], r['ru_sentence'], r['collocaties'], r['def_nl'], r['ru_short']])

        st.download_button(
            label="📥 Скачать anki_cards.csv",
            data=csv_buffer.getvalue(),
            file_name="anki_cards.csv",
            mime="text/csv"
        )
