"""
Конфигурация для Anki CSV Builder
"""

from typing import List, Dict, Tuple

# ==========================
# Модели: дефолтный список + динамическая подгрузка из API
# ==========================

DEFAULT_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
]

_PREFERRED_ORDER: Dict[str, int] = {  # чем меньше число — тем выше в списке
    "gpt-5": 0,
    "gpt-5-mini": 1,
    "gpt-5-nano": 2,
    "gpt-4.1": 3,
    "gpt-4o": 4,
    "gpt-4o-mini": 5,
    "o3": 6,
    "o3-mini": 7,
}

# Модели, которые исключаем по подстроке в ID (нам нужен именно текст-генератор)
_BLOCK_SUBSTRINGS: Tuple[str, ...] = (
    "audio", "realtime",           # gpt-4o-audio-*, gpt-4o-realtime-*
    "embed", "embedding",          # text-embedding-*
    "whisper", "asr", "transcribe", "speech", "tts",  # ASR/TTS
    "moderation",                  # модерация
    "search",                      # поисковые/вспомогательные
    "vision", "vision-preview",    # чисто визуальные/превью
    "distill", "distilled",        # дистиллированные спец-модели
    "batch", "preview"             # служебные/превью/батчевые
)

# Разрешённые семейства (по префиксу) для текстовой генерации
_ALLOWED_PREFIXES: Tuple[str, ...] = ("gpt-5", "gpt-4.1", "gpt-4o", "o3")

# ==========================
# Сигнальные слова и профили промптов
# ==========================

SIGNALWORDS_B1: List[str] = [
    "omdat", "maar", "dus", "want", "terwijl", "daarom", "daardoor", "toch"
]

SIGNALWORDS_B2_PLUS: List[str] = [
    "hoewel", "zodat", "doordat", "bovendien", "echter", "bijvoorbeeld", "tenzij", "ondanks",
    "desondanks", "daarentegen", "aangezien", "zodra", "voordat", "nadat", "enerzijds ... anderzijds",
    "niet alleen ... maar ook", "opdat"
]

PROMPT_PROFILES: Dict[str, str] = {
    "strict": "Be literal and concise; avoid figurative language; keep the simplest structure that satisfies CEFR.",
    "balanced": "Natural and clear; minor synonymy allowed if it improves fluency.",
    "exam": "Neutral-formal register; precise; avoid colloquialisms.",
    "creative": "Allow mild figurativeness if it keeps clarity and CEFR constraints.",
}

# ==========================
# L1 языки и локализации заголовков CSV
# ==========================

L1_LANGS: Dict[str, Dict[str, str]] = {
    "RU": {"label": "RU", "name": "Russian", "csv_translation": "Перевод", "csv_gloss": "Перевод слова"},
    "EN": {"label": "EN", "name": "English", "csv_translation": "Translation", "csv_gloss": "Word gloss"},
    "ES": {"label": "ES", "name": "Spanish", "csv_translation": "Traducción", "csv_gloss": "Glosa"},
    "DE": {"label": "DE", "name": "German", "csv_translation": "Übersetzung", "csv_gloss": "Kurzgloss"},
}

# Для совместимости, если где-то ожидается отдельная мапа
CSV_HEADERS_LOCALIZATION = L1_LANGS

# Фиксированные части заголовков CSV (NL-колонки)
CSV_HEADERS_FIXED: Dict[str, str] = {
    "nl_word": "NL-слово",
    "nl_sentence_cloze": "Предложение NL (с cloze)",
    "collocations_nl": "Коллокации (NL)",
    "definition_nl": "Определение NL",
}

# ==========================
# Системный промпт
# ==========================

PROMPT_SYSTEM: str = ""  # перемещён в prompts.py при необходимости

# ==========================
# Демо-данные
# ==========================

DEMO_WORDS: List[Dict[str, str]] = [
    {"woord": "aanraken", "def_nl": "iets met je hand of een ander deel van je lichaam voelen"},
    {"woord": "begrijpen", "def_nl": "snappen wat iets betekent of inhoudt"},
    {"woord": "gillen", "def_nl": "hard en hoog schreeuwen"},
    {"woord": "kloppen", "def_nl": "met regelmaat bonzen of tikken"},
    {"woord": "toestaan", "def_nl": "goedkeuren of laten gebeuren"},
    {"woord": "opruimen", "def_nl": "iets netjes maken door het op zijn plaats te leggen"},
]

# ==========================
# Настройки UI
# ==========================

PAGE_TITLE: str = "Anki CSV/Anki Builder — Dutch Cloze Cards"
PAGE_LAYOUT: str = "wide"

# Настройки temperature slider
TEMPERATURE_MIN: float = 0.2
TEMPERATURE_MAX: float = 0.8
TEMPERATURE_DEFAULT: float = 0.4
TEMPERATURE_STEP: float = 0.1

# Настройки CSV
CSV_DELIMITER: str = '|'
CSV_LINETERMINATOR: str = '\n'

# Настройки предпросмотра
PREVIEW_LIMIT: int = 20

# Задержка между API запросами (в секундах)
API_REQUEST_DELAY: float = 0.1

# ==========================
# Заголовки CSV
# ==========================

CSV_HEADERS: List[str] = [
    "NL-слово",
    "Предложение NL (с cloze)",
    "Перевод RU",
    "Коллокации",
    "Определение NL",
    "Перевод слова RU",
]

# ==========================
# Сообщения и подсказки
# ==========================

MESSAGES = {
    # Общие подсказки
    "demo_loaded": "🔁 Демо-набор из 6 слов подставлен",
    "no_api_key": "Укажи OPENAI_API_KEY в Secrets или в поле слева.",
    "temperature_unavailable": "Температура недоступна для этой модели; она будет проигнорирована.",
    "help_temperature": "Лучшее качество — gpt-5 (если доступен); баланс — gpt-4.1; быстрее/дешевле — gpt-4o / gpt-5-mini.",
    "help_stream": "Streaming в Responses API: финальный JSON будет доступен после завершения стрима",
    "placeholder_custom_model": "например, gpt-5-2025-08-07",
    "footer_tips": (
        "Лайфхаки: 1) Чем лучше NL-дефиниции на входе, тем точнее пример и глосс. "
        "2) На уровнях B1+ примерно половина предложений будет со signaalwoorden. "
        "3) Для некоторых моделей (gpt-5/o3) температура не поддерживается и будет игнорироваться."
    ),

    # UI тексты
    "app_title": "📘 Anki CSV/Anki Builder — Dutch Cloze Cards",
    "sidebar_api_header": "🔐 API Settings",
    "api_key_label": "OpenAI API Key",
    "model_label": "Model",
    "model_help": "Лучшее качество — gpt-5 (если доступен); баланс — gpt-4.1; быстрее/дешевле — gpt-4o / gpt-5-mini.",
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
    "upload_hint": "Загрузите файл или нажмите Try demo",
    "try_demo_button": "Try demo",
    "clear_button": "Очистить",
    "generate_button": "Сгенерировать карточки",
    "preview_title_fmt": "📋 Предпросмотр карточек (первые {limit})",
    "csv_download_label": "📥 Скачать anki_cards.csv",
    "apkg_download_label": "🧩 Скачать колоду Anki (.apkg)",
    "apkg_install_hint": "Для экспорта в .apkg добавь в requirements.txt строку 'genanki' и перезагрузи приложение.",

    # Сообщения об ошибках (форматные строки)
    "error_card_processing_fmt": "Ошибка при обработке слова '{woord}': {error}",
    "error_apkg_build_fmt": "Не удалось собрать .apkg: {error}",
}

# ==========================
# Anki: идентификаторы и имена
# ==========================

ANKI_MODEL_ID: int = 1607392319
ANKI_DECK_ID: int = 1970010101
ANKI_MODEL_NAME: str = "Dutch Cloze (L2/L1)"
ANKI_DECK_NAME: str = "Dutch • Cloze"

# ==========================
# Шаблоны Anki (HTML/CSS)
# ==========================

FRONT_HTML_TEMPLATE: str = """
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

BACK_HTML_TEMPLATE: str = """
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

CSS_STYLING: str = """
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

# ==========================
# Функции для работы с конфигурацией
# ==========================

def get_preferred_order() -> Dict[str, int]:
    """Возвращает словарь предпочтительного порядка моделей"""
    return _PREFERRED_ORDER.copy()

def get_block_substrings() -> Tuple[str, ...]:
    """Возвращает кортеж запрещенных подстрок"""
    return _BLOCK_SUBSTRINGS

def get_allowed_prefixes() -> Tuple[str, ...]:
    """Возвращает кортеж разрешенных префиксов"""
    return _ALLOWED_PREFIXES

