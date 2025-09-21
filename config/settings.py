"""
Configuration for Anki CSV Builder
"""

from typing import List, Dict, Tuple

# ==========================
# Models: default list + dynamic loading from API
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

_PREFERRED_ORDER: Dict[str, int] = {  # lower number = higher in list
    "gpt-5": 0,
    "gpt-5-mini": 1,
    "gpt-5-nano": 2,
    "gpt-4.1": 3,
    "gpt-4o": 4,
    "gpt-4o-mini": 5,
    "o3": 6,
    "o3-mini": 7,
}

# Models to exclude by substring in ID (we need text generators only)
_BLOCK_SUBSTRINGS: Tuple[str, ...] = (
    "audio", "realtime",           # gpt-4o-audio-*, gpt-4o-realtime-*
    "embed", "embedding",          # text-embedding-*
    "whisper", "asr", "transcribe", "speech", "tts",  # ASR/TTS
    "moderation",                  # moderation
    "search",                      # search/auxiliary
    "vision", "vision-preview",    # vision-only/preview
    "distill", "distilled",        # distilled specialized models
    "batch", "preview"             # service/preview/batch models
)

# Allowed families (by prefix) for text generation
_ALLOWED_PREFIXES: Tuple[str, ...] = ("gpt-5", "gpt-4.1", "gpt-4o", "o3")

# ==========================
# Signal words and prompt profiles
# ==========================

SIGNALWORDS_B1: List[str] = [
    "omdat", "maar", "dus", "want", "terwijl", "daarom", "daardoor", "toch"
]

SIGNALWORDS_B2_PLUS: List[str] = [
    "hoewel", "zodat", "doordat", "bovendien", "echter", "bijvoorbeeld", "tenzij", "ondanks",
    "desondanks", "daarentegen", "aangezien", "zodra", "voordat", "nadat", "enerzijds ... anderzijds",
    "niet alleen ... maar ook", "opdat"
]


from config.signalword_groups import SIGNALWORD_GROUPS
PROMPT_PROFILES: Dict[str, str] = {
    "strict": "Be literal and concise; avoid figurative language; keep the simplest structure that satisfies CEFR.",
    "balanced": "Natural and clear; minor synonymy allowed if it improves fluency.",
    "exam": "Neutral-formal register; precise; avoid colloquialisms.",
    "creative": "Allow mild figurativeness if it keeps clarity and CEFR constraints.",
}

# CEFR level rules for prompts
LEVEL_RULES_EN: Dict[str, str] = {
    "A1": "6–9 words; no subclauses; no passive; no perfect.",
    "A2": "8–12 words; may use modal verbs; simple past allowed; no complex clauses.",
    "B1": "10–14 words; simple subclause allowed (omdat/als/terwijl); ~50% with one signal word.",
    "B2": "12–16 words; complex allowed; passive allowed; ~50% with one signal word (extended list).",
    "C1": "14–18 words; advanced structures; neutral‑formal.",
    "C2": "No length limit; native‑like precision.",
}

# ==========================
# L1 languages and CSV header localizations
# ==========================

L1_LANGS: Dict[str, Dict[str, str]] = {
    "RU": {"label": "RU", "name": "Russian", "csv_translation": "Перевод", "csv_gloss": "Перевод слова"},
    "EN": {"label": "EN", "name": "English", "csv_translation": "Translation", "csv_gloss": "Word gloss"},
    "ES": {"label": "ES", "name": "Spanish", "csv_translation": "Traducción", "csv_gloss": "Glosa"},
    "DE": {"label": "DE", "name": "German", "csv_translation": "Übersetzung", "csv_gloss": "Kurzgloss"},
}

# For compatibility if separate map expected elsewhere
CSV_HEADERS_LOCALIZATION = L1_LANGS

# Fixed parts of CSV headers (NL columns)
CSV_HEADERS_FIXED: Dict[str, str] = {
    "nl_word": "Dutch Word",
    "nl_sentence_cloze": "Dutch Sentence (with cloze)",
    "collocations_nl": "Collocations (Dutch)",
    "definition_nl": "Definition (Dutch)",
}

# ==========================
# Demo data
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
# UI settings
# ==========================

PAGE_TITLE: str = "Anki CSV/Anki Builder — Dutch Cloze Cards"
PAGE_LAYOUT: str = "wide"

# Temperature slider settings
TEMPERATURE_MIN: float = 0.2
TEMPERATURE_MAX: float = 0.8
TEMPERATURE_DEFAULT: float = 0.4
TEMPERATURE_STEP: float = 0.1

# CSV settings
CSV_DELIMITER: str = '|'
CSV_LINETERMINATOR: str = '\n'

# Preview settings
PREVIEW_LIMIT: int = 20

# Delay between API requests (in seconds)
API_REQUEST_DELAY: float = 0.1

# ==========================
# Audio / TTS settings
# ==========================

AUDIO_TTS_MODEL: str = "gpt-4o-mini-tts"
AUDIO_TTS_FALLBACK: str | None = "gpt-4o-tts"
AUDIO_VOICES: List[Dict[str, str]] = [
    {"id": "ash", "label": "Ash — NL male (neutral)"},
    {"id": "verse", "label": "Verse — NL male"},
    {"id": "alloy", "label": "Alloy — NL female"},
    {"id": "shimmer", "label": "Shimmer — energetic NL female"},
    {"id": "ballad", "label": "Ballad — soft NL female"},
]
AUDIO_TTS_INSTRUCTIONS: Dict[str, str] = {
    "Dutch_sentence_news": (
        "Speak in Dutch (nl-NL). Use a clear Randstad Dutch accent, "
        "as in NOS news broadcasts. Keep intonation natural and flowing, "
        "moderate speed, no raspiness."
    ),
    "Dutch_sentence_learning": (
        "Speak Dutch (nl-NL) with a neutral, standard accent, "
        "suitable for language learners. Keep tempo slightly slower than normal, "
        "articulate clearly, no Flemish influence."
    ),
    "Dutch_sentence_radio": (
        "Read in Dutch (nl-NL) with a standard accent as used in national radio. "
        "Use warm, clear tone, no rasp, steady rhythm."
    ),
    "Dutch_word_dictionary": (
        "Pronounce in Dutch (nl-NL) with standard dictionary pronunciation. "
        "Say the single word only, clear and clean, no added intonation."
    ),
    "Dutch_word_learning": (
        "Speak this word in Dutch (nl-NL), with careful articulation, neutral accent. "
        "Produce only the word, no surrounding sounds or rasp."
    ),
    "Dutch_word_academic": (
        "Give the single word in Dutch (nl-NL), with exact phonetic accuracy. "
        "Neutral Dutch pronunciation, no Flemish accent, no raspiness."
    ),
}
AUDIO_SENTENCE_INSTRUCTION_DEFAULT: str = "Dutch_sentence_learning"
AUDIO_WORD_INSTRUCTION_DEFAULT: str = "Dutch_word_dictionary"
AUDIO_INCLUDE_WORD_DEFAULT: bool = True
AUDIO_INCLUDE_SENTENCE_DEFAULT: bool = True

# ==========================
# CSV headers
# ==========================

CSV_HEADERS: List[str] = [
    "Dutch Word",
    "Dutch Sentence (with cloze)",
    "Translation",
    "Collocations",
    "Dutch Definition",
    "Word Gloss",
]

# ==========================
# Messages and hints
# ==========================

MESSAGES = {
    # General hints
    "demo_loaded": "🔁 Demo set of 6 words loaded",
    "no_api_key": "Please provide OPENAI_API_KEY in Secrets or in the field on the left.",
    "temperature_unavailable": "Temperature unavailable for this model; it will be ignored.",
    "help_temperature": "Best quality — gpt-5 (if available); balanced — gpt-4.1; faster/cheaper — gpt-4o / gpt-5-mini.",
    "help_stream": "Streaming in Responses API: final JSON will be available after stream completion",
    "placeholder_custom_model": "e.g., gpt-5-2025-08-07",
    "footer_tips": (
        "Tips: 1) Better NL definitions on input = more accurate examples and glosses. "
        "2) At B1+ levels, roughly half the sentences will include signaalwoorden. "
        "3) Some models (gpt-5/o3) don't support temperature and will ignore it."
    ),

    # UI texts
    "app_title": "📘 Anki CSV/Anki Builder — Dutch Cloze Cards",
    "sidebar_api_header": "🔐 API Settings",
    "api_key_label": "OpenAI API Key",
    "model_label": "Model",
    "model_help": "Best quality — gpt-5 (if available); balanced — gpt-4.1; faster/cheaper — gpt-4o / gpt-5-mini.",
    "profile_label": "Prompt profile",
    "cefr_label": "CEFR Level",
    "l1_label": "Your language (L1)",
    "temp_label": "Temperature",
    "csv_header_checkbox": "CSV: include header row",
    "csv_header_help": "Uncheck if Anki imports the first row as a record.",
    "anki_guid_policy_label": "Anki GUID policy",
    "anki_guid_policy_options": [
        "stable (update/skip existing)",
        "unique per export (import as new)"
    ],
    "anki_guid_policy_help": (
        "stable: same notes are recognized as existing/updatable\n"
        "unique: each export gets a new GUID — Anki treats them as new notes."
    ),
    "uploader_label": "Upload .txt / .md",
    "recognized_rows_title": "🔍 Recognized rows",
    "upload_hint": "Upload a file or click Try demo",
    "try_demo_button": "Try demo",
    "clear_button": "Clear",
    "generate_button": "Generate cards",
    "preview_title_fmt": "📋 Card preview (first {limit})",
    "csv_download_label": "📥 Download anki_cards.csv",
    "apkg_download_label": "🧩 Download Anki deck (.apkg)",
    "apkg_install_hint": "To export to .apkg, add 'genanki' to requirements.txt and redeploy the app.",

    # Error messages (format strings)
    "error_card_processing_fmt": "Error processing word '{woord}': {error}",
    "error_apkg_build_fmt": "Failed to build .apkg: {error}",
}

# ==========================
# Anki: identifiers and names
# ==========================

ANKI_MODEL_ID: int = 1607392319
ANKI_DECK_ID: int = 1970010101
ANKI_MODEL_NAME: str = "Dutch Cloze (L2/L1)"
ANKI_DECK_NAME: str = "Dutch • Cloze"

# ==========================
# Anki templates (HTML/CSS)
# ==========================

FRONT_HTML_TEMPLATE: str = """
<div class="card-inner">
  {{cloze:L2_cloze}}
  <div class="audio-inline">
    {{#AudioSentence}}
    <span class="audio-icon">{{AudioSentence}}</span>
    {{/AudioSentence}}
  </div>
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
    <div class="section l1">
      {{L1_sentence}}
      {{#AudioSentence}}
      <span class="audio-icon">{{AudioSentence}}</span>
      {{/AudioSentence}}
    </div>
    {{/L1_sentence}}

    {{#L2_collocations}}
    <div class="section">
      <div class="colloc-container"></div>
      <script>
        (function () {
          var collocText = "{{L2_collocations}}";
          if (!collocText || collocText.trim() === "") return;
          
          var items = collocText.split(';').map(function (s) { return s.trim(); }).filter(function(s) { return s.length > 0; });
          var container = document.querySelector('.colloc-container');
          if (!container || items.length === 0) return;
          
          var ul = document.createElement('ul');
          ul.className = 'colloc';
          
          for (var i = 0; i < items.length; i++) {
            var li = document.createElement('li');
            li.textContent = items[i];
            ul.appendChild(li);
          }
          
          container.appendChild(ul);
        })();
      </script>
    </div>
    {{/L2_collocations}}

    {{#L2_definition}}
    <div class="section def">{{L2_definition}}</div>
    {{/L2_definition}}

    {{#L2_word}}
    <div class="section lemma">
      <span class="lemma-nl">{{L2_word}}</span>
      {{#AudioWord}}
      <span class="audio-icon">{{AudioWord}}</span>
      {{/AudioWord}}
      {{#L1_gloss}}
      <span class="lemma-l1">— {{L1_gloss}}</span>
      {{/L1_gloss}}
    </div>
    {{/L2_word}}
  </div>
</div>
""".strip()

CSS_STYLING: str = """
/* ===== Scaling and layout ===== */
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
.audio-inline{ margin-top:.4em; }
.audio-icon{ display:inline-flex; align-items:center; justify-content:center; margin-left:.35em; }
.audio-icon audio{ display:inline-block; height:26px; width:140px; vertical-align:middle; }
.lemma .audio-icon audio{ height:24px; width:120px; }
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
# Configuration utility functions
# ==========================

def get_preferred_order() -> Dict[str, int]:
    """Returns dictionary of preferred model ordering"""
    return _PREFERRED_ORDER.copy()

def get_block_substrings() -> Tuple[str, ...]:
    """Returns tuple of blocked substrings"""
    return _BLOCK_SUBSTRINGS

def get_allowed_prefixes() -> Tuple[str, ...]:
    """Returns tuple of allowed prefixes"""
    return _ALLOWED_PREFIXES
