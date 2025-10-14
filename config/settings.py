"""
Configuration for Anki CSV Builder
"""

import os

from typing import Any, Dict, List, Tuple

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
    #{"woord": "kloppen", "def_nl": "met regelmaat bonzen of tikken"},
    #{"woord": "toestaan", "def_nl": "goedkeuren of laten gebeuren"},
    #{"woord": "opruimen", "def_nl": "iets netjes maken door het op zijn plaats te leggen"},
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
    # Sentences
    "Dutch_sentence_news": (
        "Speak Dutch (nl-NL, Netherlands). Neutral Randstad accent. "
        "Natural, flowing prosody at normal pace. Do not translate or explain. "
        "No Flemish influence, no English coloring, no spelling of words."
    ),
    "Dutch_sentence_learning": (
        "Speak Dutch (nl-NL, Netherlands) with a neutral standard accent, "
        "slightly slower than normal for learners, very clear articulation. "
        "Do not translate, spell letters, or add explanations."
    ),
    "Dutch_sentence_radio": (
        "Read in Dutch (nl-NL) with warm, clear tone similar to national radio. "
        "Steady rhythm, no raspiness, no Flemish accent, no explanations."
    ),
    # Single words
    "Dutch_word_dictionary": (
        "Pronounce in Dutch (nl-NL, Netherlands). Produce exactly the single word only — "
        "no translation, no letter spelling, no punctuation, no extra words. "
        "Avoid English or Flemish accent; use neutral Dutch dictionary pronunciation."
    ),
    "Dutch_word_learning": (
        "Say the single word in Dutch (nl-NL) with careful, clean articulation. "
        "Output only the word itself — do not spell letters, do not translate, no extra sounds."
    ),
    "Dutch_word_academic": (
        "Pronounce the single word in Dutch (nl-NL) with precise, neutral pronunciation. "
        "No Flemish or English accent; do not add anything besides the word."
    ),
}
AUDIO_SENTENCE_INSTRUCTION_DEFAULT: str = "Dutch_sentence_learning"
AUDIO_WORD_INSTRUCTION_DEFAULT: str = "Dutch_word_dictionary"
AUDIO_INCLUDE_WORD_DEFAULT: bool = True
AUDIO_INCLUDE_SENTENCE_DEFAULT: bool = True

ELEVENLABS_DEFAULT_API_KEY: str = os.environ.get("ELEVENLABS_API_KEY", "")


def _style_label_from_key(key: str) -> str:
    if key.startswith("Dutch_sentence_"):
        suffix = key.split("Dutch_sentence_", 1)[1].replace("_", " ")
        return f"Sentence · {suffix.capitalize()}"
    if key.startswith("Dutch_word_"):
        suffix = key.split("Dutch_word_", 1)[1].replace("_", " ")
        return f"Word · {suffix.capitalize()}"
    if key.startswith("Eleven_sentence_"):
        suffix = key.split("Eleven_sentence_", 1)[1].replace("_", " ")
        return f"Sentence · {suffix.capitalize()}"
    if key.startswith("Eleven_word_"):
        suffix = key.split("Eleven_word_", 1)[1].replace("_", " ")
        return f"Word · {suffix.capitalize()}"
    return key


AUDIO_ELEVEN_VOICES: List[Dict[str, str]] = [
    {"id": "21m00Tcm4TlvDq8ikWAM", "label": "Rachel — balanced NL"},
    {"id": "AZnzlk1XvdvUeBnXmlld", "label": "Domi — calm feminine"},
    {"id": "ErXwobaYiN019PkySvjV", "label": "Antoni — clear masculine"},
    {"id": "EXAVITQu4vr4xnSDxMaL", "label": "Bella — bright feminine"},
    # Extra multilingual presets that perform well even without explicit NL label
    {"id": "N2lVS1w4EtoT3dr4eOWO", "label": "Callum — newsy male"},
    {"id": "XB0fDUnXU5powFXDhCwa", "label": "Charlotte — warm neutral"},
    {"id": "XrExE9yKIg1WjnnlVkGX", "label": "Matilda — clear neutral"},
    {"id": "ZQe5CZNOzWyzPSCn5a3c", "label": "James — neutral male"},
    {"id": "piTKgcLEGmPE4e6mEKli", "label": "Nicole — soft neutral"},
    {"id": "ODq5zmih8GrVes37Dizd", "label": "Patrick — deep male"},
    {"id": "SOYHLrjzK2X1ezoPC6cr", "label": "Harry — friendly male"},
    {"id": "Zlb1dXrM653N07WRdFW3", "label": "Joseph — warm baritone"},
]

AUDIO_ELEVEN_STYLES: Dict[str, Dict[str, Any]] = {
    "sentence": {
        "Eleven_sentence_tutor": {
            "label": _style_label_from_key("Eleven_sentence_tutor"),
            "description": "Neutral Dutch tutor voice. Calm delivery, clear articulation (style 0.35).",
            "payload": {
                "voice_settings": {
                    "stability": 0.55,
                    "similarity_boost": 0.85,
                    "style": 0.35,
                    "use_speaker_boost": True,
                },
                "spoken_language": "nl",
            },
        },
        "Eleven_sentence_radio": {
            "label": _style_label_from_key("Eleven_sentence_radio"),
            "description": "Lively radio-style narration with more energy (style 0.6).",
            "payload": {
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.8,
                    "style": 0.6,
                    "use_speaker_boost": True,
                },
                "spoken_language": "nl",
            },
        },
        "Eleven_sentence_story": {
            "label": _style_label_from_key("Eleven_sentence_story"),
            "description": "Narrative storytelling tone with gentle emphasis (style 0.45).",
            "payload": {
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.45,
                    "use_speaker_boost": False,
                },
                "spoken_language": "nl",
            },
        },
    },
    "word": {
        "Eleven_word_dictionary": {
            "label": _style_label_from_key("Eleven_word_dictionary"),
            "description": "Dictionary-style single word: clean and precise (style 0.15).",
            "payload": {
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.9,
                    "style": 0.15,
                    "use_speaker_boost": False,
                },
                "spoken_language": "nl",
            },
        },
        "Eleven_word_learning": {
            "label": _style_label_from_key("Eleven_word_learning"),
            "description": "Language-learning focus: slightly slower, very clear (style 0.25).",
            "payload": {
                "voice_settings": {
                    "stability": 0.65,
                    "similarity_boost": 0.88,
                    "style": 0.25,
                    "use_speaker_boost": False,
                },
                "spoken_language": "nl",
            },
        },
        "Eleven_word_dynamic": {
            "label": _style_label_from_key("Eleven_word_dynamic"),
            "description": "Dynamic pronunciation with slight emphasis (style 0.45).",
            "payload": {
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.82,
                    "style": 0.45,
                    "use_speaker_boost": True,
                },
                "spoken_language": "nl",
            },
        },
    },
}


def _build_instruction_styles(prefix: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for key, text in AUDIO_TTS_INSTRUCTIONS.items():
        if not key.startswith(prefix):
            continue
        result[key] = {
            "label": _style_label_from_key(key),
            "description": text,
            "payload": {"instructions": text},
        }
    return result


AUDIO_PROVIDER_DEFAULT: str = "openai"
AUDIO_TTS_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "type": "openai",
        "model": AUDIO_TTS_MODEL,
        "fallback_model": AUDIO_TTS_FALLBACK,
        "voices": AUDIO_VOICES,
        "voice_default": AUDIO_VOICES[0]["id"] if AUDIO_VOICES else "",
        "include_word_default": AUDIO_INCLUDE_WORD_DEFAULT,
        "include_sentence_default": AUDIO_INCLUDE_SENTENCE_DEFAULT,
        "sentence_styles": _build_instruction_styles("Dutch_sentence_"),
        "word_styles": _build_instruction_styles("Dutch_word_"),
        "sentence_default": AUDIO_SENTENCE_INSTRUCTION_DEFAULT,
        "word_default": AUDIO_WORD_INSTRUCTION_DEFAULT,
    },
    "elevenlabs": {
        "label": "ElevenLabs",
        "type": "elevenlabs",
        "model": "eleven_multilingual_v2",
        "fallback_model": None,
        "voices": AUDIO_ELEVEN_VOICES,
        "voice_default": AUDIO_ELEVEN_VOICES[0]["id"] if AUDIO_ELEVEN_VOICES else "",
        "include_word_default": True,
        "include_sentence_default": True,
        "sentence_styles": AUDIO_ELEVEN_STYLES["sentence"],
        "word_styles": AUDIO_ELEVEN_STYLES["word"],
        "sentence_default": "Eleven_sentence_tutor",
        "word_default": "Eleven_word_dictionary",
        "dynamic_voices": False,
        "voice_language_codes": ["nl"],
    },
}

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
ANKI_DECK_NAME: str = "Dutch"

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
/* ===== Minimal, unified styling for all cards ===== */
:root{
  --fs-base: clamp(18px, 1.2vw + 1.1vh, 28px);
  --fs-sm: calc(var(--fs-base) * .9);
  --fs-lg: calc(var(--fs-base) * 1.12);
  --hl-col:#1976d2;
  --hl-bg:rgba(25,118,210,.14);
}
.card{ font-size: var(--fs-base); line-height: 1.55; margin:0; background: transparent; }
.card-inner{ width: min(92vw, 80ch); margin: 2.5vh auto; padding: 0 3vw; }
.answer { margin-top:.75em; }
.audio-inline{ margin-top:.4em; }
.audio-icon{ display:inline-flex; align-items:center; justify-content:center; margin-left:.35em; }
.audio-icon audio{ display:inline-block; height:26px; width:140px; vertical-align:middle; }
.lemma .audio-icon audio{ height:24px; width:120px; }
.section + .section { margin-top:.55em; padding-top:.45em; border-top:1px solid rgba(0,0,0,.14); }
@media (prefers-color-scheme: dark){ .section + .section { border-top-color: rgba(255,255,255,.22); } }
.def { font-style: italic; opacity:.9; font-size: var(--fs-sm); }
.lemma { font-weight:600; }
.lemma-nl{ color:var(--hl-col); font-variant: small-caps; letter-spacing:.02em; }
.colloc{ margin:.1em 0 0 1.1em; padding:0; }
.colloc li{ margin:.12em 0; }
.cloze{ color:var(--hl-col); font-weight:700; }
mark.hl{ background:var(--hl-bg); color:inherit; padding:0 .12em; border-radius:.18em; }
img{ max-width:100%; height:auto; }
@media (max-width: 420px){ .card-inner{ width: 94vw; padding: 0 3vw; } }
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
