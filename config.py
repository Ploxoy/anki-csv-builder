"""
Configuration file for Anki CSV Builder.
Contains all configurable constants, templates, and demo data.
"""

# ==========================
# OpenAI Models Configuration
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

FALLBACK_MODEL = "gpt-4o-mini"

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

SIGNALWORD_GROUPS = {
    "cause_effect": {
        "A1": ["omdat", "want", "dus"],
        "A2": ["daardoor", "daarom", "zodat"],
        "B1": ["doordat", "waardoor"],
        "B2": ["ten gevolge van", "als gevolg van", "te danken aan", "te wijten aan", "wegens", "vanwege", "aangezien"],
        "C1": ["derhalve", "vandaar dat", "dien ten gevolge", "op grond daarvan"],
        "C2": ["immers", "in zoverre", "zulks omdat", "dit impliceert dat", "uit dien hoofde"]
    },
    "contrast": {
        "A1": ["maar"],
        "A2": ["toch", "of", "ofwel"],
        "B1": ["echter"],
        "B2": ["niettemin", "enerzijds ... anderzijds", "daarentegen", "integendeel", "in tegenstelling tot"],
        "C1": ["desondanks", "nochtans", "hoezeer ook", "ondanks dat"],
        "C2": ["hoe paradoxaal ook", "zij het dat", "al ware het maar", "weliswaar ... maar"]
    },
    "condition_goal": {
        "A1": ["als", "om ... te"],
        "A2": ["wanneer", "tenzij"],
        "B1": ["indien", "mits"],
        "B2": ["opdat", "daartoe", "met als doel", "met behulp van", "door middel van"],
        "C1": ["gesteld dat", "ingeval", "voor zover", "op voorwaarde dat"],
        "C2": ["indien en voorzover", "in de veronderstelling dat", "teneinde", "met het oog op"]
    },
    "example_addition": {
        "A1": ["en", "ook"],
        "A2": ["bijvoorbeeld", "zoals"],
        "B1": ["verder", "bovendien"],
        "B2": ["eveneens", "zowel ... als", "daarnaast", "ten slotte", "onder andere", "ter illustratie", "ter verduidelijking"],
        "C1": ["neem nu", "stel dat", "dat wil zeggen", "met name"],
        "C2": ["te weten", "als zodanig", "zulks ter illustratie", "onder meer ... doch niet uitsluitend"]
    },
    "comparison": {
        "A1": ["zoals", "net als"],
        "A2": ["hetzelfde als"],
        "B1": ["evenals"],
        "B2": ["in vergelijking met", "vergeleken met"],
        "C1": ["analoge wijze", "op gelijke wijze", "evenzeer"],
        "C2": ["mutatis mutandis", "naar analogie van"]
    },
    "summary_conclusion": {
        "A1": ["dus"],
        "A2": ["daarom"],
        "B1": ["kortom"],
        "B2": ["uiteindelijk", "samenvattend", "concluderend", "hieruit volgt", "met andere woorden", "al met al"],
        "C1": ["alles overziend", "alles bijeen genomen", "resumerend"],
        "C2": ["derhalve concluderen wij dat", "dit leidt onvermijdelijk tot de slotsom dat"]
    }
}


PROMPT_PROFILES: Dict[str, str] = {
    "strict": "Be literal and concise; avoid figurative language; keep the simplest structure that satisfies CEFR.",
    "balanced": "Natural and clear; minor synonymy allowed if it improves fluency.",
    "exam": "Neutral-formal register; precise; avoid colloquialisms.",
    "creative": "Allow mild figurativeness if it keeps clarity and CEFR constraints.",
}

# ==========================
# Temperature Configuration
# ==========================

DEFAULT_TEMPERATURE = 0.3
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

# ==========================
# CEFR Levels Configuration
# ==========================

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
DEFAULT_CEFR_LEVEL = "B1"

# ==========================
# Language Configuration
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
# CSV Export Configuration
# ==========================

CSV_DELIMITER = "|"  # Pipe delimiter for Anki compatibility

# ==========================
# Validation Rules
# ==========================

MAX_COLLOCATIONS = 3
MIN_COLLOCATIONS = 3
MAX_L1_GLOSS_WORDS = 2

# ==========================
# Anki Template Configuration
# ==========================

# HTML templates for front and back of flashcards
ANKI_FRONT_TEMPLATE = """
<!-- Front template with cloze deletion and collocations -->
<div class="card">
    <!-- ...existing code... -->
</div>
"""

ANKI_BACK_TEMPLATE = """
<!-- Back template showing answer with translation and definition -->
<div class="card">
    <!-- ...existing code... -->
</div>
"""

ANKI_CSS = """
/* CSS styling for flashcards with responsive design */
.card {
    /* ...existing code... */
}
"""

# ==========================
# Demo Data
# ==========================

# Sample input data for testing and demonstration
DEMO_DATA = {
    "simple": """opruimen
aanraken
bedanken""",
    
    "with_context": """opruimen | Ik ga mijn kamer opruimen voordat mijn ouders thuiskomen.
aanraken | Je mag de schilderijen in het museum niet aanraken.
bedanken | Ik wil je bedanken voor je hulp met mijn huiswerk.""",
    
    "markdown_table": """| Dutch Word | Context |
|------------|---------|
| opruimen | Ik ga mijn kamer opruimen voordat mijn ouders thuiskomen. |
| aanraken | Je mag de schilderijen in het museum niet aanraken. |
| bedanken | Ik wil je bedanken voor je hulp met mijn huiswerk. |"""
}

# ==========================
# System Messages
# ==========================

# Error messages and user feedback text
MESSAGES = {
    "api_error": "Error calling OpenAI API: {error}",
    "json_parse_error": "Could not parse JSON response",
    "validation_error": "Card validation failed: {error}",
    "repair_failed": "Failed to repair invalid card",
    "export_success": "Export completed successfully",
    "no_cards_generated": "No valid cards were generated",
}

# ==========================
# Regular Expressions
# ==========================

# Patterns for text processing and validation
REGEX_PATTERNS = {
    "json_extraction": r"\{[\s\S]*\}",  # Extract JSON from response
    "cloze_marker": r"\{\{c\d+::[^}]+\}\}",  # Validate cloze deletion format
    "single_brace": r'\{(?![\{])',  # Find single opening braces
    "single_brace_close": r'(?<![}])\}',  # Find single closing braces
}

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
    <div class="section l1">{{L1_sentence}}</div>
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
      <span class="lemma-nl">{{L2_word}}</span> — <span class="lemma-l1">{{L1_gloss}}</span>
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

