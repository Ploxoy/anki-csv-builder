# Anki CSV Builder - AI Agent Instructions

## Project Overview
This is a Streamlit app that generates Dutch language learning flashcards for Anki using OpenAI's API. It creates cloze deletion cards with translations, definitions, and collocations.

## Key Architecture Patterns

### Configuration-First Design
- `config.py` contains all configurable constants (models, templates, messages, demo data)
- `anki_csv_builder.py` imports from config with comprehensive fallbacks
- Always update config.py for new constants rather than hardcoding values

### Prompt Engineering Pipeline
- `prompts.py` handles CEFR-level specific instruction generation
- Uses `compose_instructions_en()` with dynamic L1 language support
- Signal words (signaalwoorden) are deterministically included based on CEFR level (50% for B1+)
- Cloze format must use exactly double curly braces: `{{c1::word}}` for regular, `{{c1::stem}} ... {{c2::particle}}` for separable verbs

### OpenAI API Integration
- Uses Responses API (`client.responses.create()`) not Chat API
- Temperature support is model-dependent (gpt-5/o3 family don't support it)
- Implements automatic retry without temperature on unsupported models
- JSON extraction uses regex: `r"\{[\s\S]*\}"` to handle wrapped responses

### Data Flow
1. Input parsing supports: plain words, TSV, markdown tables, em-dash format
2. OpenAI call → JSON extraction → sanitization → validation → optional repair
3. Export to CSV (pipe-delimited) and/or Anki .apkg via genanki

## Critical Functions

### sanitize()
Handles cloze marker escaping and pipe character replacement:
```python
s = re.sub(r'\{(?![\{])', '{{', s)  # Single { -> {{
s = re.sub(r'(?<![}])\}', '}}', s)  # Single } -> }}
```

### validate_card()
Enforces business rules: exactly 3 collocations, max 2 words for L1_gloss, required cloze markers

## Anki Template System
- Front/back HTML templates in config with JavaScript for collocation rendering
- Templates use Anki's template syntax: `{{field_name}}` for fields, `{{#field}}{{/field}}` for conditionals
- CSS uses CSS custom properties for responsive design

## Development Patterns
- Use session_state for persistent data across Streamlit reruns
- L1_code determines UI language and field names dynamically
- Debug mode available via sidebar checkbox for troubleshooting API responses
- Model filtering: only text-generation models, exclude audio/vision/embedding variants

## File Responsibilities
- `anki_csv_builder.py`: Main Streamlit app, UI, API calls, data processing
- `config.py`: All configuration, templates, constants, demo data
- `prompts.py`: CEFR-aware prompt generation, language-specific instructions

When modifying templates or prompts, test with separable verbs (like "opruimen") to ensure proper cloze handling.
