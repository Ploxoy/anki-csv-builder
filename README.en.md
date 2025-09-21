# Anki CSV Builder

Streamlit app that turns Dutch vocabulary into ready-to-import Anki decks with help from the OpenAI Responses API.

## Features

- **Automatic CEFR-aligned cloze cards** with sentences, translations, definitions, and collocations
- **Flexible input**: Markdown tables, TSV/CSV, plain text, or the built-in manual editor
- **Word validation with flags** plus an override option when you want to force generation
- **Balanced signal words and separable-verb support**, including deterministic selection driven by a seed
- **Smart model selection** with automatic fallback when `response_format` is not supported
- **CSV and .apkg export** so you can either import or immediately load the deck into Anki
- **Optional TTS** – after generation you can synthesize MP3 for both the word and the sentence (OpenAI TTS) with caching, fallback, and preserved summary logs with selectable voice and style presets (news, learning, radio, etc.)

## Card structure

Each generated note contains the following fields:

- `woord` – target Dutch word
- `cloze_sentence` – Dutch sentence with cloze markup
- `ru_sentence` – sentence translation (default L1 is Russian; other languages are available in the UI)
- `collocaties` – three frequent collocations
- `def_nl` – Dutch definition
- `ru_short` – short gloss in the selected L1

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd anki-csv-builder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the OpenAI API key:
   - create `.streamlit/secrets.toml`, or
   - enter the key in the Streamlit sidebar when the app starts

## Running the app

Preferred entrypoint (new UI):

```bash
streamlit run app/app.py
```

Legacy shim that delegates to the same module (kept for compatibility):

```bash
streamlit run anki_csv_builder.py
```

## Project layout

```
anki-csv-builder/
├── app/
│   └── app.py             # Streamlit UI
├── anki_csv_builder.py    # Compatibility entrypoint that imports app/app.py
├── core/                  # Parsing, generation, sanitisation, export helpers
├── config/                # Settings, templates, signal-word groups, i18n
├── notes/                 # Working notes (status, vision, idea, etc.)
├── requirements.txt       # Dependencies
├── README.md              # Russian documentation
├── README.en.md           # English documentation
└── tests/                 # Sample inputs and unit tests
```

## Configuration

Key settings live in the `config/` package:

- `config/settings.py` – available models, UI defaults, limits, request pacing
- `config/templates_anki.py` – HTML/CSS note templates and Anki model/deck metadata
- `config/signalword_groups.py` – signal-word pools grouped by CEFR level
- `config/i18n.py` – CSV header localisation and L1 labels

## Input formats

### Markdown table
```markdown
| woord    | definitie NL | RU      |
|----------|--------------|---------|
| aanraken | iets voelen  | трогать |
```

### TSV (tab-separated)
```
aanraken	iets voelen	трогать
begrijpen	snappen	понимать
```

### Plain text
```
aanraken - iets voelen - трогать
begrijpen - snappen - понимать
```

## OpenAI API setup

1. Grab an API key from https://platform.openai.com
2. Optional local secrets file:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

## Supported model families

- `gpt-5*` – highest quality
- `gpt-4.1*` – balance of speed and quality
- `gpt-4o*` – faster and cheaper
- `o3*` – reasoning-focused alternatives

## How to use

1. Upload a word list or load the demo dataset
2. Pick the OpenAI model and tweak generation settings
3. Press "Generate" and review the preview
4. Download the CSV or .apkg file and import it into Anki

### Extra tools

- **Manual editor** – the ✍️ tab lets you build or tweak the list before generation.
- **Quality flags** – warnings explain why a word was flagged; enable “Force generate for flagged entries” to process them anyway.
- **Signal-word seed** – keep the same seed to reproduce the connector set across runs.
- **Audio synthesis** – open the “🔊 Audio” panel, pick a voice, choose word/sentence toggles, then generate audio before exporting.
- **Pronunciation styles** – select separate instruction presets for sentences vs. words (newsreader, learning, dictionary, academic).

## Troubleshooting

- **Invalid API key** – double-check the key in `.streamlit/secrets.toml` or the sidebar field.
- **Slow generation** – switch to a faster model like `gpt-4o` or reduce the list size.
- **Schema errors** – the app retries without `response_format`; if issues persist, re-run the item or choose a different model.

## Performance notes

- **Request spacing** – 100 ms between calls (configurable via `config/settings.py`).
- **Preview limit** – the UI shows the first 20 cards; full exports contain every successful item.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Open a pull request

## License

MIT License

## Support

Open an issue if you run into problems or have ideas to discuss.



## 📥 Uploading to Anki

1. Launch Anki Desktop and open the deck where you want to import.
2. File → Import …
   - For CSV: choose `anki_cards.csv`, set Type = Notes (Cloze) and delimiter `|`.
   - For APKG: just select `dutch_cloze.apkg` (creates deck immediately).
3. Confirm the field mapping (L2_word → Cloze).
4. Click `Import` and review the cards in the deck.
