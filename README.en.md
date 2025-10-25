# 📘 Anki CSV Builder

Streamlit app that turns Dutch vocabulary into ready-to-import Anki decks with help from the OpenAI Responses API.

## Features

- **Automatic CEFR-aligned cloze cards** with sentences, translations, definitions, and collocations
- **Flexible input**: Markdown tables, TSV/CSV, plain text, or the built-in manual editor
- **Word validation with flags** plus an override option when you want to force generation
- **Balanced signal words and separable-verb support**, including deterministic selection driven by a seed
- **Smart model selection** with automatic fallback when `response_format` is not supported
- **CSV and .apkg export** with optional Basic / Type In subdecks that share the same styling and audio attachments
- **Optional TTS** — synthesize MP3 for both the word and the sentence (OpenAI TTS and ElevenLabs) with caching, retries, and per-card voice mapping

## UI flow

1. **Generate** — launch batch generation with live progress
2. **Preview & fix** — inspect cards, flagged rows, and quick fixes
3. **Audio (optional)** — pick provider, voice, and run TTS; exports pick up audio automatically
4. **Export deck** — download CSV and/or `.apkg`, including the extra subdecks if enabled

## Card structure

Each generated note includes:

- `woord` — target Dutch word
- `cloze_sentence` — Dutch sentence with cloze markup
- `ru_sentence` — sentence translation (UI lets you pick other L1 languages)
- `collocaties` — three frequent collocations
- `def_nl` — Dutch definition
- `ru_short` — short gloss in the selected L1

## Installation

```bash
git clone <repository-url>
cd anki-csv-builder
pip install -r requirements.txt
```

## Running the app

Preferred entrypoint:

```bash
streamlit run app/app.py
```

Legacy shim (kept for compatibility):

```bash
streamlit run anki_csv_builder.py
```

## Project layout

```
anki-csv-builder/
├── app/                 # Streamlit UI modules
├── core/                # Parsing, generation, sanitisation, export helpers
├── config/              # Settings, templates, signal-word groups, i18n
├── notes/               # Project status, vision, specs
├── tests/               # Unit tests and sample inputs
├── README.md            # Russian documentation
├── README.en.md         # English documentation
└── requirements.txt     # Dependencies
```

## Configuration

- `config/settings.py` — available models, UI defaults, pacing, template paths
- `config/templates/` — HTML/CSS templates for Cloze, Basic, and Type In decks
- `config/signalword_groups.py` — signal-word pools grouped by CEFR level

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

## OpenAI & ElevenLabs API setup

1. Create API keys at https://platform.openai.com and https://elevenlabs.io (optional).
2. Store them in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-openai-key"
ELEVENLABS_API_KEY = "your-elevenlabs-key"
```

…or enter them in the Streamlit sidebar when the app is running.

## Supported model families

- `gpt-5*` — highest quality
- `gpt-4.1*` — balance of speed and quality
- `gpt-4o*` — faster and cheaper
- `o3*` — reasoning-focused alternatives

## How to use

1. Upload a word list or load the demo dataset.
2. Pick the OpenAI model and adjust generation settings.
3. Press **Generate**, watch the progress, and review the preview.
4. (Optional) Open **Audio** to synthesise word/sentence MP3.
5. Download the CSV or `.apkg` file and import it into Anki.

### Extra tools

- **Manual editor** — build or tweak the list before generation.
- **Quality flags** — see why a word was flagged; enable “Force generate for flagged entries” to include it anyway.
- **Signal-word seed** — keep the same seed to reuse connector choices.
- **Audio presets** — switch providers, voices, and instruction presets for sentences vs. words.
- **Random voice per card** — optional mapping that keeps a consistent voice within each card.

## Uploading to Anki

1. Launch Anki Desktop and open the target deck.
2. File → Import …
   - For CSV: choose `anki_cards.csv`, set Type = Notes (Cloze) and delimiter `|`.
   - For APKG: select `dutch_cloze.apkg` (creates the deck immediately).
3. Confirm the field mapping (`L2_word` → Cloze field).
4. Click **Import** and review the cards.

## Troubleshooting

- **Invalid API key** — double-check the key in `.streamlit/secrets.toml` or the sidebar field.
- **Slow generation** — switch to a faster model like `gpt-4o` or reduce the batch size.
- **Schema errors** — the app retries without `response_format`; if issues persist, re-run the item or choose a different model.
- **No ElevenLabs voices** — use the refresh button in the audio panel or fall back to curated presets.

## Performance notes

- Default request spacing: 100 ms between API calls (configurable in `config/settings.py`).
- Preview shows the first 20 cards; exports contain every successful item.
- TTS workers automatically respect ElevenLabs rate limits (≤2 concurrent requests).

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes and add tests when possible.
4. Open a pull request.

## License

MIT License.

## Support

Open an issue if you run into problems or have ideas to discuss.

## Changelog highlights

See `notes/status.md` for the full status log. Recent updates:

- Audio panel rebuilt with OpenAI + ElevenLabs support, presets, and detailed run summaries.
- Anki templates moved to `config/templates/*` and loaded lazily during export.
- `.apkg` export now bundles Cloze, Basic (reversed), and Type In subdecks with shared styling and audio hooks.
