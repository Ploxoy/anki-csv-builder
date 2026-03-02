# 📘 Anki CSV Builder

FastAPI service for generating Dutch Anki cards (generation + TTS) powered by the OpenAI Responses API.

The Streamlit UI is considered legacy and is being phased out.

## Features

- **Automatic CEFR-aligned cloze cards** with sentences, translations, definitions, and collocations
- **Flexible input**: Markdown tables, TSV/CSV, plain text, or the built-in manual editor
- **Word validation with flags** plus an override option when you want to force generation
- **Balanced signal words and separable-verb support**, including deterministic selection driven by a seed
- **Smart model selection** with automatic fallback when `response_format` is not supported
- **CSV and .apkg export** with optional Basic / Type In subdecks that share the same styling and audio attachments
- **Optional TTS** — synthesize MP3 for both the word and the sentence (OpenAI TTS and ElevenLabs) with caching, retries, and per-card voice mapping
- **Persistent TTS cache** — every synthesized MP3 is stored under `cache/audio/`, so already-paid audio survives reruns and quotas; reruns reuse the cached files automatically
- **Schema-aware generation** — probes model `response_format` support once, caches the result, and falls back to text parsing only when required
- **Completion diagnostics** — Run report now includes raw-vs-final character counts per model so runaway outputs (e.g., overly verbose GPT-5 runs) stand out immediately
- **Run report 2.0** — per-model token usage, repair/fallback share, and cost estimates (text + TTS) with downloadable JSON

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

## Running the app (legacy Streamlit UI)

Preferred entrypoint:

```bash
streamlit run app/app.py
```

Legacy shim (kept for compatibility):

```bash
streamlit run anki_csv_builder.py
```

## Running the API (FastAPI)

Locally (from repo root):

```bash
export API_SHARED_SECRET="change-me"
export OPENAI_API_KEY="..."
# Optional (only if you use ElevenLabs TTS)
export ELEVENLABS_API_KEY="..."
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

For local development you can temporarily disable the `X-API-Key` check:

```bash
export API_REQUIRE_SHARED_SECRET=0
```

### Phase 0.5: user settings + usage (beta)

Multi-user beta uses **invite tokens**:
- Admin creates an invite via `/api/admin/invite` (requires `X-API-Key: API_SHARED_SECRET`) and receives a `token`.
- User pastes the `token` into the web UI (`web/`); requests are sent with `Authorization: Bearer <token>`.

Example (create invite):
```bash
export API_SHARED_SECRET="change-me"
curl -sS -X POST http://localhost:8000/api/admin/invite \
  -H "X-API-Key: $API_SHARED_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"label":"alice"}'
```

Admin endpoints (beta):
- List users: `GET /api/admin/users`
- Block/unblock: `POST /api/admin/users/{user_id}/status` with `{ "status": "blocked|active" }`
- Rotate token: `POST /api/admin/users/{user_id}/rotate` → returns new token

If `Load/Save settings` “does not persist”, it's usually one of:
- your `api` container image was built without `psycopg` (rebuild after dependency changes)
- the `db` container is not running / not reachable

Quick fix:
```bash
docker compose build api
docker compose up -d db api
```

## Docker Compose (API + Postgres)

```bash
docker compose build api
docker compose up -d db api
```

Ports can be overridden via environment variables:
- `API_PORT` (default: `8000`)
- `DB_PORT` (default: `5432`)
- `WEB_PORT` (default: `5173`)

Note about `DATABASE_URL`:
- Docker Compose uses an internal Postgres URL: `postgresql://...@db:5432/...`
- If your local `.env` contains `DATABASE_URL=...@localhost:5432/...` (for running without Docker), it should **not** break Compose: the compose file uses `DATABASE_URL_DOCKER` (optional) instead of `DATABASE_URL`.

### Docker secrets (recommended)

1) Create local (uncommitted) files:

- `secrets/API_SHARED_SECRET`
- `secrets/OPENAI_API_KEY`
- `secrets/ELEVENLABS_API_KEY`

2) Start Compose with the overlay file:

```bash
docker compose -f docker-compose.yml -f docker-compose.secrets.yml up -d --build db api
```

### Synology DS224+ (Container Manager)

There is a dedicated NAS deployment bundle:

- `deploy/synology/docker-compose.synology.yml`
- `deploy/synology/.env.example`
- `deploy/synology/README.md`
- `deploy/synology/REVERSE_PROXY.md`
- `deploy/synology/RUNBOOK_192.168.2.123.md` (personal checklist)
- `deploy/synology/scripts/*` (prepare / validate / smoke / update / sleep / wake)

Quick start (from NAS SSH shell):

```bash
cd /volume1/docker/anki-csv-builder/app
git pull --ff-only
bash deploy/synology/scripts/prepare.sh
bash deploy/synology/scripts/validate_env.sh
```

Deploy from DSM UI:
1. `Container Manager -> Project -> Create`
2. Compose file: `/volume1/docker/anki-csv-builder/app/deploy/synology/docker-compose.synology.yml`
3. Env file: `/volume1/docker/anki-csv-builder/app/deploy/synology/.env`
4. `Deploy`

Smoke check:

```bash
bash deploy/synology/scripts/smoke.sh
```

Updates:

```bash
bash deploy/synology/scripts/update.sh
```

CLI alternative:

```bash
cp deploy/synology/.env.example deploy/synology/.env
# fill secrets in deploy/synology/.env
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env up -d --build
```

## Minimal Web UI (React + Vite)

The minimal UI lives in `web/` and calls the FastAPI service.

### Option A: local (requires Node.js)

```bash
cd web
npm install
npm run dev
```

### Option B: Docker (no Node.js on the host)

```bash
docker compose up -d db api
docker compose up web
```

Open the UI: `http://localhost:5173`.

By default, Vite proxies `/api` to `http://localhost:8000` (see `web/vite.config.ts`). In Docker Compose, the `web` service sets `VITE_API_TARGET=http://api:8000`, so you don't need CORS.

## Project layout

```
anki-csv-builder/
├── api/                 # FastAPI service
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

### Model selection notes

- `gpt-4.1-mini` keeps completion usage low (~300 tokens per card) and is the recommended economy default.
- `gpt-5-mini` often returns the same-length JSON but is billed for hidden reasoning tokens; expect ~4–5× higher completion token cost unless you specifically need its premium quality.

### Cache housekeeping

- Schema probe results persist in `cache/response_format.json`. Delete that file if you want to force the app to re-test `response_format` support for every model.

### Saving location

- The app suggests file names for CSV/APKG/JSON, but the browser decides the folder. To be prompted for a path on every download, enable “Ask where to save each file” in your browser settings (Downloads preferences).

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
- **Update pricing tables** if OpenAI releases new models or adjusts rates:
  - Text generation: `config/pricing.py::MODEL_PRICING_USD_PER_1M`
  - TTS: `config/pricing.py::AUDIO_MODEL_PRICING_USD_PER_1M_CHAR`
  The run report uses these tables to compute cost estimates; when a model is missing, the UI shows a warning with the model name.
