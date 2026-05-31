# Anki CSV Builder

This repository is now focused on a **web + API** architecture for Anki card generation, with an async generation pipeline designed for Vercel.

## Repository layout

- `web/` — React + Vite UI (`Generate / Settings / Admin`)
- `api/` — FastAPI backend
- `core/` — generation, TTS, export, queue logic
- `config/` — model/template configuration
- `tests/` — tests
- `notes/` — project notes and status

## Current architecture (Plan C)

Long-running generation is handled as async jobs to avoid request timeouts:

1. `POST /api/jobs/generate` — enqueue job
2. `POST /api/jobs/generate/worker` — process part of queue
3. `GET /api/jobs/generate/{job_id}` — poll status/result

The UI already uses this flow, with fallback to sync `/api/generate` if job endpoints are unavailable.

## Quick local start

### 1) API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt

export DATABASE_URL="postgresql://..."
export OPENAI_API_KEY="..."
export API_SHARED_SECRET="..."
export API_REQUIRE_SHARED_SECRET=1
export CRON_SECRET="..."

uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2) Web

```bash
cd web
npm ci
npm run dev
```

## Vercel deployment

Full deployment runbook:

- `deploy_vercel.md`

Already included in repo:

- `vercel.json`
- `api/index.py`
- `api/requirements.txt`

## Required production environment variables

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `API_SHARED_SECRET`
- `API_REQUIRE_SHARED_SECRET=1`
- `CRON_SECRET`

Optional:

- `ELEVENLABS_API_KEY`
- `GENERATE_JOB_MAX_ITEMS_PER_WORKER`
- `GENERATE_JOB_STALE_SECONDS`

## Main API endpoints

- Health: `GET /health`
- Generate sync (legacy fallback): `POST /api/generate`
- Generate async:
  - `POST /api/jobs/generate`
  - `GET /api/jobs/generate/{job_id}`
  - `POST /api/jobs/generate/worker`
- TTS:
  - `GET /api/tts/options`
  - `POST /api/tts`
- Export:
  - `POST /api/export/csv`
  - `POST /api/export/apkg`
- Settings/usage/admin:
  - `GET/PUT /api/settings`
  - `GET /api/usage`
  - `POST /api/admin/invite`
  - `GET /api/admin/users`

## Note

Legacy layers (Streamlit UI, Synology deployment scripts, old docker-compose stack) were removed from the active project scope.
