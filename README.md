# Anki CSV Builder

Проект перезапущен как **web + API** приложение для генерации карточек Anki с асинхронным пайплайном под Vercel.

## Что внутри

- `web/` — React + Vite UI (`Generate / Settings / Admin`)
- `api/` — FastAPI backend
- `core/` — генерация, TTS, экспорт, логика очередей
- `config/` — конфиг моделей/шаблонов
- `tests/` — тесты
- `notes/` — рабочая документация проекта

## Текущая архитектура (Plan C)

Длинная генерация работает через очередь, чтобы избежать HTTP timeout:

1. `POST /api/jobs/generate` — поставить задачу в очередь
2. `POST /api/jobs/generate/worker` — обработать часть очереди
3. `GET /api/jobs/generate/{job_id}` — получить статус/результат

UI уже использует этот flow, с fallback на sync `/api/generate`, если job-endpoints недоступны.

## Быстрый локальный запуск

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

## Деплой на Vercel

Подробный runbook:

- `deploy_vercel.md`

В репозитории уже есть:

- `vercel.json`
- `api/index.py`
- `api/requirements.txt`

## Обязательные ENV для продакшена

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `API_SHARED_SECRET`
- `API_REQUIRE_SHARED_SECRET=1`
- `CRON_SECRET`

Опционально:

- `ELEVENLABS_API_KEY`
- `GENERATE_JOB_MAX_ITEMS_PER_WORKER`
- `GENERATE_JOB_STALE_SECONDS`

## Основные API endpoints

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

## Примечание

Legacy-слои (Streamlit UI, Synology deploy scripts, старый docker-compose стек) удалены из активного контура проекта.
