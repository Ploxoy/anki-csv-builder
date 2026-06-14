# API Contracts (Phase 0 targets)

*(draft — align with Vision 2.0, last updated: 2026-06-14)*

## Auth (Phase 0)

By default, the API is protected with a shared secret:

- Header: `X-API-Key: <API_SHARED_SECRET>`
- Server-side config: `API_SHARED_SECRET` (env or docker secret)

Local development override:

- Set `API_REQUIRE_SHARED_SECRET=0` to disable the shared-secret check (dev only).

## `/api/generate` — text → cards

**Request (JSON)**
```json
{
  "run_id": "uuid-optional",
  "prompt_version": "vX.Y",
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "cefr": "B1",
  "profile": "strict",
  "l1": "RU",
  "temperature": 0.4,
  "max_output_tokens": 1200,
  "guid_policy": "stable",
  "flags": {
    "force_schema": true,
    "allow_repair": true,
    "reuse_text_cache": false
  },
  "items": [
    {
      "id": "row-1",
      "woord": "voorbeeld",
      "def_nl": "een illustratie van iets",
      "translation": "пример"
    }
  ]
}
```

Notes:
- `l1` must be one of: `EN`, `RU`, `ES`, `DE` (case-insensitive; server normalizes to uppercase). Invalid values return HTTP 400.
- `flags`/`guid_policy` are accepted for forward compatibility; not all fields are used by the current implementation.
- `flags.reuse_text_cache=true` enables optional generated-card reuse. A saved card is reused only when normalized
  input (`woord/def_nl/translation`) and generation settings (`provider/model/prompt_version/CEFR/profile/L1/temperature`)
  match. Cache hits do not call the text provider and are reported in `timing.text_cache_hits`.

**Response (200)**
```json
{
  "run_id": "uuid",
  "prompt_version": "vX.Y",
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "items": [
    {
      "id": "row-1",
      "status": "ok|repaired|failed|flagged",
      "card": {
        "L2_word": "...",
        "L2_cloze": "...",
        "L1_sentence": "...",
        "L2_collocations": "...",
        "L2_definition": "...",
        "L1_gloss": "..."
      },
      "error": null,
      "usage": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "input_tokens": 123,
        "output_tokens": 456,
        "cached_tokens": 0,
        "elapsed_ms": 812
      }
    }
  ],
  "run_report": { "... same shape as app/run_report.py output ..." },
  "timing": {
    "elapsed_ms": 8123,
    "text_cache_hits": 0,
    "text_assets_stored": 1,
    "text_cache_errors": 0
  }
}
```

## `/health` — service health

**Response (200)**
```json
{ "status": "ok" }
```

## `/api/jobs/generate` — async text-generation job

Vercel-oriented async wrapper around `/api/generate`.

**Create job**
- `POST /api/jobs/generate`
- Body: same as `/api/generate`.
- Response:
```json
{
  "job_id": "uuid",
  "run_id": "uuid",
  "status": "queued"
}
```

**Poll job**
- `GET /api/jobs/generate/{job_id}`
- Response:
```json
{
  "job_id": "uuid",
  "run_id": "uuid",
  "status": "queued|running|done|failed",
  "processed_items": 8,
  "total_items": 20,
  "error": null,
  "result": null
}
```

**Worker tick**
- `POST /api/jobs/generate/worker`
- Used by Vercel cron/manual worker tick. The web UI may also trigger worker ticks while polling.

Current limitation:
- This avoids one long `/api/generate` request, but it is not yet a full durable `run_items` resume model for 1000+ rows.

## `/api/tts` — cards → audio

**Request (JSON)**
```json
{
  "run_id": "uuid-optional",
  "provider": "openai",
  "model": "gpt-4o-mini-tts-2025-12-15",
  "voice": "alloy",
  "items": [
    { "card_id": "row-1", "type": "word", "text": "voorbeeld" },
    { "card_id": "row-1", "type": "sentence", "text": "Dit is een voorbeeldzin." }
  ]
}
```

Notes:
- `model` is optional. If omitted, server picks provider default.
- Server applies a single automatic retry only for transient synthesis errors (`429/5xx/timeout`).
- Audio reuse is durable across runs when `provider + model + voice + clip type + text + style/instructions`
  match exactly after server normalization. Durable hits return `status: "cached"` and do not call the TTS provider.
- `/api/tts` persists per-run media in `run_media_assets` when `run_id` is present, and stores reusable global audio in `audio_assets`.
- Current Vercel limitation: long TTS runs still happen as web-managed batches of `/api/tts` calls. Cache/resume reduces repeated work, but there is no durable server-side `audio_jobs` queue yet.

**Response (200)**
```json
{
  "run_id": "uuid",
  "provider": "openai",
  "model": "gpt-4o-mini-tts-2025-12-15",
  "audios": [
    {
      "card_id": "row-1",
      "type": "word",
      "status": "ok",
      "filename": "word_voorbeeld__alloy__abcd1234.mp3",
      "audio_b64": "<base64 or omitted if served as file>",
      "error": null,
      "usage": {
        "provider": "openai",
        "model": "gpt-4o-mini-tts-2025-12-15",
        "audio_chars": 15,
        "audio_tokens": null,
        "seconds": null,
        "elapsed_ms": 420
      }
    }
  ],
  "summary": {
    "ok": 2,
    "failed": 0,
    "cached": 0,
    "usage": { "audio_chars": 42 },
    "cost": { "estimated_usd": 0.000063, "notes": null }
  },
  "storage": {
    "persisted": true,
    "stored_clips": 2,
    "error": null
  },
  "timing": {
    "elapsed_ms": 780,
    "synthesis_ms": 620,
    "storage_ms": 40,
    "items": 2,
    "unique_media_files": 2,
    "cache_hits": 1,
    "durable_cache_hits": 1,
    "audio_assets_stored": 1,
    "total_requests": 1,
    "provider": "openai"
  }
}
```

`status` meanings:
- `ok`: generated in this run.
- `cached`: reused from cache.
- `failed`: synthesis failed; check per-clip `error` and `summary.errors`.

## `/api/tts/options` — text/TTS model and voice options

**Request**
- `GET /api/tts/options`

**Response (200)**
```json
{
  "text_models": ["gpt-4.1-mini"],
  "providers": ["openai", "elevenlabs"],
  "by_provider": {
    "openai": {
      "models": ["gpt-4o-tts"],
      "voices": [{ "id": "alloy", "label": "alloy" }],
      "default_model": "gpt-4o-tts",
      "default_voice": "alloy"
    },
    "elevenlabs": {
      "models": ["eleven_multilingual_v2"],
      "voices": [{ "id": "voice-id", "label": "Voice label" }],
      "default_model": "eleven_multilingual_v2",
      "default_voice": "voice-id"
    }
  }
}
```

Current limitation:
- ElevenLabs voices are discovered/filtered by the backend. Manual “use this voiceID from my ElevenLabs library” is a planned UI/API improvement.

## `/api/audio/assets/check` — verify reusable audio filenames

Used by the web UI before TTS/export to avoid trusting stale `[sound:...]` fields from saved cards.

**Request**
```json
{
  "filenames": ["word_voorbeeld__alloy__abcd1234.mp3"]
}
```

**Response (200)**
```json
{
  "found": ["word_voorbeeld__alloy__abcd1234.mp3"],
  "missing": [],
  "error": null
}
```

## `/api/export/csv` and `/api/export/apkg` — cards → downloadable files

Both endpoints accept generated cards plus export settings. APKG can load persisted media by `run_id`, avoiding large browser-to-server `media_map` payloads.

**Request core fields**
```json
{
  "run_id": "uuid-optional",
  "l1": "RU",
  "cefr": "B1",
  "profile": "balanced",
  "model": "gpt-4.1-mini",
  "deck_name": "Dutch",
  "guid_policy": "stable",
  "include_basic_reversed": true,
  "include_basic_typein": true,
  "use_persisted_media": true,
  "media_map": null,
  "cards": [
    {
      "L2_word": "voorbeeld",
      "L2_cloze": "Dit is een {{c1::voorbeeld}}.",
      "L1_sentence": "Это пример.",
      "L2_collocations": "...",
      "L2_definition": "...",
      "L1_gloss": "пример",
      "L1_hint": "",
      "AudioWord": "[sound:word_voorbeeld.mp3]",
      "AudioSentence": "[sound:sentence_voorbeeld.mp3]"
    }
  ]
}
```

**CSV response**
- JSON `ExportFileResponse` with `content_b64`.

**APKG response**
- Binary streaming response with `Content-Disposition`.

Important APKG errors:
- `413`: request body is too large for Vercel.
- `409`: referenced audio is missing from `media_map`, `run_media_assets`, and global `audio_assets`.

## Error envelope (shared)

Target: use a single envelope for 4xx/5xx:
```json
{
  "error": {
    "code": "bad_request|unauthorized|forbidden|insufficient_funds|rate_limited|upstream_error",
    "message": "...",
    "details": { "retry_after_seconds": 10, "provider": "openai" }
  }
}
```

Current behavior (FastAPI default):
- Errors are returned as `{ "detail": "..." }` for many cases.

## UsageEvent shape (normalized)

Applies to every provider call:
```json
{
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "input_tokens": 123,
  "output_tokens": 456,
  "cached_tokens": 0,
  "audio_chars": null,
  "audio_tokens": null,
  "seconds": null,
  "raw_cost_usd": 0.00123,
  "raw_cost_eur": 0.00112,
  "charged_cost_eur": 0.00123,
  "markup_tier": "t1",
  "markup_multiplier": 1.1,
  "request_id": "provider-trace-id",
  "elapsed_ms": 812
}
```
