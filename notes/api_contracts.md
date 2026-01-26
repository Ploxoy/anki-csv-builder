# API Contracts (Phase 0 targets)

*(draft — align with Vision 2.0, last updated: 2025-12-21)*

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
    "allow_repair": true
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
  "timing": { "elapsed_ms": 8123 }
}
```

## `/health` — service health

**Response (200)**
```json
{ "status": "ok" }
```

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
      "filename": "word_voorbeeld__alloy__abcd1234.mp3",
      "audio_b64": "<base64 or omitted if served as file>",
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
  }
}
```

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
