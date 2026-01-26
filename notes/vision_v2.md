# Vision 2.0 — Doedutch / Anki CSV Builder (Productized)

*(product vision & migration plan — last updated: 2025-12-21)*

## 1) What we are building

**Doedutch** is a paid tool that turns a user’s Dutch study list into high-quality Anki cards (CSV/APKG) with optional audio, while keeping **quality predictable** (strict JSON) and **cost transparent** (tokens/audio usage → € cost → user charge).

We already have a working Streamlit MVP. Vision 2.0 describes the productized version: **separated user access, saved settings, prepaid billing, anti-abuse**, and a migration path to a backend-first architecture (Railway + FastAPI), with the UI later replaceable (Streamlit is temporary).

## 2) Core user value

1. Fast input (paste/upload) → generate cards with predictable structure (strict JSON).
2. Easy export (CSV/APKG) + optional post-generation TTS.
3. “Run report” explaining what happened: models, retries, repairs/fallback, tokens/audio usage, cost.
4. A paid experience with saved defaults and a personal budget (no public/shared access).

## 3) Product constraints (non-negotiables)

**Quality & determinism**
- LLM output must be **strict JSON** compatible with our schema/validator + repair-pass.

**Billing accuracy**
- Provider must return **usage** (input/output tokens + cached when available; and audio usage in provider units).
- If usage is not available/reliable → model/provider is **not eligible** for paid runs.

**Data minimization**
- We store **only**: user profile, settings, usage metrics, and billing ledger.
- We do **not** store prompts, generated cards JSON, exports, or audio files.

**Business model**
- **No free tier / no trial.**
- Users pay in advance (prepaid) and are charged per usage.

## 4) Monetization model (EUR, prepaid, tiered markup)

**Ledger**
- Balance is a ledger in **EUR**: `topups - charges`.

**How we charge**
- For each provider call (text or TTS) we create a `UsageEvent` and compute:
  - `raw_cost_eur`: provider cost estimate (usually USD→EUR via configured FX)
  - `charged_cost_eur`: user price = `raw_cost_eur * markup_multiplier`

**Markup tiers**
- Markup is **inversely proportional** to the user’s top-up amount.
- We store the applied tier and multiplier in each charge for auditability.

## 5) Default limits & anti-abuse

- Primary hard limit: **max words per run** (configurable).
- Dynamic server-side controls (changeable without redeploying UI):
  - per-user concurrency (how many runs at once)
  - rate limits (requests/min)
  - max tokens per request (`max_output_tokens`, safety caps)
  - minimum balance required to start a run (precheck)

## 6) Target architecture (vNext)

**Backend (Railway)**
- `FastAPI` app (single service initially).
- Postgres (prefer Supabase Postgres if using Supabase Auth).
- Public endpoints:
  - `/auth/*` (handled via Supabase client flow on frontend)
  - `/api/generate` (text generation)
  - `/api/tts` (audio synthesis)
  - `/api/settings` (persisted settings)
  - `/api/ledger` (balance + usage)
  - `/api/admin/*` (admin-only)
  - `/stripe/webhook` (Stripe)

**Auth**
- Google OAuth via **Supabase Auth**.
- Backend validates Supabase JWT → resolves `user_id`.

**UI**
- Streamlit UI remains as a temporary frontend during migration.
- Later: replaceable UI (TBD; could be a minimal web frontend) calling the same backend API.

## 6.1 API contracts (Phase 0 target)

Goal: make generation/TTS callable as pure APIs with strict schemas and usage for billing.

### `/api/generate` (text → cards)
**Request (JSON)**
```json
{
  "run_id": "uuid-optional",
  "prompt_version": "vX.Y",
  "model": "gpt-4.1-mini",
  "provider": "openai",
  "cefr": "B1",
  "profile": "strict",
  "l1": "RU",
  "temperature": 0.4,
  "max_output_tokens": 1200,
  "items": [
    {"id": "row-1", "woord": "voorbeeld", "def_nl": "…", "translation": "…"}
  ],
  "guid_policy": "stable|unique",
  "flags": {
    "force_schema": true,
    "allow_repair": true
  }
}
```

**Response (JSON)** — success (200
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
  "run_report": { "... same structure as current run_report ..." },
  "timing": { "elapsed_ms": 8123 }
}
```

### `/api/tts` (cards → audio)
**Request (JSON)**
```json
{
  "run_id": "uuid-optional",
  "provider": "openai",
  "model": "gpt-4o-mini-tts-2025-12-15",
  "voice": "alloy",
  "items": [
    {
      "card_id": "row-1",
      "type": "word|sentence",
      "text": "Ik ruim mijn kamer op."
    }
  ]
}
```

**Response (JSON)** — success (200)
```json
{
  "run_id": "uuid",
  "provider": "openai",
  "model": "gpt-4o-mini-tts-2025-12-15",
  "audios": [
    {
      "card_id": "row-1",
      "type": "word",
      "filename": "word_ruim__alloy__abcd1234.mp3",
      "audio_b64": "<base64 or omitted if served as file>",
      "usage": {
        "provider": "openai",
        "model": "gpt-4o-mini-tts-2025-12-15",
        "audio_chars": 42,
        "audio_tokens": null,
        "seconds": null,
        "elapsed_ms": 420
      }
    }
  ],
  "summary": {
    "ok": 1,
    "failed": 0,
    "cached": 0,
    "usage": { "audio_chars": 42 },
    "cost": { "estimated_usd": 0.000063, "notes": null }
  }
}
```

### Errors (both endpoints)
- 400 validation: `{ "error": { "code": "bad_request", "message": "...", "details": {...} } }`
- 401/403 auth: `{ "error": { "code": "unauthorized", "message": "..." } }`
- 402 balance: `{ "error": { "code": "insufficient_funds", "message": "...", "required_eur": 1.23 } }`
- 429 throttling: `{ "error": { "code": "rate_limited", "retry_after_seconds": 10 } }`
- 5xx upstream: `{ "error": { "code": "upstream_error", "provider": "openai", "message": "...", "retryable": true } }`

> Note: `run_report` already contains tokens/usage/cost; Phase 0 requires we always include `provider` and `model` in usage to support multi-provider billing.

## 7) Data model (minimal)

**users**
- `id (uuid)`, `email`, `name`, `created_at`, `is_admin`, `status(blocked/active)`

**user_settings**
- `user_id`, `settings_json`, `updated_at`

**usage_events**
- `id`, `user_id`, `provider`, `model`, `input_tokens`, `output_tokens`, `cached_tokens`,
  `audio_chars`, `audio_tokens`, `seconds`, `raw_cost_usd`, `raw_cost_eur`, `charged_cost_eur`,
  `markup_tier`, `markup_multiplier`, `request_id`, `created_at`

**ledger_entries**
- `id`, `user_id`, `type(topup|charge|manual_adjustment)`, `amount_eur`, `currency`,
  `ref_id (usage_event_id or stripe_event_id)`, `created_at`

**payments**
- `id`, `user_id`, `stripe_customer_id`, `stripe_session_id`, `stripe_event_id`, `status`, `amount_eur`, `created_at`

## 8) Multi-provider strategy (post-MVP)

We support alternative providers only if they satisfy:
- strict JSON output
- reliable usage reporting
- stable batching / retry semantics
- transparent pricing unit

**First evaluation candidates**
- Text: **Gemini Flash** (goal: cheaper € per valid card than OpenAI baseline)
- TTS: **Azure Neural TTS** (goal: better NL voice + per-character billing)

## 9) Phased migration plan (from MVP → vNext)

### Phase 0 — Stabilize boundaries (now)
**Goal:** make the current code “backend-ready”.
- Define request/response schemas for generation and TTS (even if still in-process).
- Ensure Run report JSON contains everything needed for billing (usage + model + provider).
- Keep prompts/schema versioned (`prompt_version`) so results are comparable.

**Exit criteria**
- A single “generate run” can be represented as a pure data request and a pure data response.

### Phase 1 — FastAPI wrapper on Railway (no auth yet)
**Goal:** deploy a backend that runs the existing generation/TTS logic.
- Implement `/api/generate` and `/api/tts` that call existing core modules.
- Keep it “single-user” temporarily (protected by a shared secret or allowlist) for internal testing.
- Streamlit can optionally call the backend instead of local code (feature flag).

**Exit criteria**
- MVP works end-to-end with backend calls (same outputs, same Run report fields).

### Phase 2 — Auth + persisted settings
**Goal:** separate access with Google login.
- Add Supabase Auth (Google OAuth).
- Add `user_settings` storage and load/save flows.

**Exit criteria**
- User logs in, sees their saved defaults on next session.

### Phase 3 — Usage ledger + max-words/run limit
**Goal:** enforce paid usage.
- Persist `usage_events` for each provider call.
- Compute `raw_cost_eur` and `charged_cost_eur` using pricing + FX + markup tiers.
- Enforce `max_words_per_run` and `min_balance_to_start`.
- Add admin controls for block/unblock and manual adjustments.

**Exit criteria**
- Every request produces a charge (or is blocked) and ledger balances reconcile.

### Phase 4 — Stripe top-ups (payments)
**Goal:** user can replenish balance.
- Add Stripe Checkout for top-ups.
- Implement `/stripe/webhook` and credit ledger on `checkout.session.completed`.
- Store Stripe IDs and event references for audit.

**Exit criteria**
- Top-up → balance increases → next run is allowed.

### Phase 5 — Replace Streamlit UI (optional, can be postponed)
**Goal:** remove Streamlit dependency.
- Build a minimal UI that talks to the backend API.
- Keep export behavior identical (CSV/APKG download).

**Exit criteria**
- Users can complete the full flow without Streamlit.

### Phase 6 — Add alternative providers behind flags (post-MVP)
**Goal:** lower cost and/or improve voice quality.
- Implement provider interfaces + pricing by `(provider, model)` units.
- Add Gemini Flash and Azure Neural TTS behind feature flags.
- Compare on a fixed test set (valid rate, repair rate, latency, € per valid card; NL voice quality).

**Exit criteria**
- Provider meets pass criteria and is safe to expose in UI.

## 10) Admin mode (required)

Admin dashboard/API should support:
- user search + status (active/blocked)
- balance view + usage summaries
- manual credit/debit (with audit trail)
- block/unblock user
- model/provider usage breakdown by time window

## 11) Open questions (to finalize before implementation)

1. Markup tiers: exact thresholds and multipliers (per top-up).
2. FX rate policy (manual value vs periodic update).
3. How we present pricing to users (show raw cost vs only charged cost).
4. ElevenLabs policy (BYOK vs separate paid add-on vs removed).
