# Audio Panel Rewrite — Technical Specification

Status note (2026-06-14):
- This document started as the legacy Streamlit audio-panel spec. The active web implementation now also uses FastAPI + Postgres-backed audio persistence.
- Current durable stores: `run_media_assets` for per-run export media and `audio_assets` for reusable global TTS clips.
- Current unresolved gap: long TTS runs on Vercel are guarded by batching/timeouts/cache, but not yet handled by a durable `audio_jobs` queue with clip-level resume.
- Implemented UX additions: manual ElevenLabs `voiceID` validation/selection in web Settings and short voice preview for the current `provider/model/voice`.
- Planned UX addition: inline audio preview in the web Review card details.

## Goal
Build a streamlined post-generation audio workflow that lets users synthesize Dutch (nl-NL) word and sentence audio after cards are generated, with a clean separation between UI and core audio logic.

## Scope
- Replace the existing implementation in `app/tts_panel.py` with a modular, testable design that follows Vision §14 and §16.
- Keep `core/audio.py` as the orchestration layer for providers; UI must only interact with public functions.
- Support OpenAI TTS and ElevenLabs providers, with the ability to add more later without touching UI internals.

## Functional Requirements
- Allow synthesis of two audio streams per card: `AudioWord` and `AudioSentence`.
- Provide Quick TTS (one-click with defaults) and Advanced controls (provider, voice, styles, worker count, instructions).
- Show accurate progress and a run summary (requests, successes, skips, errors, cache hits, fallback usage).
- Integrate results into export flow: CSV `[sound:…]` fields and `.apkg` media via `media_map` returned from `ensure_audio_for_cards`.

## Provider Requirements
- **OpenAI TTS**: use `gpt-4o-mini-tts` with fallback `gpt-4o-tts`, instructions pulled from `config.settings` presets.
- **ElevenLabs**: use `eleven_multilingual_v2` (or successor compatible model) with `spoken_language='nl'`. API key supplied via Streamlit secrets/env or manual entry. Requests must obey rate limits (≤2 parallel workers, respect retry headers).
- Track provider-specific quotas: ElevenLabs тарифицирует токены помесячно, поэтому UI должен показывать текущий расход/лимит, а OpenAI использует общий счёт API-ключа.
- Обе модели склонны давать англоязычный акцент на одиночных словах — нужен способ пометить «NL-рекомендуемые» голоса и предупреждать/давать быстрый отказ для некачественных дорожек.
- Adding a provider must require only config additions plus a small adapter in `core/audio.py`.

## Voice Selection
- Build and maintain a curated list of voices suitable for Dutch (nl-NL) pronunciation for every provider.
- ElevenLabs UI must fetch the live catalogue, filter for NL-capable voices, and merge with the curated list when live data is unavailable.
- Voice dropdowns never display empty results; when live data is missing, show the curated NL list with a plain info message.

## UI/UX Requirements
- Single Streamlit expander `🔊 Audio`, collapsed by default until cards exist.
- Provider select remembers the last choice per session; switching providers preserves per-provider voice/styling selections.
- EleventLabs key field: password input that masks the value, persists within the session, and exposes a «Replace key» option. No flashing placeholders or input resets between reruns.
- Quick TTS button available as soon as a provider has a valid key; Advanced controls shown in an `st.expander` or similar.
- Progress display: textual stage + progress bar, updated via callback; summary shows counts and first few error messages.
- UI copy remains English-only; avoid modal popups, rely on subtle info/warning blocks.
- Summary выводит примерную стоимость TTS (на основе длины текста и `config/pricing.py::AUDIO_MODEL_PRICING_USD_PER_1K_CHAR`).

## Error Handling
- Missing key or voice → inline warning and disabled actions.
- API errors → retry with exponential backoff; log concise warning in UI summary and continue processing remaining items.
- Validation of generated audio metadata (filenames, byte sizes) before exposing downloads.

## State Management
- Store all user choices in `st.session_state` (`audio_provider`, `audio_voice_map`, `audio_sentence_instruction`, `audio_word_instruction`, etc.).
- Core audio cache (`audio_cache`) prevents duplicate synth for identical text/voice pairs.
- Summary data (`audio_summary`) is kept until user runs another synthesis or clears results.

## Non-Goals
- Legacy spec note: originally no new backends, queues, or databases. This is superseded for the web/Vercel path by Postgres persistence (`run_media_assets`, `audio_assets`); durable TTS job queues remain future work.
- No automatic background synthesis; user must trigger every run explicitly.
- No multilingual support beyond Dutch for this iteration.

## Deliverables
1. Refactored `app/tts_panel.py` (or new submodule) aligned with this spec.
2. Updated provider configuration in `config/settings.py` if voice lists are adjusted.
3. Unit tests for ElevenLabs catalogue fetch/filter and audio orchestration edge cases.
4. Updated documentation (`notes/status.md`, `README`) reflecting new audio workflow.
