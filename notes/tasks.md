# Task Tracker — Anki CSV Builder

## 🔥 High Priority
- [x] H1 — Исправить зависающий индикатор прогресса в панели «🔊 Озвучивание» (`app/app.py`): прогресс/проценты должны обновляться при каждом завершённом запросе, без зависания на 100%.
- [x] H2 — Перевести TTS на потоковую/параллельную обработку и добавить отчёт по успешным/пропущенным элементам (асинхронность, отзывчивость UI на больших наборах).
- [x] H3 — Декомпозировать `app/app.py` (>1300 строк) на модули: управление батчами, UI-компоненты, панель TTS, утилиты прогресса.
- [x] H4 — Починить выбор ElevenLabs в аудио-панели: ввод ключа теперь не сбрасывается, каталог голосов загружается по кнопке с кешем и не блокирует rerun (`app/tts_panel.py`, `app/audio_catalog.py`, `app/audio_state.py`).
 - [x] H5 — Manual input: перейти с `st.data_editor` на «text area + parse» (поддержка форматов как у file upload), добавить append/clear.
- [x] H6 — Checkpoint fixed + current scope merged: web+api export endpoints, dynamic TTS options, resilient audio flow.
- [x] H7 — TTS Reliability/UX hardening: clip-level `status/error` в `/api/tts`, single-retry для transient ошибок, progress stage/elapsed/waiting-provider, partial-audio warnings на экспорте, тесты API/core.
- [x] H8 — Deep UI Rework v1 (clarity-first): light theme, декомпозиция `App.tsx` на табовые компоненты, scoped notices по секциям, один primary CTA в Generate, dirty-state в Settings, структурированный Admin.

## 🎯 Near-Term Plan
- [x] W0 — Минимальный web UI: React+Vite в `web/` + сервис `web` в Docker Compose (Node в контейнере), чтобы не зависеть от локального npm на Windows.
- [x] P1 — Реализовать Run report в UI: собирать `response_format_removed`, `repair_attempted`, usage сигнал-слов, время/стоимость и выводить отчёт после завершения запуска.
- [x] P2 — Расширить авто-тесты: покрыты `core/llm_clients`, BatchRunner (оркестрация + воркеры), аудио кеш/фолбэк и smoke-тест rerun.
- [x] P3 — Проанализировать модели (`gpt-5-nano`, `gpt-4o-mini` и др.) на поддержку `response_format`, зафиксировать стратегию (schema vs текст) и обновить конфиг/документацию.
- [x] P4 — Расширить probe/кэш `response_format`: сохранять результаты между сессиями и логировать статистику по моделям.
- [x] P5 — Пересмотреть рекомендации batch/workers для очень маленьких списков: сейчас `recommend_batch_params(total<=10)` даёт один батч (`batch_size=total`) с параллельными потоками (`app/ui_helpers.py`).
- [x] P6 — Добавить сбор и отображение токенов (prompt/output) по моделям в статике/Run report, оценивать стоимость и выводить в UI.
- [x] N1 — Навигация по ошибкам в превью: фильтр «только ошибки» + кнопка «следующая ошибка» (`app/preview_panel.py`).
- [ ] N2 — Убрать всплывающее «No results» в ElevenLabs selectbox сразу после ввода ключа: каталог голосов подгружается, но UI показывает пустой список до следующего взаимодействия. *(отложено на неопределённый срок: вероятно связано с изменениями внешнего ElevenLabs API; вернуться после стабилизации интеграции/контракта)*
- [x] N3 — Добавить ручное редактирование в превью (② Preview & fix): редактирование и сохранение полей `L2_word/L2_cloze/L1_sentence/L2_collocations/L2_definition/L1_gloss/L1_hint` в `session_state.results` (`app/preview_panel.py`).
- [x] N4 — `Reload model list` в web теперь обновляет и текстовые, и TTS-модели из backend (без постоянного показа fallback-списка по умолчанию при успешном live-fetch) (`api/main.py`, `web/src/App.tsx`, `web/src/features/settings/SettingsTab.tsx`).
- [x] N5 — Добавить авто-актуализацию model/voice list в web: тихий периодический refresh настроек TTS при активном токене, чтобы backend-изменения подхватывались без ручного клика (`web/src/App.tsx`).
- [ ] N6 — Перенести ввод `X-API-Key`/`API_SHARED_SECRET` из `Settings` в `Admin` tab (или добавить зеркальный контрол в `Admin`), чтобы админ-действия (`Create invite`, `List users`, `Rotate`) настраивались в том же контексте.
- [ ] N7 — ElevenLabs voiceID UX: добавить в Settings возможность вручную указать ElevenLabs `voiceID`, проверить/сохранить этот голос и использовать его даже если live catalogue/filter его не показывает.
- [ ] N8 — Review audio preview: в раскрывающейся карточке/JSON-сводке Review добавить inline проигрывание `AudioWord` и `AudioSentence` при наличии audio asset/media.
- [x] D1 — Synology internet-stage toolkit: gate-check/reachability скрипты (`check_wan_mode.sh`, `check_public_endpoints.sh`), direct path docs (`REVERSE_PROXY.md`) и Cloudflare fallback (`CLOUDFLARE_TUNNEL.md`, `docker-compose.cloudflared.yml`).
- [x] D2 — Windows LAN deployment pipeline: `deploy/synology/Deploy-FromLan.ps1` (SSH-key preflight, `git pull --ff-only`, `validate_env`, `compose up --build`, smoke + local health retries) + обновлён runbook `RUNBOOK_192.168.2.10.md`.
- [x] D3 — Power-save layer на Synology: добавлены `waker` + `socket-proxy` (auto sleep/wake, front-door на WEB_PORT), новые env-параметры `WAKER_IDLE_*` и статус `/_waker/status`.
- [x] D4 — One-command install для Synology: `deploy/synology/scripts/install.sh` (NAS) и `deploy/synology/scripts/install.ps1` (Windows -> SSH sync + remote install), плюс унификация docker-вызовов через `deploy/synology/scripts/docker_cmd.sh`.
- [x] D5 — Timeout hotfix для длинных запусков: `WAKER_PROXY_TIMEOUT_SECONDS=600` + timeout в `web/deploy/nginx.synology.conf` (снижение риска HTTP 504 на длинной генерации).
- [x] D6 — Инцидент после внедрения sleep/wake закрыт: проблема 504 на длинной text-generation воспроизводилась при одном длинном `/api/generate`; исправлено батчевой генерацией в `web/src/App.tsx` + корректным `sudo`-перезапуском контейнеров на Synology; подтверждён успешный прогон длинного списка (52 записи, audio off).
- [x] V1 — Vercel Plan C (async generate): добавлены queue/job endpoints (`/api/jobs/generate`, `/api/jobs/generate/{id}`, `/api/jobs/generate/worker`), storage таблица `generation_jobs` и polling-worker flow в web (`web/src/App.tsx`) для избежания длинных синхронных таймаутов.
- [x] V2 — Vercel bootstrap: добавлены `vercel.json`, `api/index.py`, `api/requirements.txt` (FastAPI function + static web build + cron endpoint wiring).
- [x] V4 — TTS diagnostics for Vercel: добавить timing в `/api/tts` и batch-level diagnostics в web UI для анализа долгих ElevenLabs-прогонов.
- [x] V3 — Vercel large-APKG path: `/api/tts` сохраняет audio clips server-side в Postgres (`run_media_assets`), а `/api/export/apkg` переиспользует их по `run_id`, чтобы не упираться в request-size limit при больших колодах с аудио.
- [x] V5 — Durable TTS asset store v1: глобальная таблица `audio_assets` по deterministic asset key (`provider/model/voice/type/text/style`) + reuse в `/api/tts`, чтобы повторные озвучки не вызывали TTS-провайдера заново.
- [x] V5a — Audio asset consistency gate: `/api/audio/assets/check` + web preflight для attached `[sound:...]`, чтобы отсутствующие mp3 из `audio_assets` досинтезировались, а не пропускались UI.
- [x] V5b — TTS hang guard: OpenAI speech timeout, provider-specific worker caps, web TTS batches по 6 clips и provider-specific abort/diagnostics.
- [x] V6a — Optional generated-card reuse v1: глобальная таблица `generated_card_assets` и флаг `reuse_text_cache`; при совпадении input+generation settings `/api/generate` возвращает сохранённую карточку без вызова LLM. Batch lookup/touch оптимизированы, чтобы повторные длинные списки не делали DB roundtrip на каждую строку.
- [ ] V6 — Long-run generation persistence: durable `run_items`/resume flow для списков 1000+ строк, чтобы один сбой не требовал перегенерации уже готовых карточек.
- [ ] V6b — Vercel long-list/TTS reliability: проблему больших списков и Vercel считаем не полностью решённой; нужен durable `audio_jobs`/resume flow, clip-level persistence/progress и безопасное продолжение после timeout без повторной оплаты уже готовых клипов.
- [ ] V7 — Lexical memory v1: спроектировать словарный слой (`lexeme_entries`, examples, translations by L1, quality flags) поверх накопленных генераций.

## 🧪 Beta readiness (Phase 0.5)
- [x] B1 — Invite-token auth v0: `/api/admin/invite` (admin) + `Authorization: Bearer <token>` (user), без Supabase.
- [x] B2 — Persist settings v1: `/api/settings` (read/write) + хранение `settings_json` по user_id в Postgres.
- [x] B3 — Usage view v0: `/api/usage` (read-only) на базе `usage_events` (по user_id/run_id).
- [x] B4 — Web UI: хранить invite token локально, грузить/сохранять settings и показывать usage.
- [x] B5 — Admin в web: создавать инвайты, листать пользователей, блок/разблок, ротация токена, просмотр usage по user_id.

## 🚀 Productization (vNext)
- [ ] A1 — Auth: Google login через Supabase Auth, `user_id` во всех запросах.
- [ ] A2 — Persist settings: хранить настройки пользователя в Postgres (без артефактов/истории).
- [ ] A3 — Budget ledger (EUR): usage → raw cost → charged cost (tiered markup), лимит max words/run, min balance precheck.
- [ ] A4 — Payments: Stripe top-ups + webhook endpoint (Railway/FastAPI).
- [ ] A5 — Admin mode: список пользователей, баланс/usage, ручной credit/debit, блокировки.
- [ ] A6 — Railway migration: FastAPI service + DB, постепенный уход со Streamlit UI.
- [x] A0 — Vision 2.0: сформулировать product vision + phased migration plan (`notes/vision_v2.md`).

## 🧩 Multi-provider (post-MVP)
- [ ] M0 — Provider discovery plan: выбрать 1–2 альтернативных провайдера для text generation и 1–2 для TTS, описать критерии допуска (structured output, latency, NL quality, usage/cost visibility, Vercel compatibility).
- [ ] M1 — Provider interfaces: `TextProvider`/`TTSProvider` + нормализованный `UsageEvent`.
- [ ] M2 — Pricing 2.0: хранить цены по `(provider, model)` + единицы (tokens/chars/seconds) + FX USD→EUR.
- [ ] M3 — Evaluate Text provider #2 (Gemini Flash): strict JSON + usage + дешевле OpenAI на нашем workload.
- [ ] M4 — Evaluate TTS provider #2 (Azure Neural TTS): NL voice quality + прозрачная тарификация; добавить за флагом.
- [ ] M5 — Decide ElevenLabs policy: либо BYOK для TTS, либо отдельный paid add-on с прозрачным лимитом.
- [ ] M6 — Capability/health UI: показывать доступность провайдера, модели, голоса, лимитов и последней ошибки без запуска генерации.

## 📜 API Contracts (Phase 0)
- [x] C1 — Зафиксировать JSON-примеры `/api/generate` и `/api/tts` + `UsageEvent` (см. `notes/api_contracts.md`).
- [x] C2 — Добавить в run_report/provider usage явные поля `provider`/`model` для биллинга.
- [x] C3 — Собирать нормализованный `usage_events` (text/audio) в run_report для дальнейшего биллинга.
- [x] C4 — Pydantic-схемы подключены в API (`api/main.py` импортирует и использует `core/api_schemas.py`).

## ⚙️ Technical Debt — Function Calling
- [ ] T1 — Определить минимальный read-only тулсет (`check_separable`, `get_cached_collocations`, доп. проверки) и формализовать контракты инструментов.
- [ ] T2 — Реализовать диспетчер function-calling (маршрутизация, лимиты, логирование) с совместимостью с Responses API.
- [ ] T3 — Добавить unit-тесты: probe, отказ от schema, happy-path tool call, обработка ошибок диспетчера.
- [ ] T4 — Интегрировать метрики tool-calls в Run report (количество вызовов, отказы, fallback на текст) и отображать в UI.
 - [x] T5 — Прояснить использование OpenAI prompt caching (cached_tokens, 5–10 мин TTL), зафиксировать сбор метрики в Run report и рекомендации по стабильному префиксу.

> Источник: обновлены `notes/status.md` и `notes/techical_debt.md` (2026-02-17).
