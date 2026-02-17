# Task Tracker — Anki CSV Builder

## 🔥 High Priority
- [x] H1 — Исправить зависающий индикатор прогресса в панели «🔊 Озвучивание» (`app/app.py`): прогресс/проценты должны обновляться при каждом завершённом запросе, без зависания на 100%.
- [x] H2 — Перевести TTS на потоковую/параллельную обработку и добавить отчёт по успешным/пропущенным элементам (асинхронность, отзывчивость UI на больших наборах).
- [x] H3 — Декомпозировать `app/app.py` (>1300 строк) на модули: управление батчами, UI-компоненты, панель TTS, утилиты прогресса.
- [x] H4 — Починить выбор ElevenLabs в аудио-панели: ввод ключа теперь не сбрасывается, каталог голосов загружается по кнопке с кешем и не блокирует rerun (`app/tts_panel.py`, `app/audio_catalog.py`, `app/audio_state.py`).
 - [x] H5 — Manual input: перейти с `st.data_editor` на «text area + parse» (поддержка форматов как у file upload), добавить append/clear.
- [x] H6 — Checkpoint fixed + current scope merged: web+api export endpoints, dynamic TTS options, resilient audio flow.
- [x] H7 — TTS Reliability/UX hardening: clip-level `status/error` в `/api/tts`, single-retry для transient ошибок, progress stage/elapsed/waiting-provider, partial-audio warnings на экспорте, тесты API/core.

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
- [ ] M1 — Provider interfaces: `TextProvider`/`TTSProvider` + нормализованный `UsageEvent`.
- [ ] M2 — Pricing 2.0: хранить цены по `(provider, model)` + единицы (tokens/chars/seconds) + FX USD→EUR.
- [ ] M3 — Evaluate Text provider #2 (Gemini Flash): strict JSON + usage + дешевле OpenAI на нашем workload.
- [ ] M4 — Evaluate TTS provider #2 (Azure Neural TTS): NL voice quality + прозрачная тарификация; добавить за флагом.
- [ ] M5 — Decide ElevenLabs policy: либо BYOK для TTS, либо отдельный paid add-on с прозрачным лимитом.

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
