# Now — Anki CSV Builder

Updated: 2026-02-17T07:36:51

## Quick pointers
- notes/status.md (project status)
- notes/tasks.md (task tracker)
- notes/vision_v2.md (product direction)

## Git status
```
## dev...origin/dev
 M app/preview_panel.py
 M notes/api_contracts.md
 M notes/now.md
 M notes/status.md
 M notes/tasks.md
 M notes/vision.md
 M notes/vision_v2.md
```

## Recent commits
```
8593509 transition to flask
592f776 updated pricing
b2e08c8 Updated prising model
27c2d75 make manual input textarea-driven and enhance run report
2ecc1ab pricing for OpenAI models
```

## Status (head)
```
# Status — Anki CSV Builder

*(обновлено: 2026-02-17)*

## Кратко
- MVP закрывает генерацию NL карточек с CSV/APKG экспортом, пакетной обработкой и базовой панелью диагностики.
- Постгенерационная озвучка теперь работает на двух провайдерах: OpenAI (по умолчанию) и ElevenLabs (премиум), со сводкой и кэшем.
- Основной стек API/UI сейчас: FastAPI (`api/main.py`) + минимальный web UI (`web/`) + legacy Streamlit.

## Что готово
- **Core / генерация**: нормализация cloze, repair-pass, сигнал-слова, batch + parallel ThreadPool, авто-адаптация воркеров на транзитных ошибках.
- **Экспорт**: CSV (pipe) + `.apkg`, в том числе привязка озвучки через `[sound:...]` и передачу `media_map` в genanki.
- **TTS**:
  - Общий слой `ensure_audio_for_cards` с кешом, прогресс-колбэком и агрегированной сводкой.
  - OpenAI TTS (`gpt-4o-mini-tts-2025-12-15` + fallback) со стилями из `config.settings`.
  - ElevenLabs: динамическая загрузка голосов по `ELEVENLABS_API_KEY`, фильтр по NL, spoken_language=nl, экспоненциальный бэк-офф на 429.
  - Голос хранится помодульно (`audio_voice_map`), переключение провайдера не сбрасывает выбор.
- **Secrets**: `OPENAI_API_KEY` и `ELEVENLABS_API_KEY` подтягиваются из secrets/env при старте, без ручного ввода.
- **Тесты**: `pytest` (23 теста) зелёный.

## Свежие изменения (февраль 2026)
- Превью в Streamlit поддерживает ручное редактирование ключевых полей карточки и сохранение изменений в `session_state.results` (`app/preview_panel.py`).
- Для малых списков рекомендация batch/workers уже оптимизирована: при `total <= 10` используется один батч (`batch_size=total`) с параллельными воркерами (`app/ui_helpers.py`).
- Навигация по ошибкам в превью уже активна (`Show only errors` + `Next error`) в `app/preview_panel.py`.
- Документация синхронизирована с фактическим состоянием: FastAPI остаётся рабочим API-слоем, web UI работает через `web/`.

## Свежие изменения (январь 2026)
- Добавлен минимальный web UI (React + Vite) в `web/`, ходит в FastAPI.
- Для dev/Windows: добавлен сервис `web` в Docker Compose (Node внутри контейнера), чтобы не требовать локальный npm.
- Vite dev-прокси на API теперь настраивается через `VITE_API_TARGET` (в Compose: `http://api:8000`).
- Проверено на Windows: после запуска Docker Desktop `docker compose up -d db api` + `docker compose up web` поднимает UI, запросы выполняются без ошибок.
- Phase 0.5: invite-token auth + админка в web (создать/блок/разблок/ротация токена, просмотр usage), `/api/settings` и `/api/usage` привязаны к user_id; в web сохраняются все ключевые настройки (модель, температура, аудио флаги word/sentence, card variants reversed/type-in, force generate flagged, default deck name).

## Свежие изменения (сентябрь 2025)
- Добавлен ElevenLabs с кэшируемым каталогом голосов и автоматическими ретраями (respect Retry-After, cap workers <=2).
- Введены стилистические пресеты с `spoken_language='nl'`, чтобы отдельные слова звучали корректно.
- UI аудио-панели переработан: выбор провайдера, предупреждения о rate limit, карта голосов, автозаполнение API key.
- Панель TTS декомпозирована: состояние вынесено в `app/audio_state.py`, каталог ElevenLabs — в `app/audio_catalog.py`, селекторы получают nonce-зависимые ключи, что устранило «залипание» выпадашек и подсказку «No results».
- Обновлены тесты (на тот момент: 11 зелёных) после рефакторинга, smoke сценарий не менялся.
- Manual input переписан в формат «text area + parse»: можно вставить/ввести много строк в форматах `woord`, `woord — definitie — vertaling`, Markdown-таблица, TSV, `woord ;; definitie ;; vertaling`; есть режим **append** и кнопка **clear**.
```

## Tasks (head)
```
# Task Tracker — Anki CSV Builder

## 🔥 High Priority
- [x] H1 — Исправить зависающий индикатор прогресса в панели «🔊 Озвучивание» (`app/app.py`): прогресс/проценты должны обновляться при каждом завершённом запросе, без зависания на 100%.
- [x] H2 — Перевести TTS на потоковую/параллельную обработку и добавить отчёт по успешным/пропущенным элементам (асинхронность, отзывчивость UI на больших наборах).
- [x] H3 — Декомпозировать `app/app.py` (>1300 строк) на модули: управление батчами, UI-компоненты, панель TTS, утилиты прогресса.
- [x] H4 — Починить выбор ElevenLabs в аудио-панели: ввод ключа теперь не сбрасывается, каталог голосов загружается по кнопке с кешем и не блокирует rerun (`app/tts_panel.py`, `app/audio_catalog.py`, `app/audio_state.py`).
 - [x] H5 — Manual input: перейти с `st.data_editor` на «text area + parse» (поддержка форматов как у file upload), добавить append/clear.

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
```

## Session scratchpad
- What I changed:
- Marked N2 as intentionally deferred indefinitely in `notes/tasks.md` with rationale (likely upstream ElevenLabs API changes).
- Updated `notes/status.md` next steps to reflect deferred N2.
- Regenerated `notes/now.md` via `scripts/update_context.py`.
- Why:
- Avoid spending time on a potentially upstream/integration-driven UX issue before API behavior stabilizes.
- Next steps:
- Continue with other roadmap items that are fully under our control.
- Open questions:
- None for N2 until ElevenLabs API contract is re-validated.
