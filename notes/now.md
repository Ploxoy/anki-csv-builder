# Now — Anki CSV Builder

Updated: 2026-01-24T21:45:00+00:00

## Quick pointers
- notes/status.md (project status)
- notes/tasks.md (task tracker)
- notes/vision_v2.md (product direction)

## Git status
```
## dev...origin/dev [ahead 1]
 M .gitignore
 M README.en.md
 M README.md
 M app/batch_runner.py
 M app/run_report.py
 M core/generation.py
 M core/prompts.py
 M docker-compose.yml
 M requirements.txt
 M tests/test_run_report.py
?? Dockerfile.api
?? api/
?? core/api_schemas.py
?? core/db.py
?? core/run_report.py
?? core/secrets.py
?? docker-compose.secrets.yml
?? notes/
?? requirements.api.txt
?? scripts/doctor.py
?? scripts/run_api.sh
?? scripts/smoke_api.py
?? scripts/update_context.py
?? web/
```

## Recent commits
```
592f776 updated pricing
b2e08c8 Updated prising model
27c2d75 make manual input textarea-driven and enhance run report
2ecc1ab pricing for OpenAI models
d731aa2 refactored generations/ tests
```

## Status (head)
```
# Status — Anki CSV Builder

*(обновлено: 2026-01-24)*

## Кратко
- MVP закрывает генерацию NL карточек с CSV/APKG экспортом, пакетной обработкой и базовой панелью диагностики.
- Постгенерационная озвучка теперь работает на двух провайдерах: OpenAI (по умолчанию) и ElevenLabs (премиум), со сводкой и кэшем.
- Основная боль: `app/generation_page.py` разросся до «комбайна» ~800 LOC, поэтому UI нужно декомпозировать.

## Что готово
- **Core / генерация**: нормализация cloze, repair-pass, сигнал-слова, batch + parallel ThreadPool, авто-адаптация воркеров на транзитных ошибках.
- **Экспорт**: CSV (pipe) + `.apkg`, в том числе привязка озвучки через `[sound:...]` и передачу `media_map` в genanki.
- **TTS**:
  - Общий слой `ensure_audio_for_cards` с кешом, прогресс-колбэком и агрегированной сводкой.
  - OpenAI TTS (`gpt-4o-mini-tts-2025-12-15` + fallback) со стилями из `config.settings`.
  - ElevenLabs: динамическая загрузка голосов по `ELEVENLABS_API_KEY`, фильтр по NL, spoken_language=nl, экспоненциальный бэк-офф на 429.
  - Голос хранится помодульно (`audio_voice_map`), переключение провайдера не сбрасывает выбор.
- **Secrets**: `OPENAI_API_KEY` и `ELEVENLABS_API_KEY` подтягиваются из secrets/env при старте, без ручного ввода.
- **Тесты**: `pytest` (11 тестов) зелёный.

## Свежие изменения (сентябрь 2025)
- Добавлен ElevenLabs с кэшируемым каталогом голосов и автоматическими ретраями (respect Retry-After, cap workers <=2).
- Введены стилистические пресеты с `spoken_language='nl'`, чтобы отдельные слова звучали корректно.
- UI аудио-панели переработан: выбор провайдера, предупреждения о rate limit, карта голосов, автозаполнение API key.
- Панель TTS декомпозирована: состояние вынесено в `app/audio_state.py`, каталог ElevenLabs — в `app/audio_catalog.py`, селекторы получают nonce-зависимые ключи, что устранило «залипание» выпадашек и подсказку «No results».
- Обновлены тесты (11 зелёных) после рефакторинга, smoke сценарий не менялся.
- Manual input переписан в формат «text area + parse»: можно вставить/ввести много строк в форматах `woord`, `woord — definitie — vertaling`, Markdown-таблица, TSV, `woord ;; definitie ;; vertaling`; есть режим **append** и кнопка **clear**.
- Upload/Manual switching: после загрузки файла остаёмся в режиме Upload (никаких скрытых автопереключений), переход в Manual — только по выбору пользователя.
- В сайдбаре появился селектор пресетов (Starter / Fast / Quality), который мгновенно выставляет модель, профиль, CEFR, L1 и флаги для быстрого старта; ручная настройка по-прежнему доступна в режиме «Custom».
- Добавлен «быстрый поток» на главной панели: кнопка `Generate → Preview → Export` запускает автоматическую генерацию, а расширенные кнопки Start/Next/Stop скрыты в `Advanced run controls`.
- Усилен onboarding: при пустом инпуте показывается карточка «Getting started», в сайдбаре появились англоязычные подсказки по API/CEFR/профилям, а причины `flagged_precheck` теперь подсвечены в превью (с примерами в скобках).
- Аудио-панель переписана: единый модуль `app/tts_panel.py` с кнопкой Quick TTS, прогресс-баром, резюме и Advanced-настройками (провайдер, голос, стили, workers). ElevenLabs теперь тянет live-каталог с fallback на кураторский список. Добавлен безопасный ввод ключа (не подставляется в поле) и троттлинг обновления каталога.
- ElevenLabs выбор починен: ключ хранится в `session_state`, переключается через Replace/Forget, кеш каталога чистится при смене ключа, загрузка голосов запускается вручную кнопкой (без повторных API вызовов на rerun).
- Исправлено: озвучка попадает в .apkg с первого экспорта — после синтеза выполняется `st.rerun()`. Секции на странице переставлены: «③ Audio» перед «④ Export».
- Статистика генерации вынесена в `app/run_status.py`: общий прогресс/summary теперь используют общие хелперы, `BatchRunner` и `RunController` стали короче и понятнее.
- Run report 2.0 (`app/run_report.py`): собираем токены/стоимость по моделям, долю repair/fallback, cached tokens, аудио-символы и стоимость TTS; отдаём готовый JSON (учёт repair-пассов). UI показывает таблицу по моделям + текстовую и аудио-оценку стоимости.
- Дополнительно отслеживаем длину ответов (raw vs очищенные поля) и число триммированных schema-ответов — помогает ловить «слетевшие» модели (например, `gpt-5-mini` с огромными completion).
- Провели сравнение `gpt-5-mini` vs `gpt-4.1-mini`: длина JSON одинаковая (≈300 raw символов), но у 5‑й серии высокие completion-токены из-за внутреннего reasoning (OpenAI тарифицирует скрытые выходные токены). Для бюджетных запусков рекомендуем `gpt-4.1-mini`, а `gpt-5-mini` помечаем как «качество с повышенной стоимостью».
- Озвучка теперь сохраняется в `cache/audio/*`: если сессия упала или квота закончилась, уже оплаченные MP3 остаются на диске и автоматически подхватываются при следующем запуске/экспорте.
- Prompt caching: `send_responses_request`/Run report теперь фиксируют `cached_tokens`, что позволяет отслеживать выгоду от повторного префикса и выявлять сбои в кэшировании.
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
- [x] P1 — Реализовать Run report в UI: собирать `response_format_removed`, `repair_attempted`, usage сигнал-слов, время/стоимость и выводить отчёт после завершения запуска.
- [x] P2 — Расширить авто-тесты: покрыты `core/llm_clients`, BatchRunner (оркестрация + воркеры), аудио кеш/фолбэк и smoke-тест rerun.
- [x] P3 — Проанализировать модели (`gpt-5-nano`, `gpt-4o-mini` и др.) на поддержку `response_format`, зафиксировать стратегию (schema vs текст) и обновить конфиг/документацию.
- [x] P4 — Расширить probe/кэш `response_format`: сохранять результаты между сессиями и логировать статистику по моделям.
- [ ] P5 — Пересмотреть рекомендации batch/workers для очень маленьких списков: сейчас 3 слова → три батча; нужен один батч с параллельными потоками.
- [x] P6 — Добавить сбор и отображение токенов (prompt/output) по моделям в статике/Run report, оценивать стоимость и выводить в UI.
- [ ] N1 — Навигация по ошибкам в превью: фильтр «только ошибки» + кнопка «следующая ошибка» (дополнительно — синхронизация с таблицей результатов при необходимости).
- [ ] N2 — Убрать всплывающее «No results» в ElevenLabs selectbox сразу после ввода ключа: каталог голосов подгружается, но UI показывает пустой список до следующего взаимодействия.
- [ ] N3 — Добавить ручное редактирование в превью (② Preview & fix): возможность править и сохранять выбранные поля сгенерированных записей прямо в таблице/форме; конкретный список полей согласуем отдельно.

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
```

## Session scratchpad
- What I changed:
  - Added a Dockerized `web/` dev flow (Node in Compose) so Windows users don't need local npm.
  - Made Vite proxy target configurable via `VITE_API_TARGET` (Compose sets it to `http://api:8000`).
  - Updated README(s) with "local vs Docker" web UI instructions and `WEB_PORT`.
  - Added Node 20 Dev Container feature (optional convenience).
  - Settings persistence in web UI now covers temperature, force-generate flagged entries, audio toggles (word/sentence), card variants (reversed/type-in), and default deck name per user; added checkboxes/inputs to Settings panel.
- Why:
  - Unblocked Windows setup where `npm install` was failing / hard to manage, while keeping the same API contract.
- Next steps:
  - Decide focus: (A) users + persisted settings + usage accounting (no payments yet), then (B) simplify UI, then (C) bugfix/optimisation pass.
- Open questions:
  - Auth approach for beta: Supabase Auth (Google) now vs keep shared-secret + allowlist for internal users.
  - What to persist as "user settings" v1 (minimal) and how to version/migrate it.
