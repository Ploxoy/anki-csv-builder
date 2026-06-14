# Now — Anki CSV Builder

Updated: 2026-06-14T13:19:38

## Quick pointers
- notes/status.md (project status)
- notes/tasks.md (task tracker)
- notes/vision_v2.md (product direction)

## Git status
```
## versel...origin/versel
 M api/main.py
 M notes/api_contracts.md
 M notes/status.md
 M tests/test_api_tts.py
 M web/src/features/admin/AdminTab.tsx
```

## Recent commits
```
b69d0bb voice admin
a8f6008 added voice public link
542fe56 public voice correction
1cac0d8 audi issues
d830884 voices preview
```

## Status (head)
```
# Status — Anki CSV Builder

*(обновлено: 2026-06-14)*

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
- **Тесты**: `pytest` зелёный, включая свежие проверки API-контрактов TTS.

## Свежие изменения (июнь 2026)
- **Vercel APKG export**: убран критичный bottleneck с большим `media_map` в теле запроса. `/api/tts` теперь сохраняет озвучку server-side в Postgres (`run_media_assets`) по `user_id + run_id + filename`.
- **Server-side media reuse**: `/api/export/apkg` умеет добирать аудио из persisted storage по `run_id`, поэтому браузер больше не обязан пересылать base64-клипы обратно на API для крупных колод.
- **Web export UX**: фронтенд предпочитает persisted-media путь для APKG, а inline `media_map` оставляет только как fallback для локальных/малых сценариев; в Generate теперь виден статус `Server storage` для аудио.
- **TTS diagnostics for Vercel**: `/api/tts` теперь возвращает `timing` (`elapsed_ms`, `synthesis_ms`, `storage_ms`, `cache_hits`, `unique_media_files`), а web UI показывает batch-level diagnostics для аудио. Это нужно для точного разбора долгих ElevenLabs-прогонов без больших повторных затрат.
- **Durable TTS asset store v1**: добавлена глобальная таблица `audio_assets` и deterministic asset key по `provider/model/voice/type/text/style`. `/api/tts` сначала ищет уже готовое аудио в Postgres, cache hits возвращает как `status="cached"` и не вызывает TTS-провайдера; новые клипы сохраняются обратно в asset store.
- **Optional generated-card reuse v1**: добавлена таблица `generated_card_assets` и флаг `reuse_text_cache`. При включении `Reuse saved cards` API переиспользует уже готовую карточку, если совпадают input (`woord/def_nl/translation`) и generation settings (`provider/model/prompt_version/CEFR/profile/L1/temperature`); UI показывает `reused saved cards` и `saved cards` в Review.
- **Long-list tuning**: web generate batches для 40+ строк укрупнены (меньше job/worker/poll кругов), worker chunk увеличен до 8 строк (до 12 при text reuse), backend default `GENERATE_JOB_MAX_ITEMS_PER_WORKER` поднят до 6. Text reuse preload теперь делает один batch lookup по карточкам на chunk.
- **Repeat-run 504 fix**: при `Reuse saved cards` web теперь использует direct text mode вместо async job queue, чтобы не тратить минуты на queue/poll при 100% text-cache hits. TTS больше не пересинтезирует clips, если `AudioWord/AudioSentence` уже прикреплены к сохранённой карточке; APKG export умеет добирать такие mp3 из глобального `audio_assets` по filename.
- **TTS 504 hardening**: retryable HTTP `429/502/503/504` дробит batch на меньшие части, а ошибка одиночного clip не помечает весь оставшийся хвост failed.
- **Audio asset consistency gate**: добавлен `/api/audio/assets/check`; перед TTS web проверяет, что уже прикреплённые `[sound:...]` реально существуют в `audio_assets`, и досинтезирует только отсутствующие клипы вместо тихого пропуска.
- **TTS hang guard**: для OpenAI TTS добавлен явный request timeout (`OPENAI_TTS_TIMEOUT_SECONDS`, default 12s), backend-параллельность по умолчанию: OpenAI 4 workers, ElevenLabs 2 workers; web отправляет TTS батчами по 6 клипов и обрывает зависший `/api/tts` батч через 30s для OpenAI / 45s для ElevenLabs с диагностикой.
- **Review summary labels**: в web Review разделены показатели text-card reuse и audio-library reuse: `reused saved cards` относится только к `generated_card_assets`, а аудио отображается как `reused audio clips` / `saved audio clips`.
- **ElevenLabs manual voiceID**: добавлен `POST /api/tts/voice/check`, который валидирует голос через серверный `ELEVENLABS_API_KEY`; Settings умеет проверить voiceID из ElevenLabs library, добавить его в dropdown и сохранить как `audio_voice`.
- **ElevenLabs model/voice discovery hardening**: список TTS-моделей ElevenLabs теперь берётся из `GET /v1/models` (`can_do_text_to_speech`) с fallback-списком, а проверка voiceID умеет fallback на `GET /v2/voices?voice_ids=...` для library/saved voices.
- **ElevenLabs shared voice import**: admin-only `POST /api/admin/tts/voice/add-shared` стал idempotent: если voiceID уже доступен server key, он просто выбирается; иначе backend пробует найти `public_owner_id` через `/v1/shared-voices?search=...`, добавить голос в server workspace и выбрать/preview его как обычный голос.
- **TTS voice preview**: добавлен `POST /api/tts/preview` и аудиоплеер в Settings для короткой проверки текущего `provider/model/voice` без полной генерации карточек и без долговременного сохранения preview audio.
- **Диагностика**: если persisted audio не найден, API возвращает явный `409` с указанием, что отсутствует в server-side storage, вместо немого провала/413 на крупном request body.
- **Тесты**: добавлены проверки `TTS -> persisted storage`, `APKG export -> persisted media reuse`, durable TTS cache-hit и generated-card cache-hit без вызова провайдера.

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
- [x] N7 — ElevenLabs voiceID UX: в Settings можно вручную указать ElevenLabs `voiceID`, проверить его через серверный API, сохранить этот голос и использовать его даже если live catalogue/filter его не показывает.
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
```

## Session scratchpad
- What I changed: made Admin ElevenLabs voice curation idempotent: already-available voice IDs now return success without Voice Library search/import; only missing voices trigger shared-voice lookup/import.
- Why: some voices are already available to the server API key but are not discoverable via Voice Library search, so admin import should not fail in that case.
- Next steps: deploy, retry Admin -> ElevenLabs voice curation with O4PMCJ0ef9FbFrmigDn4; it should select the existing server voice instead of failing search.
- Open questions: whether to add a visible `source` badge (`existing_voice` vs `shared_voice`) in the Admin success card.
