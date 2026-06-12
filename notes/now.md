# Now — Anki CSV Builder

Updated: 2026-06-12T00:00:00

## Quick pointers
- notes/status.md (project status)
- notes/tasks.md (task tracker)
- notes/vision_v2.md (product direction)

## Git status
```
## dev...origin/dev
 M notes/status.md
 M notes/tasks.md
```

## Recent commits
```
88fccca multithreadintg
3d11f55 time outs fix
c47b3af power sfe modes
37c4a79 deploy pipline
8904787  implement silent mode
```

## Status (head)
```
# Status — Anki CSV Builder

*(обновлено: 2026-03-30)*

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

## Свежие изменения (март 2026)
- **Synology deploy/docs**: обновлён пакет деплоя для NAS — `deploy/synology/REVERSE_PROXY.md` (gate-check public IP vs CGNAT), `deploy/synology/CLOUDFLARE_TUNNEL.md` (fallback), `deploy/synology/docker-compose.cloudflared.yml`, шаблон `deploy/synology/.env.cloudflare.example`.
- **Проверочные скрипты**: добавлены `deploy/synology/scripts/check_wan_mode.sh` и `deploy/synology/scripts/check_public_endpoints.sh` для верификации internet-stage.
- **Runbook**: зафиксирован персональный чеклист `deploy/synology/RUNBOOK_192.168.2.10.md` c актуальным пользователем `VKotenok` и шагами внешнего доступа.
- **Windows/LAN pipeline**: добавлен `deploy/synology/Deploy-FromLan.ps1` (update-only) — SSH preflight, `git fetch/checkout/pull --ff-only`, `validate_env`, `docker compose up --build`, smoke + локальные HTTP health-checks с retry.
- **Web TTS options UX**: `Reload model list` в web теперь обновляет и text-модели, и TTS-модели/голоса из backend; добавлен тихий авто-refresh списка по мере работы backend.
- **API TTS options**: `_filter_openai_tts_models` сначала возвращает live-discovered список, fallback к дефолтным моделям используется только если discovery недоступен/пуст.
- **Тесты**: расширен `tests/test_api_tts.py` (покрытие фильтра TTS-моделей).
- **Synology power-save stack (2026-03-30)**:
  - Добавлен front-door сервис `waker` + `socket-proxy` в `deploy/synology/docker-compose.synology.yml` для авто-пробуждения и авто-сна (`WAKER_IDLE_*`).
  - `web` теперь публикуется через `waker`, добавлен статус-эндпойнт `/_waker/status`.
  - Добавлены one-command install сценарии: `deploy/synology/scripts/install.sh` (NAS) и `deploy/synology/scripts/install.ps1` (Windows -> SSH sync + remote install).
  - Скрипты `sleep.sh` / `wake.sh` / `update.sh` переведены на общий helper `deploy/synology/scripts/docker_cmd.sh` для DSM-окружений с нестандартным `PATH`.
  - Введён hotfix таймаутов для длинных запусков: `WAKER_PROXY_TIMEOUT_SECONDS=600` + таймауты в `web/deploy/nginx.synology.conf`, чтобы снизить HTTP 504 при длинной генерации.
- **Resolved long-run 504 (2026-03-30)**: подтверждён успешный запуск длинного списка (52 записи, audio off). Корневая причина была составной: один длинный HTTP-запрос `/api/generate` и неприменённый deploy (compose не выполнялся из-за `docker.sock` permissions без `sudo`). Исправление: batched text-generation в `web/src/App.tsx` + перезапуск стека через `sudo`.

## Свежие изменения (февраль 2026)
- Deep UI Rework v1 (web): интерфейс переведён на light theme по `notes/Doedutch_UI_Guide.md`, логика вкладок сохранена (`Generate / Settings / Admin`).
- `web/src/App.tsx` декомпозирован на `AppShell`, табовые фичи (`GenerateTab/SettingsTab/AdminTab`), `Notice` и `ProgressPanel`; добавлены `web/src/lib/uiState.ts` и `web/src/lib/messages.ts`.
- Сообщения/ошибки изолированы по вкладкам и секциям (scoped notices), убрано глобальное смешивание статусов между Generate/Settings/Admin.
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
- [x] D1 — Synology internet-stage toolkit: gate-check/reachability скрипты (`check_wan_mode.sh`, `check_public_endpoints.sh`), direct path docs (`REVERSE_PROXY.md`) и Cloudflare fallback (`CLOUDFLARE_TUNNEL.md`, `docker-compose.cloudflared.yml`).
- [x] D2 — Windows LAN deployment pipeline: `deploy/synology/Deploy-FromLan.ps1` (SSH-key preflight, `git pull --ff-only`, `validate_env`, `compose up --build`, smoke + local health retries) + обновлён runbook `RUNBOOK_192.168.2.10.md`.
- [x] D3 — Power-save layer на Synology: добавлены `waker` + `socket-proxy` (auto sleep/wake, front-door на WEB_PORT), новые env-параметры `WAKER_IDLE_*` и статус `/_waker/status`.
- [x] D4 — One-command install для Synology: `deploy/synology/scripts/install.sh` (NAS) и `deploy/synology/scripts/install.ps1` (Windows -> SSH sync + remote install), плюс унификация docker-вызовов через `deploy/synology/scripts/docker_cmd.sh`.
- [x] D5 — Timeout hotfix для длинных запусков: `WAKER_PROXY_TIMEOUT_SECONDS=600` + timeout в `web/deploy/nginx.synology.conf` (снижение риска HTTP 504 на длинной генерации).
- [x] D6 — Инцидент после внедрения sleep/wake закрыт: проблема 504 на длинной text-generation воспроизводилась при одном длинном `/api/generate`; исправлено батчевой генерацией в `web/src/App.tsx` + корректным `sudo`-перезапуском контейнеров на Synology; подтверждён успешный прогон длинного списка (52 записи, audio off).

## 🧪 Beta readiness (Phase 0.5)
- [x] B1 — Invite-token auth v0: `/api/admin/invite` (admin) + `Authorization: Bearer <token>` (user), без Supabase.
- [x] B2 — Persist settings v1: `/api/settings` (read/write) + хранение `settings_json` по user_id в Postgres.
- [x] B3 — Usage view v0: `/api/usage` (read-only) на базе `usage_events` (по user_id/run_id).
- [x] B4 — Web UI: хранить invite token локально, грузить/сохранять settings и показывать usage.
- [x] B5 — Admin в web: создавать инвайты, листать пользователей, блок/разблок, ротация токена, просмотр usage по user_id.

## 🚀 Productization (vNext)
```

## Session scratchpad
- What I changed:
- Updated `notes/tasks.md`: marked D6 as completed (504 incident closed).
- Updated `notes/status.md`: added closure note for long-run 504 and switched next-step #1 to monitoring.
- Regenerated this file via `python scripts/update_context.py`.
- Why:
- Capture that long list generation now works in production conditions after deploy + web batching fix.
- Next steps:
- Monitor a few additional long runs (>50 rows) to confirm stability and usage consistency.
- Keep Synology deploy path with `sudo` for compose operations to avoid false-positive "updated but not rebuilt" states.
- Open questions:
- None critical; incident is currently considered resolved based on successful long-list run.

- Current focus: Vercel export path now uses persisted TTS media in Postgres (`run_media_assets`) so APKG downloads do not need a huge base64 request body.
