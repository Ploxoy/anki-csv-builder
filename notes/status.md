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
- **Vercel Plan C foundation (2026-05-31)**:
  - В API добавлены async endpoints для генерации: `POST /api/jobs/generate`, `GET /api/jobs/generate/{job_id}`, `POST/GET /api/jobs/generate/worker`.
  - В Postgres добавлена очередь `generation_jobs` (payload/state/result/progress) и helper-функции в `core/db.py`.
  - Web `onGenerate` переведён на job-пайплайн (enqueue + worker tick + poll status) с fallback на старый sync `/api/generate`, если job-endpoints недоступны.
  - Для Vercel добавлены `vercel.json`, `api/index.py`, `api/requirements.txt` (единая FastAPI функция + SPA build + cron route для worker).

## Свежие изменения (февраль 2026)
- Deep UI Rework v1 (web): интерфейс переведён на light theme по `notes/Doedutch_UI_Guide.md`, логика вкладок сохранена (`Generate / Settings / Admin`).
- `web/src/App.tsx` декомпозирован на `AppShell`, табовые фичи (`GenerateTab/SettingsTab/AdminTab`), `Notice` и `ProgressPanel`; добавлены `web/src/lib/uiState.ts` и `web/src/lib/messages.ts`.
- Сообщения/ошибки изолированы по вкладкам и секциям (scoped notices), убрано глобальное смешивание статусов между Generate/Settings/Admin.
- В Generate оставлен один primary action, flow перестроен в явную последовательность `Input → Run → Review → Export`.
- В Settings добавлен `dirty state` с `Save / Revert / Reload`, блоки переупорядочены (`Access`, `Generation defaults`, `Audio defaults`, `Export defaults`).
- В Admin усилена структура: `User management` + `Admin usage`, читаемая таблица пользователей и локальная карточка invite/rotate с copy-action.
- Зафиксирован checkpoint `e7f8e94`: merged scope по web+api (export endpoints, dynamic TTS options, resilient audio flow).
- TTS Reliability/UX hardening в FastAPI + web:
  - `/api/tts` теперь возвращает clip-level `status` (`ok|failed|cached`) и `error` для failed-клипов.
  - В synthesis-пайплайне добавлен один автоматический retry только для транзитных ошибок (`429/5xx/timeout`) с backoff; валидационные ошибки не ретраятся.
  - В web-прогрессе добавлены явные поля `stage`, `done/total`, `batch`, `elapsed` и индикатор `waiting provider...` для длинных батчей.
  - При partial audio UI показывает компактный summary + first error + переключатель `Show all errors`; экспорт CSV/APKG доступен с явным предупреждением о неполной озвучке.
- Контракты и тесты синхронизированы: добавлены API/core тесты для clip-status и retry-политики.
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
- Поддержка `response_format` теперь пробуется один раз на модель и кешируется между сессиями (`cache/response_format.json`), UI напоминает о падении на schema и разрешает принудительное включение.
- Инструкции/промпт перегруппированы: статический пролог вынесен в `compose_instructions_en`, `input_text` содержит общий блок `DATA/SCHEMA`, добавлено логирование (hash, tokens, cached) в `core/generation`.
- OpenAI prompt caching: модель `gpt-5` пока возвращает `cached_tokens=0`, несмотря на стабильный префикс (~900 токенов). Канал наблюдения (лог + Run report) оставляем включённым, ждём поддержки от API.
- Экспорт .apkg теперь может включать три сабдека в одном пакете: `Cloze`, `Basic (and reversed)`, `Type In`. Для дополнительных колод применён общий стиль и озвучка добавляется к NL‑стороне.
- HTML/CSS шаблоны Anki вынесены в `config/templates/*`; `config/settings.py` хранит пути, а экспорт загружает их лениво (включая заготовки для Basic и Type In).
- Базовое имя колоды теперь `Dutch` (вместо `Dutch • Cloze`).
- CSS карточек упрощён и унифицирован; удалены лишние правила, выравнивание и размеры согласованы между типами.
- Обновлены настройки `config.settings` (провайдеры, дефолтные ключи), `core/audio.py` (fetch_elevenlabs_voices, AudioSynthesisSummary.provider), `requirements.txt` (requests).
- Появился модуль `app/generation_section.py`: оркестрация запуска и прогресс вынесены из `generation_page.py`, файл стал чистым маршрутизатором UI-секций.
- Превью и блок экспорта уже живут в выделенных панелях (`preview_panel`, `export_panel`), повторное использование упростилось.
- Управление батчами переведено на `RunController`, кнопки вынесены в `_render_run_controls`.
- Основная генерация вынесена в `BatchRunner`, что убрало вложенные функции из `render_generation_page` и упростило повторное использование.
- Промт для генерации карточек уточнён: теперь явно поддерживаются multi-word выражения/коллокации в `L2_word`, разрешена естественная пунктуация (включая многоточия), описано поведение для случаев, когда `given_L2_definition` — контекстное NL-предложение, и зафиксировано требование к артиклю в `L2_word` для существительных.
- Try demo/демо-набор обновлён: теперь содержит несколько показательных случаев (чистая лемма с определением, лемма с контекстным предложением во втором поле, multi-word выражение в первом поле, существительное во множественном числе без артикля, глагол в прошедшем времени, разделяемый глагол во входе как «finite+particle»), чтобы наглядно проверить новое поведение промта и генерации.
- Дополнительно уточнены правила TARGET FORM/CLOZE: явно прописано, что для существительных во входе во множественном числе без артикля `L2_word` должен быть в виде «de/het + singular lemma» (с примерами), для глаголов во входе в прошедшем времени `L2_word` — инфинитив, для multi-word выражений типа «naar bed gaan» в cloze требуется multi-span c1 одновременно на глаголе и дополнении (конкретный GOOD/BAD пример с Ik {{c1::ga}} … {{c1::naar bed}} …), а для разделяемых глаголов, переданных во входе как «finite+particle» (например, `kwam erachter`), `L2_word` нормализуется до инфинитива с частицей (`achter komen`), при этом в cloze клоузятся именно формы в предложении: "Hij {{c1::kwam}} {{c1::erachter}} …".
- Manual editor упрощён: убран авто-добавляемый «вечно пустой» хвостовой ряд, который каждый rerun менял форму таблицы и мог приводить к потере только что введённого текста при переходе между ячейками; теперь `data_editor` работает с стабильным списком строк, а динамическое добавление/удаление делается средствами самого компонента.
- Шаблон Anki обновлён: блок перевода на L1 теперь использует контрастный цвет для светлой/тёмной тем и подчёркивает слово, чтобы текст на AnkiWeb не «сливался» с фоном.

## Соответствие видению
- Пункты 14.3–14.6 Vision синхронизированы: архитектура TTS отражает текущую реализацию (OpenAI + ElevenLabs + кеши).
- Roadmap скорректирован: в краткосрочных шагах остались streaming, UI refactor, manual editor, URL import, отчёты.
- Риски в Vision дополнены жёсткими лимитами ElevenLabs (см. 12.1).

## Ограничения и риски
- **Контракт состояния**: `generation_section` опирается на `st.session_state` (input_data/results/run_stats); нужно зафиксировать интерфейсы перед дальнейшими UI-рефакторами.
- **Vercel + длинные списки**: проблему больших списков и Vercel считаем пока не полностью решённой. Text-cache и audio-cache снижают повторную стоимость, но длинный TTS всё ещё идёт через серию HTTP batch-запросов и может упираться в function timeout/provider latency. Нужен durable resume/job слой для генерации и особенно TTS.
- **Streaming/job model**: прогресс TTS обновляется батчами, нет durable job/resume на уровне отдельного audio clip; при сбое можно продолжить за счёт кэша, но нет полноценного resumable run на 1000+ строк.
- **Каталог ElevenLabs**: фильтр по NL может вернуть мало голосов — отображаем fallback, но хорошо бы добавить режим «все голоса».
- **ElevenLabs voice library UX**: ручное добавление `voiceID` в Settings реализовано; следующая итерация — preview/quality labels и более удобная библиотека избранных голосов.
- **Качество TTS**: OpenAI даёт надёжный baseline, но голоса звучат плоско; ElevenLabs выразительнее, однако требует отдельного биллинга по токенам/месяцам, и оба провайдера иногда читают отдельные слова с англоязычным акцентом.
- **AnkiWeb + Chrome forced dark mode**: если в Chrome включён «auto dark theme for sites»/`chrome://flags/#enable-force-dark`, встроенный CSS AnkiWeb перекрашивает контент в белый, и наши cloze/def поля становятся невидимыми. Решение: отключить forced dark (или использовать стандартный режим/Edge). В шаблоны вмешиваться не планируем.
- **Инцидент 504 после внедрения sleep/wake (закрыт)**: проблема подтверждена и закрыта; длинный список теперь проходит успешно после применения batched `/api/generate` и корректного `sudo`-перезапуска контейнеров.

## Следующие шаги (предлагаемые)
1. **Long-list reliability on Vercel**: спроектировать durable `run_items`/`audio_jobs` resume flow, чтобы списки 1000+ строк и длинная озвучка не зависели от одного browser/API session.
2. **Provider abstraction**: проработать подключение альтернативных провайдеров для генерации текстов и генерации аудио через единые `TextProvider`/`TTSProvider`, usage/cost и capability discovery.
3. **ElevenLabs voice library UX v2**: добавить избранные голоса, preview и quality labels поверх уже реализованного manual `voiceID`.
4. **Review audio preview**: в раскрывающейся JSON-сводке по карточке добавить предпрослушивание `AudioWord`/`AudioSentence`.
5. **TTS-опыт**: curated список голосов, предпрослушка, быстрые метки качества/акцента.
6. **Users + персональные настройки + учёт usage (Phase 0.5, без платежей)**: продолжить invite-token auth/settings/usage как текущую beta-модель до полноценного auth/billing.
7. **Обновление прайс-листа**: автоматизировать/проверять `config/pricing.py` при появлении новых моделей и дублировать краткую инструкцию в README.
8. **Vision 2.0**: держать актуальным `notes/vision_v2.md`.

## Проверка окружения (smoke)
1. `pip install -r requirements.txt`
2. `pytest`
3. `streamlit run app/app.py`
4. Try demo → Generate, затем в панели «🔊 Audio» синтезировать слова+предложения для OpenAI и ElevenLabs (при наличии ключа).
