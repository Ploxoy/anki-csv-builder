# 📘 Anki CSV Builder

FastAPI-сервис для генерации голландских карточек Anki (генерация + TTS) на базе OpenAI Responses API.

Streamlit UI считается legacy и постепенно выводится из эксплуатации.

## 🚀 Возможности

- **Автоматическая генерация CEFR-согласованных cloze-карточек** с предложениями, переводами, определениями и коллокациями
- **Гибкий ввод**: Markdown-таблицы, TSV/CSV, простой текст или встроенный ручной редактор
- **Проверка слов и подсветка флагов** с опцией принудительной генерации спорных записей
- **Сбалансированные сигнал-слова и поддержка разделяемых глаголов**, включая детерминированный выбор по seed
- **Умный выбор моделей OpenAI** и автоматический fallback при неподдержке `response_format`
- **Экспорт в CSV и .apkg** с дополнительными сабдеками Basic / Type In, использующими единый стиль и озвучку
- **Опциональная озвучка** — синтез MP3 для слова и предложения (OpenAI TTS и ElevenLabs) с кешем, ретраями и привязкой голосов к карточкам

## 🧭 UI поток

1. **Generate** — запуск пакетной генерации с прогрессом
2. **Preview & fix** — просмотр карточек, флагов и быстрых правок
3. **Audio (optional)** — выбор провайдера/голоса и запуск TTS; экспорт автоматически подхватывает файлы
4. **Export deck** — выгрузка CSV и/или `.apkg`, включая дополнительные сабдеки

## 📋 Структура карточки

Каждая карточка содержит:

- `woord` — целевое голландское слово
- `cloze_sentence` — голландское предложение с cloze-разметкой
- `ru_sentence` — перевод предложения (в UI можно выбрать другие L1)
- `collocaties` — три частотные коллокации
- `def_nl` — определение на голландском
- `ru_short` — короткий глосс на выбранном L1

## ⚙️ Установка

```bash
git clone <repository-url>
cd anki-csv-builder
pip install -r requirements.txt
```

## ▶️ Запуск приложения (legacy Streamlit UI)

Предпочтительный вход:

```bash
streamlit run app/app.py
```

Унаследованный шим (для совместимости):

```bash
streamlit run anki_csv_builder.py
```

## ▶️ Запуск API (FastAPI)

Локально (из корня репозитория):

```bash
export API_SHARED_SECRET="change-me"
export OPENAI_API_KEY="..."
# (опционально, если используете ElevenLabs для TTS)
export ELEVENLABS_API_KEY="..."
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Или через модуль:

```bash
python -m api
```

Для локальной разработки можно временно отключить проверку `X-API-Key`:

```bash
export API_REQUIRE_SHARED_SECRET=0
```

Проверка:

```bash
curl http://localhost:8000/health
curl -H "X-API-Key: $API_SHARED_SECRET" http://localhost:8000/docs >/dev/null
```

### Phase 0.5: user settings + usage (beta)

Для multi-user beta используются **инвайт‑токены**:
- Админ создаёт инвайт через `/api/admin/invite` (нужен `X-API-Key: API_SHARED_SECRET`) и получает `token`.
- Пользователь вставляет `token` в web UI (`web/`) и дальше запросы идут с `Authorization: Bearer <token>`.

Пример (создать инвайт):
```bash
export API_SHARED_SECRET="change-me"
curl -sS -X POST http://localhost:8000/api/admin/invite \
  -H "X-API-Key: $API_SHARED_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"label":"alice"}'
```

Если `Load/Save settings` “не сохраняет”, почти всегда причина одна из двух:
- `api` контейнер собран без `psycopg` (нужно пересобрать образ после обновления зависимостей)
- `db` контейнер не запущен/недоступен

Быстрый фикс:
```bash
docker compose build api
docker compose up -d db api
```

> В Phase 1 (Vision 2.0) ключи провайдеров не передаются от клиента:
> `OPENAI_API_KEY`/`ELEVENLABS_API_KEY` должны быть только в env сервера.
> Для beta пользователь аутентифицируется токеном (`Authorization: Bearer ...`), а `X-API-Key` нужен только админу (и для legacy/dev режима).

## 🐳 Docker Compose (API + Postgres)

```bash
docker compose build api
docker compose up -d db api
```

Порты можно переопределить через переменные окружения:
- `API_PORT` (по умолчанию `8000`)
- `DB_PORT` (по умолчанию `5432`)
- `WEB_PORT` (по умолчанию `5173`)

Примечание про `DATABASE_URL`:
- Для Docker Compose используется внутренняя строка подключения к Postgres: `postgresql://...@db:5432/...`
- Если у вас в `.env` есть `DATABASE_URL=...@localhost:5432/...` (для локального запуска без Docker), это **не должно** ломать Compose: в compose-файле используется `DATABASE_URL_DOCKER` (опционально) вместо `DATABASE_URL`.

### Админ-операции (beta)
- Создать инвайт: `POST /api/admin/invite` (`X-API-Key` обязателен) → `{ user_id, token }`.
- Список пользователей: `GET /api/admin/users` (`X-API-Key`).
- Блокировка/разблокировка: `POST /api/admin/users/{user_id}/status` с `{ "status": "blocked|active" }`.
- Ротация токена: `POST /api/admin/users/{user_id}/rotate` → новый `token`.

### Docker secrets (рекомендуется)

1) Создайте локально (не коммитьте) файлы:

- `secrets/API_SHARED_SECRET`
- `secrets/OPENAI_API_KEY`
- `secrets/ELEVENLABS_API_KEY`

2) Запустите Compose с overlay-файлом:

```bash
docker compose -f docker-compose.yml -f docker-compose.secrets.yml up -d --build db api
```

### Synology DS224+ (Container Manager)

Для NAS есть отдельный deployment-набор:

- `deploy/synology/docker-compose.synology.yml`
- `deploy/synology/.env.example`
- `deploy/synology/README.md`
- `deploy/synology/REVERSE_PROXY.md`
- `deploy/synology/CLOUDFLARE_TUNNEL.md`
- `deploy/synology/RUNBOOK_192.168.2.10.md` (персональный чеклист)
- `deploy/synology/scripts/*` (prepare / validate / smoke / update / sleep / wake / check_wan_mode / check_public_endpoints)

Быстрый запуск (через SSH на NAS):

```bash
cd /volume1/docker/anki-csv-builder/app
git pull --ff-only
bash deploy/synology/scripts/prepare.sh
bash deploy/synology/scripts/validate_env.sh
```

Деплой через DSM UI:
1. `Container Manager -> Project -> Create`
2. Compose file: `/volume1/docker/anki-csv-builder/app/deploy/synology/docker-compose.synology.yml`
3. Env file: `/volume1/docker/anki-csv-builder/app/deploy/synology/.env`
4. `Deploy`

Smoke-check:

```bash
bash deploy/synology/scripts/smoke.sh
```

Обновления:

```bash
bash deploy/synology/scripts/update.sh
```

CLI-альтернатива запуска:

```bash
cp deploy/synology/.env.example deploy/synology/.env
# заполните секреты в deploy/synology/.env
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env up -d --build
```

## 🌐 Минимальный веб-интерфейс (React + Vite)

Минимальный UI лежит в `web/` и ходит в FastAPI.

### Вариант A: локально (нужен Node.js)

```bash
cd web
npm install
npm run dev
```

### Вариант B: через Docker (без Node.js на хосте)

```bash
docker compose up -d db api
docker compose up web
```

Откройте UI: `http://localhost:5173`.

По умолчанию Vite проксирует `/api` на `http://localhost:8000` (см. `web/vite.config.ts`). В Docker Compose для `web` автоматически выставляется `VITE_API_TARGET=http://api:8000`, поэтому CORS не требуется.

## 🗂 Структура проекта

```
anki-csv-builder/
├── api/                 # FastAPI сервис
├── app/                 # Модули Streamlit UI
├── core/                # Парсинг, генерация, санитайзинг, экспорт
├── config/              # Настройки, шаблоны, группы сигнал-слов, i18n
├── notes/               # Статус, vision, спецификации
├── tests/               # Юнит-тесты и примеры входных данных
├── README.md            # Документация на русском
├── README.en.md         # Документация на английском
└── requirements.txt     # Зависимости
```

## 🛠 Конфигурация

- `config/settings.py` — доступные модели, UI-дефолты, задержки, пути к шаблонам
- `config/templates/` — HTML/CSS-шаблоны для колод Cloze, Basic и Type In
- `config/signalword_groups.py` — группы сигнал-слов по уровням CEFR

## 🩺 Диагностика запуска (если “сервисы не стартуют”)

Быстрый чек окружения без внешних запросов:

```bash
python scripts/doctor.py
```

## 📥 Форматы ввода

### Markdown-таблица

```markdown
| woord    | definitie NL | RU      |
|----------|--------------|---------|
| aanraken | iets voelen  | трогать |
```

### TSV (таб-разделитель)

```
aanraken	iets voelen	трогать
begrijpen	snappen	понимать
```

### Простой текст

```
aanraken - iets voelen - трогать
begrijpen - snappen - понимать
```

## 🔐 Настройка OpenAI и ElevenLabs API

1. Получите ключи на https://platform.openai.com и (опционально) https://elevenlabs.io.
2. Сохраните их в `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "ваш-openai-ключ"
ELEVENLABS_API_KEY = "ваш-elevenlabs-ключ"
```

…или введите вручную в сайдбаре Streamlit.

## 🤖 Поддерживаемые семейства моделей

- `gpt-5*` — максимальное качество
- `gpt-4.1*` — баланс скорости и качества
- `gpt-4o*` — быстрее и дешевле
- `o3*` — альтернативы с упором на рассуждения

## 📈 Как работать

1. Загрузите список слов или используйте демо-набор.
2. Выберите модель OpenAI и при необходимости скорректируйте настройки.
3. Нажмите **Generate**, наблюдайте прогресс и проверьте превью.
4. (Опционально) Откройте **Audio** и сгенерируйте MP3 для слова/предложения.
5. Скачайте CSV или `.apkg` и импортируйте в Anki.

### 🔧 Дополнительные инструменты

- **Manual editor** — ручное редактирование списка перед запуском.
- **Quality flags** — подсказки, почему слово отмечено; флаг “Force generate…” принудительно генерирует его.
- **Signal-word seed** — фиксирует выбор соединительных слов между запусками.
- **Audio presets** — выбор провайдера, голосов и инструкций отдельно для предложений и слов.
- **Random voice per card** — опция, закрепляющая случайный голос за каждой карточкой.

## 📤 Импорт в Anki

1. Откройте Anki Desktop и нужную колоду.
2. File → Import …
   - Для CSV: выберите `anki_cards.csv`, Type = Notes (Cloze), разделитель `|`.
   - Для APKG: укажите `dutch_cloze.apkg` — колода создаётся сразу.
3. Проверьте сопоставление полей (`L2_word` → Cloze).
4. Нажмите **Import** и просмотрите карточки.

## 🩹 Частые проблемы

- **Неверный API-ключ** — проверьте `.streamlit/secrets.toml` или поле в сайдбаре.
- **Медленная генерация** — переключитесь на более быструю модель `gpt-4o` или уменьшите размер батча.
- **Ошибки схемы** — приложение повторяет запрос без `response_format`; при повторных сбоях попробуйте другую модель.
- **Нет голосов ElevenLabs** — воспользуйтесь кнопкой обновления каталога или fallback-предустановками.

## ⚡ Нюансы производительности

- Задержка между запросами по умолчанию — 100 мс (меняется в `config/settings.py`).
- Превью показывает первые 20 карточек; экспорт содержит все успешные элементы.
- Для ElevenLabs применяется ограничение ≤2 параллельных запросов.

## 🤝 Как внести вклад

1. Форкните репозиторий.
2. Создайте ветку с задачей.
3. Реализуйте изменения и добавьте тесты, если возможно.
4. Откройте pull request.

## 📄 Лицензия

MIT License.

## 💬 Поддержка

Создайте issue, если нашли проблему или хотите обсудить идею.

## 📜 Основные изменения

Полный журнал — в `notes/status.md`. Последние обновления:

- Переписана аудио-панель с поддержкой OpenAI + ElevenLabs, пресетами и подробными отчётами.
- Шаблоны Anki перенесены в `config/templates/*` и лениво подгружаются при экспорте.
- Экспорт `.apkg` собирает сабдеки Cloze, Basic (reversed) и Type In с единым стилем и аудио.
