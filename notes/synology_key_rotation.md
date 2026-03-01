# Synology DS224+ — ротация скомпрометированного ключа без потери БД

Дата: 2026-03-01

## Что делать при компрометации `OPENAI_API_KEY`

1. Отозвать старый ключ в OpenAI и создать новый.
2. На NAS обновить только `deploy/synology/.env`:

```bash
cd /volume1/docker/anki-csv-builder/app
vi deploy/synology/.env
```

Заменить:

```env
OPENAI_API_KEY=<new_key>
```

3. Перезапустить только API:

```bash
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env up -d api
```

4. Проверить health и логи:

```bash
curl -fsS http://192.168.2.123:8000/health
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env logs --tail=100 api
```

## Если скомпрометирован `API_SHARED_SECRET`

1. Обновить `API_SHARED_SECRET` в `deploy/synology/.env`.
2. Перезапустить `api`:

```bash
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env up -d api
```

3. В web UI обновить админ-ключ:
- `Settings -> Advanced access -> X-API-Key` (ввести новый `API_SHARED_SECRET`).

## Как не потерять базу пользователей

Нельзя:
- удалять `/volume1/docker/anki-csv-builder/pgdata`,
- запускать `docker compose ... down -v`,
- менять `SYNO_BASE_PATH` без миграции данных,
- менять `POSTGRES_PASSWORD` после инициализации БД без отдельной процедуры смены пароля внутри Postgres.

Можно:
- менять `OPENAI_API_KEY`, `API_SHARED_SECRET` и перезапускать только `api`.

## Быстрый чек статуса

```bash
cd /volume1/docker/anki-csv-builder/app
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env ps
```

Ожидаемо: `db`, `api`, `web` в состоянии `Up` / `healthy`.
