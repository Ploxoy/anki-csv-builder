# Runbook (Synology DS224+) — 192.168.2.123

Target:
- NAS IP: `192.168.2.123`
- SSH user: `VKotenol`
- App path on NAS: `/volume1/docker/anki-csv-builder/app`

## 1) First-time setup (one time)

From your local machine:

```bash
ssh VKotenol@192.168.2.123
```

On NAS:

```bash
mkdir -p /volume1/docker/anki-csv-builder
cd /volume1/docker/anki-csv-builder
git clone <repo_url> app
cd app

bash deploy/synology/scripts/prepare.sh
```

Edit env:

```bash
vi deploy/synology/.env
```

Required values in `.env`:
- `POSTGRES_PASSWORD` (strong)
- `API_SHARED_SECRET` (strong)
- `OPENAI_API_KEY` (real key)
- optional `ELEVENLABS_API_KEY`

Validate:

```bash
bash deploy/synology/scripts/validate_env.sh
```

## 2) Deploy from DSM UI (LAN-first)

In DSM:
1. Open `Container Manager -> Project -> Create`.
2. Select compose file:
   - `/volume1/docker/anki-csv-builder/app/deploy/synology/docker-compose.synology.yml`
3. Select env file:
   - `/volume1/docker/anki-csv-builder/app/deploy/synology/.env`
4. Click `Deploy`.
5. Wait until `db`, `api`, `web` are `Running`.

## 3) Smoke checks

From local machine:

```bash
curl -fsS http://192.168.2.123:8000/health
curl -fsS http://192.168.2.123:5173/ > /dev/null && echo "web ok"
```

From NAS (inside repo):

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/smoke.sh 192.168.2.123
```

UI check:
1. Open `http://192.168.2.123:5173`
2. Generate 1-2 cards
3. Preview
4. Export CSV

## 4) Standard update procedure

From local machine:

```bash
ssh VKotenol@192.168.2.123
```

On NAS:

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/update.sh
```

What `update.sh` does:
1. `git pull --ff-only`
2. `docker compose ... up -d --build`
3. post-update smoke check

## 5) Quick diagnostics

On NAS:

```bash
cd /volume1/docker/anki-csv-builder/app
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env ps
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env logs -f api
```

Common fixes:
- Port conflict: change `API_PORT` / `WEB_PORT` in `deploy/synology/.env`, redeploy project.
- API errors with running containers: verify `OPENAI_API_KEY` and `API_SHARED_SECRET`.
- DB not healthy: verify `POSTGRES_PASSWORD` and write permissions on `/volume1/docker/anki-csv-builder/pgdata`.

## 6) Stage 2 (later, internet access)

After LAN is stable, configure reverse proxy + TLS:
- `deploy/synology/REVERSE_PROXY.md`
