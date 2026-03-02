# Synology DS224+ Deployment (DSM UI, LAN-first)

This guide implements a 2-stage deployment model:
1. Bring up the stack in LAN via DSM Container Manager.
2. Add internet access later via Reverse Proxy + HTTPS.

## 1) Prerequisites

- DSM 7.2+ with **Container Manager** installed.
- SSH enabled (Control Panel -> Terminal & SNMP -> Enable SSH).
- NAS user with rights to:
  - `/volume1/docker/*`
  - Container Manager project creation/start/stop.
- Local machine with `git` to copy the repository, or direct `git clone` on NAS.

## 2) Copy project to NAS

Recommended target path:
- `/volume1/docker/anki-csv-builder/app`

Example from NAS SSH shell:

```bash
mkdir -p /volume1/docker/anki-csv-builder
cd /volume1/docker/anki-csv-builder
git clone <repo_url> app
cd app
```

## 3) Prepare `.env` and persistent folders

Run helper scripts from repo root on NAS:

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/prepare.sh
bash deploy/synology/scripts/validate_env.sh
```

What this does:
- creates `deploy/synology/.env` from `.env.example` (if missing),
- creates persistent folders under `SYNO_BASE_PATH`:
  - `pgdata`, `cache`, `logs`,
- checks required secrets are set.

Required values in `deploy/synology/.env`:
- `API_SHARED_SECRET`
- `OPENAI_API_KEY`
- `POSTGRES_PASSWORD`

Optional:
- `ELEVENLABS_API_KEY`

## 4) Deploy via DSM Container Manager (UI path)

1. Open **Container Manager** -> **Project** -> **Create**.
2. Choose **Create project from docker-compose file**.
3. Compose file:
   - `/volume1/docker/anki-csv-builder/app/deploy/synology/docker-compose.synology.yml`
4. Environment file:
   - `/volume1/docker/anki-csv-builder/app/deploy/synology/.env`
5. Deploy and wait until services are running:
   - `db`
   - `api`
   - `web`

Notes:
- first build can take several minutes on DS224+,
- images are built locally on NAS (no registry required).

## 5) LAN access and smoke checks

Default endpoints:
- Web UI: `http://<NAS_IP>:5173`
- API health: `http://<NAS_IP>:8000/health`

Quick smoke from NAS shell:

```bash
bash deploy/synology/scripts/smoke.sh
```

Or specify host explicitly:

```bash
bash deploy/synology/scripts/smoke.sh 192.168.1.20
```

UI smoke scenario:
1. open web UI,
2. add 1-2 words,
3. run `Generate -> Preview -> Export CSV`.

## 6) Gentle mode (lower NAS background activity)

This stack is tuned for NAS-friendly defaults:
- API access logs are disabled (`uvicorn --no-access-log`),
- docker logs are rotated (`DOCKER_LOG_MAX_SIZE`, `DOCKER_LOG_MAX_FILE`),
- healthchecks are less frequent (`DB_HEALTHCHECK_*`, `API_HEALTHCHECK_*`).

You can tune these in `deploy/synology/.env`.

For even quieter behavior, increase:
- `DB_HEALTHCHECK_INTERVAL` (for example `180s`),
- `API_HEALTHCHECK_INTERVAL` (for example `240s`).

Then apply changes:

```bash
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env up -d
```

## 7) Sleep mode (manual or scheduled)

Warm sleep (keep DB running, stop API+Web):

```bash
bash deploy/synology/scripts/sleep.sh warm
```

Deep sleep (stop all services):

```bash
bash deploy/synology/scripts/sleep.sh deep
```

Wake up:

```bash
bash deploy/synology/scripts/wake.sh
```

Tip: run these scripts from DSM Task Scheduler for night/off-hours.

## 8) Update cycle (`git pull`)

From NAS repo root:

```bash
bash deploy/synology/scripts/update.sh
```

What it does:
1. `git pull --ff-only`,
2. `docker compose ... up -d --build`,
3. `smoke.sh` post-check.

If you prefer manual update in DSM UI:
1. run `git pull --ff-only`,
2. in Project: Rebuild/Update,
3. run smoke check again.

## 9) Acceptance checklist

- Cold start: `db`, `api`, `web` all running.
- Health: `/health` is reachable after container restart.
- Core flow: Generate/Preview/Export works without API 5xx.
- Persistence: DB/cache survive container restart.
- Secrets sanity: wrong `OPENAI_API_KEY` does not crash containers, only generation fails.
- Update path: `git pull` + rebuild works with no manual compose edits.

## 10) Troubleshooting

- Ports busy (`5173` / `8000`):
  - change `WEB_PORT` / `API_PORT` in `deploy/synology/.env`, redeploy.
- Web loads, API requests fail:
  - inspect API logs and key:
  - `docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env logs -f api`
- DB not healthy:
  - verify `POSTGRES_PASSWORD`,
  - verify write access to `${SYNO_BASE_PATH}/pgdata`.

## 11) Stage 2: internet access (later)

When LAN deployment is stable, follow:
- [`REVERSE_PROXY.md`](./REVERSE_PROXY.md)

This covers DSM reverse proxy, Let's Encrypt TLS, and exposure hardening.
