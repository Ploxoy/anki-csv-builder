# Runbook (Synology DS224+) — 192.168.2.10

Target:
- NAS IP: `192.168.2.10`
- SSH user: `VKotenok`
- App path on NAS: `/volume1/docker/anki-csv-builder/app`

## 1) First-time setup (one time)

From your local machine:

```bash
ssh VKotenok@192.168.2.10
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
- `PUBLIC_DOMAIN=app.doedutch.nl`
- `ASUS_DDNS_HOST=<name>.asuscomm.com` (for direct path)
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
curl -fsS http://192.168.2.10:8000/health
curl -fsS http://192.168.2.10:5173/ > /dev/null && echo "web ok"
```

From NAS (inside repo):

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/smoke.sh 192.168.2.10
```

UI check:
1. Open `http://192.168.2.10:5173`
2. Generate 1-2 cards
3. Preview
4. Export CSV

## 4) Standard update procedure

From local machine:

```bash
ssh VKotenok@192.168.2.10
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

## 5) Gentle mode (reduce idle activity)

Edit:

```bash
cd /volume1/docker/anki-csv-builder/app
vi deploy/synology/.env
```

Recommended values:

```env
DOCKER_LOG_MAX_SIZE=10m
DOCKER_LOG_MAX_FILE=3
DB_HEALTHCHECK_INTERVAL=180s
API_HEALTHCHECK_INTERVAL=240s
```

Apply:

```bash
docker compose -f deploy/synology/docker-compose.synology.yml --env-file deploy/synology/.env up -d
```

## 6) Sleep mode

Warm sleep (DB stays up):

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/sleep.sh warm
```

Deep sleep (all containers stopped):

```bash
bash deploy/synology/scripts/sleep.sh deep
```

Wake:

```bash
bash deploy/synology/scripts/wake.sh
```

For automation, schedule these commands in DSM Task Scheduler.

## 7) Quick diagnostics

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

## 8) Stage 2 (internet access)

Gate check (must pass for direct routing):

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/check_wan_mode.sh <WAN_IP_FROM_ASUS>
```

If exit code is `0`:
1. Configure Asus DDNS and port forwarding (`80`, `443`) to `192.168.2.10`.
2. At Strato create `CNAME`:
   - `app.doedutch.nl -> <name>.asuscomm.com`
3. In DSM issue Let's Encrypt cert for `app.doedutch.nl`.
4. In DSM create reverse proxy:
   - `HTTPS app.doedutch.nl:443 -> HTTP 127.0.0.1:5173`.

If exit code is `20`:
- use Cloudflare fallback guide:
  - `deploy/synology/CLOUDFLARE_TUNNEL.md`

External validation (run from LTE/5G):

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/check_public_endpoints.sh app.doedutch.nl
```

## 9) Windows LAN deploy pipeline (PowerShell, update-only)

From a Windows machine in LAN:

```powershell
pwsh -File .\deploy\synology\Deploy-FromLan.ps1
```

Default target values:
- `NasHost=192.168.2.10`
- `NasUser=VKotenok`
- `ProjectPath=/volume1/docker/anki-csv-builder/app`
- `Branch=dev`
- `SshKeyPath=~/.ssh/id_ed25519`

Example with explicit key path:

```powershell
pwsh -File .\deploy\synology\Deploy-FromLan.ps1 -SshKeyPath "C:\Users\<you>\.ssh\id_ed25519"
```

What pipeline does:
1. Local preflight (`ssh` in PATH, key exists, SSH ping).
2. Remote update (`git fetch/checkout/pull --ff-only`).
3. Env validation + `docker compose up -d --build`.
4. Remote smoke check + compose status.
5. Local HTTP checks (`/health`, web UI) with retry window.
