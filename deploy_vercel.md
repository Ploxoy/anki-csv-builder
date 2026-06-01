# Deploy to Vercel (Plan C: Async Generate)

This runbook deploys the current repo to Vercel with async generation:

- `POST /api/jobs/generate` (enqueue)
- `POST/GET /api/jobs/generate/worker` (process queue)
- `GET /api/jobs/generate/{job_id}` (poll status)

The web UI already uses this flow and falls back to sync `/api/generate` if job endpoints are unavailable.

## 1) Prerequisites

- GitHub branch with latest changes (including `vercel.json`, `api/index.py`, async job endpoints).
- Vercel account and project connected to this repository.
- Postgres database reachable from Vercel (`DATABASE_URL`).
- OpenAI API key.

## 2) Create Vercel project

1. In Vercel: **Add New -> Project**.
2. Import this repository and select the target branch.
3. Keep **Root Directory** as repository root.
4. Deploy once (can fail until env vars are set; that is acceptable).

## 3) Required environment variables

Set these in Vercel Project Settings -> Environment Variables:

- `DATABASE_URL` = postgres connection string
- `OPENAI_API_KEY` = OpenAI key
- `API_SHARED_SECRET` = strong random secret for admin endpoints
- `API_REQUIRE_SHARED_SECRET` = `1`
- `CRON_SECRET` = strong random secret for cron worker auth

Optional:

- `ELEVENLABS_API_KEY` = ElevenLabs key
- `GENERATE_JOB_MAX_ITEMS_PER_WORKER` = `2` (default)
- `GENERATE_JOB_STALE_SECONDS` = `300` (default)
- `API_ALLOW_LEGACY_USER_ID` = `0` (recommended)

After setting vars, redeploy.

## 4) Verify cron config

`vercel.json` already contains:

- cron path: `/api/jobs/generate/worker`
- schedule: every minute (`*/1 * * * *`)

Vercel will call it automatically with `Authorization: Bearer <CRON_SECRET>`.

## 5) Smoke test (bash/curl)

Set variables locally:

```bash
BASE_URL="https://<your-project>.vercel.app"
ADMIN_KEY="<API_SHARED_SECRET>"
```

### 5.1 Create invite token (admin)

```bash
INVITE_JSON=$(curl -sS -X POST "$BASE_URL/api/admin/invite" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ADMIN_KEY" \
  -d '{"label":"vercel-smoke"}')

echo "$INVITE_JSON"
USER_TOKEN=$(echo "$INVITE_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['token'])")
```

### 5.2 Check user identity

```bash
curl -sS "$BASE_URL/api/whoami" \
  -H "Authorization: Bearer $USER_TOKEN"
```

### 5.3 Enqueue generate job

```bash
JOB_JSON=$(curl -sS -X POST "$BASE_URL/api/jobs/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -d '{
    "run_id":"vercel-smoke-1",
    "prompt_version":"p0",
    "provider":"openai",
    "model":"gpt-4.1-mini",
    "cefr":"B1",
    "profile":"balanced",
    "l1":"EN",
    "temperature":0.4,
    "items":[
      {"id":"1","woord":"aanraken","def_nl":"iets voelen","translation":"to touch"},
      {"id":"2","woord":"begrijpen","def_nl":"snappen wat iets betekent","translation":"to understand"}
    ]
  }')

echo "$JOB_JSON"
JOB_ID=$(echo "$JOB_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
```

### 5.4 Trigger worker manually (optional, speeds up test)

```bash
curl -sS -X POST "$BASE_URL/api/jobs/generate/worker" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -d "{\"job_id\":\"$JOB_ID\",\"max_items\":2}"
```

### 5.5 Poll job status

```bash
for i in $(seq 1 40); do
  STATUS_JSON=$(curl -sS "$BASE_URL/api/jobs/generate/$JOB_ID" \
    -H "Authorization: Bearer $USER_TOKEN")
  STATUS=$(echo "$STATUS_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "status=$STATUS"
  if [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ]; then
    echo "$STATUS_JSON"
    break
  fi
  sleep 2
done
```

## 6) Smoke test (PowerShell)

```powershell
$BaseUrl   = "https://<your-project>.vercel.app"
$AdminKey  = "<API_SHARED_SECRET>"

$invite = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/admin/invite" `
  -Headers @{ "X-API-Key" = $AdminKey } `
  -ContentType "application/json" `
  -Body '{"label":"vercel-smoke"}'

$UserToken = $invite.token

$job = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/jobs/generate" `
  -Headers @{ "Authorization" = "Bearer $UserToken" } `
  -ContentType "application/json" `
  -Body '{
    "run_id":"vercel-smoke-ps",
    "prompt_version":"p0",
    "provider":"openai",
    "model":"gpt-4.1-mini",
    "cefr":"B1",
    "profile":"balanced",
    "l1":"EN",
    "temperature":0.4,
    "items":[
      {"id":"1","woord":"aanraken","def_nl":"iets voelen","translation":"to touch"},
      {"id":"2","woord":"begrijpen","def_nl":"snappen wat iets betekent","translation":"to understand"}
    ]
  }'

$jobId = $job.job_id

Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/jobs/generate/worker" `
  -Headers @{ "Authorization" = "Bearer $UserToken" } `
  -ContentType "application/json" `
  -Body "{`"job_id`":`"$jobId`",`"max_items`":2}" | Out-Null

for ($i = 0; $i -lt 40; $i++) {
  $status = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/jobs/generate/$jobId" `
    -Headers @{ "Authorization" = "Bearer $UserToken" }
  Write-Host "status=$($status.status)"
  if ($status.status -eq "done" -or $status.status -eq "failed") {
    $status | ConvertTo-Json -Depth 8
    break
  }
  Start-Sleep -Seconds 2
}
```

## 7) Export test (after status=done)

Use the `result.items` from job status and pass successful cards to export:

- `POST /api/export/csv`
- `POST /api/export/apkg`

Both endpoints require `Authorization: Bearer <user_token>`.

## 8) Troubleshooting

- `503 DATABASE_URL...`:
  - check `DATABASE_URL` in Vercel env vars.
- job stuck in `queued`:
  - check function logs for `/api/jobs/generate/worker`
  - manually call `POST /api/jobs/generate/worker`.
- `401 Unauthorized` on worker cron:
  - ensure `CRON_SECRET` exists and matches Vercel cron auth behavior.
- `failed` status with model/auth error:
  - verify `OPENAI_API_KEY`, model access, and quotas.
- UI still uses sync mode:
  - clear browser localStorage override:
    - `localStorage.setItem("use_async_generate", "1")`

## 9) Recommended next hardening

- Add retention cleanup for old `generation_jobs` (cron endpoint/admin endpoint).
- Add per-user queue limits to prevent accidental overload.
- Add structured logs for job lifecycle (`queued -> running -> done/failed`).

## 10) Go-live perf preset (Hobby + Neon, 3-5 users)

Use this baseline first, then tune only if needed.

### 10.1 Vercel project settings

- Function region: set to an EU region close to Neon DB (for example `fra1`).
- Keep API and DB in the same region family to reduce DB roundtrip overhead.

### 10.2 Environment variables

Required:

- `DATABASE_URL` = **real Postgres URL** (no placeholders)
- `OPENAI_API_KEY` = active key
- `API_SHARED_SECRET` = strong random secret
- `API_REQUIRE_SHARED_SECRET` = `1`
- `CRON_SECRET` = strong random secret

Performance defaults:

- `DB_CONNECT_RETRIES` = `1`
- `DB_CONNECT_TIMEOUT_SECONDS` = `4`
- `GENERATE_JOB_STALE_SECONDS` = `90`
- `GENERATE_JOB_MAX_ITEMS_PER_WORKER` = `3`

Optional:

- `ELEVENLABS_API_KEY` (only if ElevenLabs is used)

### 10.3 UI mode defaults

- Small/medium runs (up to ~40 rows): prefer direct sync generate (lower latency).
- Large runs: async jobs mode.

Quick override in browser console:

```js
localStorage.setItem("use_async_generate", "0") // force sync
localStorage.setItem("use_async_generate", "1") // force async
```

### 10.4 What “normal” looks like

- `sum(row latency)` should be close to total `elapsed` for sync runs.
- If `overhead >> llm sum`, time is spent in orchestration (queue/poll/auth/DB/cold starts), not in model inference.

### 10.5 Smoke performance check

1. Run 8 rows in sync mode (`use_async_generate=0`).
2. Run 31 rows in sync mode.
3. Compare:
   - `elapsed`
   - `llm sum`
   - `overhead`
4. If overhead is still large, inspect Vercel Runtime Logs:
   - function `Start Type` (`Cold`/`Hot`)
   - request durations for `/api/jobs/generate/worker` and `/api/jobs/generate/{id}`.
