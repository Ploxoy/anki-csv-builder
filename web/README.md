# Doedutch Web (Minimal UI)

This is a minimal React + Vite UI that calls the FastAPI service.

Dev setup:

```bash
cd web
npm install
npm run dev
```

Docker (no local Node.js):

```bash
docker compose up -d db api
docker compose up web
```

Notes:
- By default, the Vite dev server proxies `/api`, `/health`, and `/docs` to `http://localhost:8000` (see `web/vite.config.ts`).
- You can override the proxy target via `VITE_API_TARGET` (Docker Compose sets it to `http://api:8000` for the `web` service).
- If your API requires `X-API-Key`, set it in the UI (stored in localStorage).
- Multi-user beta uses an invite token (`Authorization: Bearer ...`) created by an admin via `/api/admin/invite`.
- If settings don't persist, ensure the API image was rebuilt after adding `psycopg`: `docker compose build api && docker compose up -d db api`.
- Admin panel in UI: with `X-API-Key` set, you can create invites, list users, block/unblock, and rotate tokens.
