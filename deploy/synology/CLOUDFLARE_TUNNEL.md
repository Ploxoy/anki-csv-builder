# Cloudflare Tunnel fallback (when ISP uses CGNAT)

Use this path only when direct inbound routing (`80/443` forward to NAS) is impossible.

## Goal

- Publish `https://app.doedutch.nl` without public IPv4 on WAN.
- Keep all inbound ports closed on router.
- Let NAS open an outbound tunnel to Cloudflare.

## 1) Prepare Cloudflare side

1. Add your domain to Cloudflare (full or delegated DNS control for `app` host).
2. Open Cloudflare Zero Trust -> Access -> Tunnels.
3. Create a new tunnel (Docker connector).
4. Add public hostname:
   - Hostname: `app.doedutch.nl`
   - Service: `http://192.168.2.10:5173`
5. Copy tunnel token.

## 2) Prepare NAS env

On NAS:

```bash
cd /volume1/docker/anki-csv-builder/app
cp deploy/synology/.env.cloudflare.example deploy/synology/.env.cloudflare
vi deploy/synology/.env.cloudflare
```

Set:

```env
CLOUDFLARE_TUNNEL_TOKEN=<real-token>
```

## 3) Run cloudflared container

```bash
cd /volume1/docker/anki-csv-builder/app
docker compose -f deploy/synology/docker-compose.cloudflared.yml --env-file deploy/synology/.env.cloudflare up -d
```

Check logs:

```bash
docker compose -f deploy/synology/docker-compose.cloudflared.yml --env-file deploy/synology/.env.cloudflare logs -f cloudflared
```

## 4) Router policy in fallback mode

- Do not forward `80`/`443` to NAS.
- Do not expose `8000`/`5173`.
- Keep only required management ports (for example `22`) restricted by source IP.

## 5) Validation

From external network:

```bash
bash deploy/synology/scripts/check_public_endpoints.sh app.doedutch.nl
```

Expected:
- `https://app.doedutch.nl/` works,
- `https://app.doedutch.nl/api/health` works,
- `:8000` and `:5173` are not reachable.
