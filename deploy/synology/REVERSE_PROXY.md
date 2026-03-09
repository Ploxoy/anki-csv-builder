# Internet access: Asus DDNS + Strato DNS + DSM Reverse Proxy

Use this only after LAN deployment is stable.

## Goal

- Public endpoint: `https://app.doedutch.nl` only.
- Keep raw ports private (`8000` API and `5173` web).
- Support dynamic WAN IP with Asus DDNS + Strato CNAME.

## 0) Gate check: Public IPv4 vs CGNAT (must-do)

1. Read WAN IPv4 in Asus UI (`WAN -> Internet Status`).
2. Run check from NAS shell:

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/check_wan_mode.sh <WAN_IP_FROM_ASUS>
```

Decision:
- Exit `0`: direct path is supported -> continue with this file.
- Exit `20`: likely CGNAT/double NAT -> use [`CLOUDFLARE_TUNNEL.md`](./CLOUDFLARE_TUNNEL.md).

## 1) Configure Asus router (direct path)

### 1.1 Enable DDNS

In Asus UI:
- `WAN -> DDNS`
- Enable DDNS and get hostname like `<name>.asuscomm.com`.

Put hostname in env for documentation consistency:

```bash
vi deploy/synology/.env
```

```env
PUBLIC_DOMAIN=app.doedutch.nl
ASUS_DDNS_HOST=<name>.asuscomm.com
```

### 1.2 Port forwarding

In Asus UI (`WAN -> Virtual Server / Port Forwarding`):
- TCP `443` -> `192.168.2.10:443`
- TCP `80` -> `192.168.2.10:80` (for Let's Encrypt HTTP challenge/renewal)

Do not forward:
- `8000`
- `5173`

## 2) Configure Strato DNS

For `app.doedutch.nl`:
- create `CNAME` -> `<name>.asuscomm.com`,
- remove conflicting `A`/`AAAA` records for `app`,
- keep low TTL during rollout (for example `300`).

Verify from shell:

```bash
dig +short app.doedutch.nl
```

Expected: resolves to DDNS target/IP (not stale A/AAAA).

## 3) DSM certificate + reverse proxy

### 3.1 Let's Encrypt certificate

DSM:
- `Control Panel -> Security -> Certificate`
- Request Let's Encrypt certificate for `app.doedutch.nl`
- Apply certificate to reverse proxy entry for this host.

### 3.2 Reverse proxy rule

DSM:
- `Control Panel -> Login Portal -> Advanced -> Reverse Proxy`

Create rule:
- Source:
  - Protocol: `HTTPS`
  - Hostname: `app.doedutch.nl`
  - Port: `443`
- Destination:
  - Protocol: `HTTP`
  - Hostname: `127.0.0.1`
  - Port: `5173`

Optional hard redirect rule:
- Source: `HTTP`, host `app.doedutch.nl`, port `80`
- Action: redirect to HTTPS.

## 4) DSM firewall and hardening

- Allow inbound:
  - `443` from internet
  - `80` from internet (certificate renewal; may be redirected to HTTPS)
- Restrict `22` to trusted source IPs or disable when not needed.
- Keep:
  - `API_REQUIRE_SHARED_SECRET=1`
  - strong `API_SHARED_SECRET`
- Follow key rotation guide: `notes/synology_key_rotation.md`.

## 5) Validation (external network only)

Run from LTE/5G or off-site host:

```bash
cd /volume1/docker/anki-csv-builder/app
bash deploy/synology/scripts/check_public_endpoints.sh app.doedutch.nl
```

Expected:
- `https://app.doedutch.nl/` works,
- `https://app.doedutch.nl/api/health` works,
- `http://app.doedutch.nl:8000/health` blocked,
- `http://app.doedutch.nl:5173/` blocked.
