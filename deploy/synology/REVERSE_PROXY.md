# Synology Reverse Proxy + HTTPS (Stage 2)

Use this only after LAN deployment is stable.

## Goal

- Expose only HTTPS (443) to the internet.
- Keep raw `:8000` API port private.
- Route public traffic to web service (`:5173`) inside NAS.

## 1) Domain and certificate

1. Point your domain/subdomain A record to your public IP.
2. In DSM, configure DDNS if needed.
3. Open DSM -> Control Panel -> Security -> Certificate.
4. Request Let's Encrypt certificate for your domain.
5. Assign the new certificate to reverse proxy destination.

## 2) Reverse proxy rule (DSM)

Open DSM -> Control Panel -> Login Portal -> Advanced -> Reverse Proxy.

Create rule:
- Source:
  - Protocol: `HTTPS`
  - Hostname: `<your-domain>`
  - Port: `443`
- Destination:
  - Protocol: `HTTP`
  - Hostname: `127.0.0.1`
  - Port: `<WEB_PORT>` (default `5173`)

## 3) Router / firewall

- Forward only `443` to NAS.
- Optional: forward `80` only for ACME HTTP challenge.
- Do not expose `8000` and `5173` publicly.
- In DSM Firewall, allow:
  - `443` from internet
  - `22` only from trusted IPs (or disable SSH after setup)

## 4) Hardening checklist

- Keep `API_REQUIRE_SHARED_SECRET=1`.
- Use strong `API_SHARED_SECRET`.
- Rotate secrets periodically.
- Monitor container logs for unauthorized/failed requests.
- Keep DSM and Container Manager updated.

## 5) Validation

From external network:
- `https://<your-domain>/` loads the web UI.
- Browser network calls to `/api/*` return expected responses.
- `https://<your-domain>/api/health` responds successfully.

From external network (must fail):
- `http://<your-domain>:8000/health`
- `http://<your-domain>:5173`
