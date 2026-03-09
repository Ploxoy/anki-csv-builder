#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"

read_env_var() {
  local file="$1"
  local key="$2"
  local value
  value="$(grep -E "^${key}=" "$file" | tail -n 1 | cut -d '=' -f 2- || true)"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "$value"
}

fetch_public_ip() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 8 https://ifconfig.me
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q -T 8 -O - https://ifconfig.me
    return
  fi
  echo "ERROR: curl or wget is required." >&2
  exit 1
}

is_ipv4() {
  local ip="$1"
  [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || return 1
  IFS='.' read -r o1 o2 o3 o4 <<<"$ip"
  [[ "$o1" -le 255 && "$o2" -le 255 && "$o3" -le 255 && "$o4" -le 255 ]]
}

is_private_or_cgnat() {
  local ip="$1"
  IFS='.' read -r o1 o2 _ _ <<<"$ip"

  # 10.0.0.0/8
  if [[ "$o1" -eq 10 ]]; then
    return 0
  fi

  # 172.16.0.0/12
  if [[ "$o1" -eq 172 && "$o2" -ge 16 && "$o2" -le 31 ]]; then
    return 0
  fi

  # 192.168.0.0/16
  if [[ "$o1" -eq 192 && "$o2" -eq 168 ]]; then
    return 0
  fi

  # 100.64.0.0/10 (CGNAT)
  if [[ "$o1" -eq 100 && "$o2" -ge 64 && "$o2" -le 127 ]]; then
    return 0
  fi

  return 1
}

WAN_IP="${1:-}"

if [[ -z "$WAN_IP" && -f "$ENV_FILE" ]]; then
  WAN_IP="$(read_env_var "$ENV_FILE" "ROUTER_WAN_IP")"
fi

if [[ -z "$WAN_IP" ]]; then
  echo "Usage: bash deploy/synology/scripts/check_wan_mode.sh <WAN_IP_from_Asus>" >&2
  echo "Tip: read WAN IPv4 in Asus router UI and pass it as argument." >&2
  exit 1
fi

if ! is_ipv4 "$WAN_IP"; then
  echo "ERROR: invalid WAN IPv4: ${WAN_IP}" >&2
  exit 1
fi

PUBLIC_IP="$(fetch_public_ip | tr -d '\r\n[:space:]')"
if ! is_ipv4 "$PUBLIC_IP"; then
  echo "ERROR: failed to detect a valid public IPv4, got: ${PUBLIC_IP}" >&2
  exit 1
fi

echo "Router WAN IPv4: ${WAN_IP}"
echo "Detected public IPv4: ${PUBLIC_IP}"

if is_private_or_cgnat "$WAN_IP"; then
  echo "Result: CGNAT/Double-NAT suspected (WAN IP is private/CGNAT range)."
  echo "Decision: use Cloudflare Tunnel fallback path."
  exit 20
fi

if [[ "$WAN_IP" != "$PUBLIC_IP" ]]; then
  echo "Result: WAN IP != detected public IP (likely upstream NAT/CGNAT)."
  echo "Decision: use Cloudflare Tunnel fallback path."
  exit 20
fi

echo "Result: WAN IP matches public IPv4 and is routable."
echo "Decision: continue direct path (Asus DDNS + CNAME + DSM reverse proxy)."
