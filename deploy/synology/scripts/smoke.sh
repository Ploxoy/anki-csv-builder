#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

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

API_PORT="$(read_env_var "$ENV_FILE" "API_PORT")"
WEB_PORT="$(read_env_var "$ENV_FILE" "WEB_PORT")"
DEFAULT_HOST="$(read_env_var "$ENV_FILE" "NAS_IP")"
HOST="${1:-${DEFAULT_HOST:-127.0.0.1}}"

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"

API_URL="http://${HOST}:${API_PORT}/health"
WEB_URL="http://${HOST}:${WEB_PORT}/"

fetch() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 10 "$url"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q -T 10 -O - "$url"
    return
  fi
  echo "ERROR: neither curl nor wget is installed" >&2
  exit 1
}

echo "Checking API health: ${API_URL}"
api_body="$(fetch "$API_URL")"
if ! printf '%s' "$api_body" | grep -Eiq 'ok|healthy'; then
  echo "ERROR: unexpected API health response: ${api_body}" >&2
  exit 1
fi
echo "API health looks good."

echo "Checking web UI: ${WEB_URL}"
_web_body="$(fetch "$WEB_URL")"
echo "Web endpoint is reachable."

echo "Smoke check passed."
