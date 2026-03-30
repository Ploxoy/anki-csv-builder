#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_TEMPLATE="${ROOT_DIR}/deploy/synology/.env.example"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"
DEFAULT_SYNO_BASE_PATH="/volume1/docker/anki-csv-builder"

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

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$ENV_TEMPLATE" ]]; then
    cp "$ENV_TEMPLATE" "$ENV_FILE"
    echo "Created ${ENV_FILE} from template."
  else
    cat >"$ENV_FILE" <<EOF
# Base path on Synology shared folder (create it first)
SYNO_BASE_PATH=${DEFAULT_SYNO_BASE_PATH}

# Optional helper value for smoke checks (can stay empty)
# NAS_IP=192.168.1.20

# Public entrypoint (Stage 2, internet access)
PUBLIC_DOMAIN=app.doedutch.nl

# ASUS DDNS hostname used as CNAME target at registrar DNS
# Example: doedutch.asuscomm.com
ASUS_DDNS_HOST=

# Service ports exposed from NAS
API_PORT=8000
# WEB_PORT is handled by waker
WEB_PORT=5173

# Postgres
POSTGRES_DB=ankicards
POSTGRES_USER=ankicards
POSTGRES_PASSWORD=change-this-db-password

# API auth
API_REQUIRE_SHARED_SECRET=1
API_SHARED_SECRET=change-this-admin-secret

# Provider keys (server-side only)
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=

# Auto sleep/wake (waker)
WAKER_IDLE_STOP=1
WAKER_IDLE_MINUTES=60
WAKER_IDLE_CHECK_SECONDS=30
WAKER_START_TIMEOUT_SECONDS=120
WAKER_TOUCH_FLUSH_SECONDS=15
WAKER_PROXY_TIMEOUT_SECONDS=600
WAKER_LOG_LEVEL=info
EOF
    echo "Template missing; created ${ENV_FILE} from built-in defaults."
  fi
else
  echo "Using existing ${ENV_FILE}."
fi

SYNO_BASE_PATH="$(read_env_var "$ENV_FILE" "SYNO_BASE_PATH")"
if [[ -z "$SYNO_BASE_PATH" ]]; then
  if [[ -f "$ENV_TEMPLATE" ]]; then
    SYNO_BASE_PATH="$(read_env_var "$ENV_TEMPLATE" "SYNO_BASE_PATH")"
  fi
fi

if [[ -z "$SYNO_BASE_PATH" ]]; then
  SYNO_BASE_PATH="$DEFAULT_SYNO_BASE_PATH"
  echo "WARN: SYNO_BASE_PATH is empty, using default ${DEFAULT_SYNO_BASE_PATH}"
fi

mkdir -p "${SYNO_BASE_PATH}/pgdata" "${SYNO_BASE_PATH}/cache" "${SYNO_BASE_PATH}/logs"
mkdir -p "${SYNO_BASE_PATH}/waker"

echo "Prepared persistent directories:"
echo "  - ${SYNO_BASE_PATH}/pgdata"
echo "  - ${SYNO_BASE_PATH}/cache"
echo "  - ${SYNO_BASE_PATH}/logs"
echo "  - ${SYNO_BASE_PATH}/waker"
echo
echo "Next:"
echo "  1) edit ${ENV_FILE}"
echo "  2) run: bash deploy/synology/scripts/validate_env.sh"
