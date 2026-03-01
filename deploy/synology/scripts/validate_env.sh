#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"
COMPOSE_FILE="${ROOT_DIR}/deploy/synology/docker-compose.synology.yml"

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

require_present() {
  local key="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "ERROR: ${key} is empty in ${ENV_FILE}" >&2
    exit 1
  fi
}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  echo "Run: bash deploy/synology/scripts/prepare.sh" >&2
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "ERROR: compose file not found: ${COMPOSE_FILE}" >&2
  exit 1
fi

SYNO_BASE_PATH="$(read_env_var "$ENV_FILE" "SYNO_BASE_PATH")"
API_PORT="$(read_env_var "$ENV_FILE" "API_PORT")"
WEB_PORT="$(read_env_var "$ENV_FILE" "WEB_PORT")"
POSTGRES_PASSWORD="$(read_env_var "$ENV_FILE" "POSTGRES_PASSWORD")"
API_SHARED_SECRET="$(read_env_var "$ENV_FILE" "API_SHARED_SECRET")"
OPENAI_API_KEY="$(read_env_var "$ENV_FILE" "OPENAI_API_KEY")"
ELEVENLABS_API_KEY="$(read_env_var "$ENV_FILE" "ELEVENLABS_API_KEY")"

require_present "SYNO_BASE_PATH" "$SYNO_BASE_PATH"
require_present "API_PORT" "$API_PORT"
require_present "WEB_PORT" "$WEB_PORT"
require_present "POSTGRES_PASSWORD" "$POSTGRES_PASSWORD"
require_present "API_SHARED_SECRET" "$API_SHARED_SECRET"
require_present "OPENAI_API_KEY" "$OPENAI_API_KEY"

if [[ "$POSTGRES_PASSWORD" == *"change-this"* ]]; then
  echo "ERROR: POSTGRES_PASSWORD still uses placeholder value." >&2
  exit 1
fi

if [[ "$API_SHARED_SECRET" == *"change-this"* ]]; then
  echo "ERROR: API_SHARED_SECRET still uses placeholder value." >&2
  exit 1
fi

if [[ "$OPENAI_API_KEY" == "sk-..." ]]; then
  echo "ERROR: OPENAI_API_KEY still uses placeholder value." >&2
  exit 1
fi

if [[ "${#POSTGRES_PASSWORD}" -lt 12 ]]; then
  echo "WARN: POSTGRES_PASSWORD is shorter than 12 chars."
fi

if [[ "${#API_SHARED_SECRET}" -lt 20 ]]; then
  echo "WARN: API_SHARED_SECRET is shorter than 20 chars."
fi

if [[ -z "$ELEVENLABS_API_KEY" ]]; then
  echo "INFO: ELEVENLABS_API_KEY is empty (allowed)."
fi

if [[ ! -d "${SYNO_BASE_PATH}/pgdata" || ! -d "${SYNO_BASE_PATH}/cache" || ! -d "${SYNO_BASE_PATH}/logs" ]]; then
  echo "ERROR: persistent directories are missing under ${SYNO_BASE_PATH}" >&2
  echo "Run: bash deploy/synology/scripts/prepare.sh" >&2
  exit 1
fi

echo "Environment validation passed."
echo "Compose file: ${COMPOSE_FILE}"
echo "Env file: ${ENV_FILE}"
