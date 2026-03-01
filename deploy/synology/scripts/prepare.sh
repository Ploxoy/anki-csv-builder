#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_TEMPLATE="${ROOT_DIR}/deploy/synology/.env.example"
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

if [[ ! -f "$ENV_TEMPLATE" ]]; then
  echo "ERROR: missing env template: $ENV_TEMPLATE" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ENV_TEMPLATE" "$ENV_FILE"
  echo "Created ${ENV_FILE} from template."
else
  echo "Using existing ${ENV_FILE}."
fi

SYNO_BASE_PATH="$(read_env_var "$ENV_FILE" "SYNO_BASE_PATH")"
if [[ -z "$SYNO_BASE_PATH" ]]; then
  SYNO_BASE_PATH="$(read_env_var "$ENV_TEMPLATE" "SYNO_BASE_PATH")"
fi

if [[ -z "$SYNO_BASE_PATH" ]]; then
  echo "ERROR: SYNO_BASE_PATH is empty in ${ENV_FILE}" >&2
  exit 1
fi

mkdir -p "${SYNO_BASE_PATH}/pgdata" "${SYNO_BASE_PATH}/cache" "${SYNO_BASE_PATH}/logs"

echo "Prepared persistent directories:"
echo "  - ${SYNO_BASE_PATH}/pgdata"
echo "  - ${SYNO_BASE_PATH}/cache"
echo "  - ${SYNO_BASE_PATH}/logs"
echo
echo "Next:"
echo "  1) edit ${ENV_FILE}"
echo "  2) run: bash deploy/synology/scripts/validate_env.sh"
