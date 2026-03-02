#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/deploy/synology/docker-compose.synology.yml"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"
SMOKE_SCRIPT="${ROOT_DIR}/deploy/synology/scripts/smoke.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

echo "Starting db + api + web..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d db api web

echo "Running smoke check..."
bash "$SMOKE_SCRIPT"

echo "Wake completed."
