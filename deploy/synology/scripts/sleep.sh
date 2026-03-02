#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/deploy/synology/docker-compose.synology.yml"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

MODE="${1:-warm}"
case "$MODE" in
  warm)
    echo "Sleep mode: warm (stop api + web, keep db running)"
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" stop api web
    ;;
  deep)
    echo "Sleep mode: deep (stop db + api + web)"
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" stop db api web
    ;;
  *)
    echo "Usage: bash deploy/synology/scripts/sleep.sh [warm|deep]" >&2
    exit 1
    ;;
esac

echo "Current service state:"
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
