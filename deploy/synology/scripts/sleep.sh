#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/deploy/synology/docker-compose.synology.yml"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"
DOCKER_HELPER="${ROOT_DIR}/deploy/synology/scripts/docker_cmd.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

if [[ ! -f "$DOCKER_HELPER" ]]; then
  echo "ERROR: docker helper not found: ${DOCKER_HELPER}" >&2
  exit 1
fi

# shellcheck source=deploy/synology/scripts/docker_cmd.sh
source "$DOCKER_HELPER"
DOCKER_BIN="$(require_docker_bin)"

MODE="${1:-warm}"
case "$MODE" in
  warm)
    echo "Sleep mode: warm (stop api + web, keep db + waker running)"
    "$DOCKER_BIN" compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" stop api web
    ;;
  deep)
    echo "Sleep mode: deep (stop db + api + web, keep waker for auto-wake)"
    "$DOCKER_BIN" compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" stop db api web
    ;;
  *)
    echo "Usage: bash deploy/synology/scripts/sleep.sh [warm|deep]" >&2
    exit 1
    ;;
esac

echo "Current service state:"
"$DOCKER_BIN" compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
