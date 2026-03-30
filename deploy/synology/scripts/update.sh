#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/deploy/synology/docker-compose.synology.yml"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"
SMOKE_SCRIPT="${ROOT_DIR}/deploy/synology/scripts/smoke.sh"
DOCKER_HELPER="${ROOT_DIR}/deploy/synology/scripts/docker_cmd.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  echo "Run: bash deploy/synology/scripts/prepare.sh" >&2
  exit 1
fi

if [[ ! -d "${ROOT_DIR}/.git" ]]; then
  echo "ERROR: ${ROOT_DIR} is not a git checkout." >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -f "$DOCKER_HELPER" ]]; then
  echo "ERROR: docker helper not found: ${DOCKER_HELPER}" >&2
  exit 1
fi

# shellcheck source=deploy/synology/scripts/docker_cmd.sh
source "$DOCKER_HELPER"
DOCKER_BIN="$(require_docker_bin)"

echo "Pulling latest changes..."
git pull --ff-only

echo "Rebuilding and starting services..."
"$DOCKER_BIN" compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build

echo "Running post-update smoke check..."
bash "$SMOKE_SCRIPT"

echo "Update completed."
