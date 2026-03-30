#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/deploy/synology/docker-compose.synology.yml"
ENV_FILE="${ROOT_DIR}/deploy/synology/.env"
ENV_TEMPLATE="${ROOT_DIR}/deploy/synology/.env.example"
PREPARE_SCRIPT="${ROOT_DIR}/deploy/synology/scripts/prepare.sh"
VALIDATE_SCRIPT="${ROOT_DIR}/deploy/synology/scripts/validate_env.sh"
SMOKE_SCRIPT="${ROOT_DIR}/deploy/synology/scripts/smoke.sh"
DOCKER_HELPER="${ROOT_DIR}/deploy/synology/scripts/docker_cmd.sh"

NO_BUILD=0
SKIP_SMOKE=0
FORCE_ENV=0
SMOKE_HOST=""

usage() {
  cat <<EOF
Usage: bash deploy/synology/scripts/install.sh [options]

Options:
  --no-build           Run docker compose up without --build
  --skip-smoke         Do not run smoke check after deploy
  --smoke-host <host>  Host/IP for smoke check (optional)
  --force-env          Recreate .env from .env.example (backup existing .env)
  -h, --help           Show help

Default behavior:
  - preserves existing deploy/synology/.env
  - prepares folders + validates env
  - runs docker compose up -d --build
  - runs smoke check
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      NO_BUILD=1
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    --smoke-host)
      SMOKE_HOST="${2:-}"
      if [[ -z "$SMOKE_HOST" ]]; then
        echo "ERROR: --smoke-host requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --force-env)
      FORCE_ENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$FORCE_ENV" -eq 1 ]]; then
  if [[ ! -f "$ENV_TEMPLATE" ]]; then
    echo "ERROR: template not found: ${ENV_TEMPLATE}" >&2
    exit 1
  fi
  if [[ -f "$ENV_FILE" ]]; then
    backup="${ENV_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$ENV_FILE" "$backup"
    echo "Backed up existing .env -> ${backup}"
  fi
  cp "$ENV_TEMPLATE" "$ENV_FILE"
  echo "Recreated ${ENV_FILE} from template."
fi

echo "Preparing folders and env file..."
bash "$PREPARE_SCRIPT"

echo "Validating environment..."
bash "$VALIDATE_SCRIPT"

if [[ ! -f "$DOCKER_HELPER" ]]; then
  echo "ERROR: docker helper not found: ${DOCKER_HELPER}" >&2
  exit 1
fi

# shellcheck source=deploy/synology/scripts/docker_cmd.sh
source "$DOCKER_HELPER"
DOCKER_BIN="$(require_docker_bin)"

compose_cmd=("$DOCKER_BIN" compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d)
if [[ "$NO_BUILD" -eq 0 ]]; then
  compose_cmd+=(--build)
fi

echo "Starting services..."
"${compose_cmd[@]}"

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  echo "Running smoke check..."
  if [[ -n "$SMOKE_HOST" ]]; then
    bash "$SMOKE_SCRIPT" "$SMOKE_HOST"
  else
    bash "$SMOKE_SCRIPT"
  fi
else
  echo "Smoke check skipped by flag."
fi

echo "Install completed."
