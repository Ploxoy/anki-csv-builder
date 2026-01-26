#!/usr/bin/env bash
set -euo pipefail

# Local dev convenience: uvicorn won't auto-load `.env` unless you run via docker-compose.
# Sourcing is safe here because we never print secrets; it only populates the process env.
if [[ "${LOAD_ENV_FILE:-1}" != "0" && -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
RELOAD="${RELOAD:-}"

ARGS=(api.main:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL")
if [[ "${RELOAD,,}" == "1" || "${RELOAD,,}" == "true" || "${RELOAD,,}" == "yes" ]]; then
  ARGS+=(--reload)
fi

exec uvicorn "${ARGS[@]}"
