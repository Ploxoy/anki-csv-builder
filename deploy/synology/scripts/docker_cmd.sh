#!/usr/bin/env bash

# Resolve docker binary path on Synology DSM shells where PATH may be minimal.
resolve_docker_bin() {
  if command -v docker >/dev/null 2>&1; then
    command -v docker
    return 0
  fi

  local candidates=(
    "/usr/local/bin/docker"
    "/var/packages/ContainerManager/target/usr/bin/docker"
    "/var/packages/Docker/target/usr/bin/docker"
  )

  local path
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done

  return 1
}

require_docker_bin() {
  local bin=""
  bin="$(resolve_docker_bin || true)"
  if [[ -z "$bin" ]]; then
    echo "ERROR: docker binary not found. Install/enable Synology Container Manager or add docker to PATH." >&2
    return 127
  fi

  if ! "$bin" compose version >/dev/null 2>&1; then
    echo "ERROR: '$bin compose' is not available. Ensure Docker Compose plugin is installed/enabled." >&2
    return 127
  fi

  printf '%s\n' "$bin"
}

