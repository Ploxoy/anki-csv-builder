#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash deploy/synology/scripts/check_public_endpoints.sh <public_domain>" >&2
  exit 1
fi

DOMAIN="$1"

fetch_status() {
  local url="$1"
  curl -k -sS -o /dev/null --connect-timeout 8 --max-time 15 -w "%{http_code}" "$url"
}

expect_success() {
  local url="$1"
  local code
  code="$(fetch_status "$url")"
  if [[ "$code" =~ ^(2|3)[0-9]{2}$ ]]; then
    echo "OK: ${url} -> ${code}"
    return 0
  fi
  echo "FAIL: expected success for ${url}, got ${code}" >&2
  return 1
}

expect_blocked() {
  local url="$1"
  local code
  code="$(fetch_status "$url" || true)"
  if [[ "$code" == "000" ]]; then
    echo "OK: ${url} is not publicly reachable (connection blocked)."
    return 0
  fi
  if [[ "$code" == "403" || "$code" == "444" || "$code" == "495" || "$code" == "496" || "$code" == "497" ]]; then
    echo "OK: ${url} is blocked by policy (${code})."
    return 0
  fi
  echo "FAIL: expected ${url} to be blocked from internet, got HTTP ${code}" >&2
  return 1
}

echo "Checking public endpoints for domain: ${DOMAIN}"
echo "Run this from an external network (LTE/5G or off-site host)."

expect_success "https://${DOMAIN}/"
expect_success "https://${DOMAIN}/api/health"
expect_blocked "http://${DOMAIN}:8000/health"
expect_blocked "http://${DOMAIN}:5173/"

echo "Public exposure checks passed."
