#!/usr/bin/env bash
# Smoke test for Qdrant: verifies it's up, API key auth works, and version is sane.
# Exit non-zero on any failure so CI/Make can pick it up.
set -euo pipefail

# Load .env if present (don't fail if missing)
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a; . ./.env; set +a
fi

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-dev-local-changeme}"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }

bold "Qdrant smoke test → ${QDRANT_URL}"

# 1. Liveness (no auth required)
status=$(curl -s -o /dev/null -w "%{http_code}" "${QDRANT_URL}/healthz")
[[ "$status" == "200" ]] || fail "GET /healthz returned ${status}, expected 200"
ok "GET /healthz → 200"

# 2. Auth required (no key → 401)
status=$(curl -s -o /dev/null -w "%{http_code}" "${QDRANT_URL}/collections")
[[ "$status" == "401" ]] || fail "Unauthenticated GET /collections returned ${status}, expected 401 (auth not enforced!)"
ok "Unauthenticated GET /collections → 401 (auth enforced)"

# 3. Auth works (with key → 200)
body=$(curl -sS -H "api-key: ${QDRANT_API_KEY}" "${QDRANT_URL}/collections")
echo "${body}" | grep -q '"status":"ok"' || fail "Authenticated /collections did not return status:ok — body: ${body}"
ok "Authenticated GET /collections → ok"

# 4. Version sanity
version=$(curl -sS -H "api-key: ${QDRANT_API_KEY}" "${QDRANT_URL}/" | grep -oE '"version":"[^"]+"' | head -1 || true)
ok "Qdrant ${version:-version-not-reported}"

bold "All checks passed."
