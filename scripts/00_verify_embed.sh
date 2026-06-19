#!/usr/bin/env bash
# Verify that the configured embedding endpoint works end-to-end.
# Exits non-zero on any failure so Make / CI can pick it up.
set -euo pipefail

if [[ -f .env ]]; then
    set -a; . ./.env; set +a
fi

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }

EMBED_BASE_URL="${EMBED_BASE_URL:?EMBED_BASE_URL must be set in .env}"
EMBED_MODEL="${EMBED_MODEL:?EMBED_MODEL must be set in .env}"
EMBED_DIM="${EMBED_DIM:?EMBED_DIM must be set in .env}"

bold "Embedding smoke test → ${EMBED_BASE_URL}  (model=${EMBED_MODEL}, expected_dim=${EMBED_DIM})"

# 1) Endpoint reachable + auth accepted
auth_header=()
if [[ -n "${EMBED_API_KEY:-}" ]]; then
    auth_header=(-H "Authorization: Bearer ${EMBED_API_KEY}")
fi

response=$(curl -sS -w "\n__HTTP__%{http_code}" -X POST \
    "${EMBED_BASE_URL%/}/embeddings" \
    "${auth_header[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${EMBED_MODEL}\",\"input\":\"smoke test from VibeAssignment\"}")

http_code="${response##*__HTTP__}"
body="${response%__HTTP__*}"

[[ "${http_code}" == "200" ]] || fail "POST /embeddings returned HTTP ${http_code}: ${body}"
ok "POST /embeddings → 200"

# 2) Response shape sanity
dim=$(echo "${body}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('data', [])
if not items:
    sys.stderr.write('No data[] array in response\n'); sys.exit(2)
emb = items[0].get('embedding')
if not isinstance(emb, list) or not emb:
    sys.stderr.write('Bad embedding payload\n'); sys.exit(3)
print(len(emb))
")
[[ "${dim}" == "${EMBED_DIM}" ]] || fail "Returned dim=${dim}, expected ${EMBED_DIM}"
ok "Embedding dim ${dim} matches EMBED_DIM"

# 3) Multilingual sanity: Vietnamese passage gets a non-zero vector
response=$(curl -sS -X POST "${EMBED_BASE_URL%/}/embeddings" \
    "${auth_header[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${EMBED_MODEL}\",\"input\":\"Tổng tài sản MSB cuối năm 2024 đạt 320 nghìn tỷ đồng.\"}")
nonzero=$(echo "${response}" | python3 -c "
import json, sys
emb = json.load(sys.stdin)['data'][0]['embedding']
print(sum(1 for x in emb if x != 0))
")
(( nonzero > 100 )) || fail "Vietnamese embedding seems degenerate (only ${nonzero} non-zero dims)"
ok "Vietnamese embedding has ${nonzero}/${EMBED_DIM} non-zero dimensions"

bold "Embedding endpoint OK."
