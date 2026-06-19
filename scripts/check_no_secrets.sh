#!/usr/bin/env bash
# Pre-commit guard: scream loudly if a real-looking API key sneaks into a
# tracked file. Patterns intentionally simple — false positives are tolerable,
# false negatives are not.
set -euo pipefail

PATTERNS=(
    'sk-or-v1-[A-Za-z0-9]{20,}'           # OpenRouter
    'sk-proj-[A-Za-z0-9_-]{30,}'          # OpenAI project
    'sk-ant-[A-Za-z0-9_-]{30,}'           # Anthropic
    'AKIA[0-9A-Z]{16}'                    # AWS access key
    'ghp_[A-Za-z0-9]{36}'                 # GitHub PAT
)

if ! command -v git >/dev/null 2>&1; then
    echo "git not found" >&2
    exit 0
fi

# Scan only files git knows about (skips .env which is gitignored).
files=$(git ls-files 2>/dev/null || true)
if [[ -z "${files}" ]]; then
    exit 0
fi

found=0
for pattern in "${PATTERNS[@]}"; do
    # -P PCRE; allow grep to fail without tripping pipefail
    matches=$(echo "${files}" | xargs grep -lP "${pattern}" 2>/dev/null || true)
    if [[ -n "${matches}" ]]; then
        echo "::error:: Possible secret matching ${pattern}:" >&2
        echo "${matches}" >&2
        found=1
    fi
done

[[ "${found}" -eq 0 ]] || exit 1
echo "No secrets detected in tracked files."
