#!/usr/bin/env bash
# Run Marker on the PDF in input/ and write markdown to output/.
#
# This is intended to be run ONCE on the developer's machine. The resulting
# markdown is committed to the repo so graders can skip Marker entirely
# (which downloads ~3GB of models and takes 10-15min on CPU).
#
# To use the GPU on Apple Silicon set TORCH_DEVICE=mps. Marker's surya-ocr
# now respects MPS; it falls back to CPU silently if something is unsupported.
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT_PDF="${INPUT_PDF:-input/source.pdf}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

if [[ ! -f "${INPUT_PDF}" ]]; then
    echo "::error:: PDF not found at ${INPUT_PDF}" >&2
    echo "Place your source PDF at input/source.pdf or set INPUT_PDF=path/to/file.pdf" >&2
    exit 1
fi

if ! command -v marker_single >/dev/null 2>&1; then
    echo "::error:: 'marker_single' CLI not found. Activate the project venv:" >&2
    echo "  source .venv/bin/activate" >&2
    echo "  pip install marker-pdf" >&2
    exit 1
fi

echo "→ Marker: ${INPUT_PDF}  →  ${OUTPUT_DIR}/"
echo "  device: ${TORCH_DEVICE:-auto}  (set TORCH_DEVICE=mps to force MPS on Apple Silicon)"

mkdir -p "${OUTPUT_DIR}"

start=$(date +%s)
TORCH_DEVICE="${TORCH_DEVICE:-mps}" marker_single \
    "${INPUT_PDF}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_format markdown \
    --disable_image_extraction
duration=$(( $(date +%s) - start ))

# marker_single creates output/<basename>/<basename>.md
basename=$(basename "${INPUT_PDF}" .pdf)
md_file="${OUTPUT_DIR}/${basename}/${basename}.md"

if [[ ! -f "${md_file}" ]]; then
    echo "::error:: Expected markdown not found at ${md_file}" >&2
    exit 1
fi

# Validation
chars=$(wc -c < "${md_file}" | tr -d ' ')
lines=$(wc -l < "${md_file}" | tr -d ' ')
tables=$(grep -c '|' "${md_file}" 2>/dev/null || echo 0)
headings=$(grep -cE '^#{1,6} ' "${md_file}" 2>/dev/null || echo 0)

echo
echo "Marker output:"
echo "  file:     ${md_file}"
echo "  duration: ${duration}s"
echo "  chars:    ${chars}"
echo "  lines:    ${lines}"
echo "  headings: ${headings}"
echo "  table-pipe lines: ${tables}"
echo
echo "Done."
