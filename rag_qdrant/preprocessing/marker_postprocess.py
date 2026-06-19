"""Post-process Marker PDF-to-markdown output for Vietnamese financial documents.

Marker's OCR pipeline produces artefacts that degrade downstream RAG quality:

1. ``<br>`` tags inserted mid-word inside table cells (PDF column wraps).
2. Digital-signature / certificate noise blocks (DN:, OID., Foxit metadata).
3. Excessive blank lines and trailing whitespace.

This module cleans those artefacts while **preserving the original file**.
The cleaned output is written to a separate path.
"""
from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Signature / certificate noise patterns
# ---------------------------------------------------------------------------
# Generic patterns to catch common Vietnamese digital signature blocks
# without hardcoding specific company names.
_SIGNATURE_PATTERNS: list[re.Pattern[str]] = [
    # Digital signature block starting with common identifiers (DN: C=VN, OID, etc.)
    # and ending with typical signature metadata (Reason, Location, Date, Foxit)
    re.compile(
        r"(?:DN:\s*C=VN|Tài liệu đính kèm/ Attachment:|Đại diện tổ chức Organization representative).*?"
        r"(?:Foxit PDF Reader Version.*?\n|document\n|Date:.*?\n\d{2}:\d{2}:\d{2}.*?\n)",
        re.DOTALL | re.IGNORECASE,
    ),
    # Standalone Foxit metadata
    re.compile(r"^Foxit PDF Reader Version:.*$", re.MULTILINE),
    # Common signature fragments that might be left behind
    re.compile(r"^(?:Reason|Lý do):\s*I am the author of this document\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(?:Location|Địa điểm):\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(?:Date|Ngày):\s*\d{4}[\./-]\d{2}[\./-]\d{2}\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\d{2}:\d{2}:\d{2}\s+(?:NAM|VN|VIETNAM)\s*$", re.MULTILINE | re.IGNORECASE),
]


def _remove_signature_noise(text: str) -> str:
    """Strip generic digital-signature and PDF-reader metadata blocks."""
    for pattern in _SIGNATURE_PATTERNS:
        text = pattern.sub("", text)
    return text


# ---------------------------------------------------------------------------
# <br> tag fixing inside markdown table cells
# ---------------------------------------------------------------------------
_BR_TAG = re.compile(r"<br>", re.IGNORECASE)

def _fix_br_in_table_cells(text: str) -> str:
    """Replace ``<br>`` with space inside markdown table cells.

    Inside a table row (line starting with ``|``), ``<br>`` almost always
    represents a column-wrap artefact, not intentional line breaks.
    Outside tables we leave them alone (rare, but possible).
    """
    lines: list[str] = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            line = _BR_TAG.sub(" ", line)
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------
def _normalize_whitespace(text: str) -> str:
    """Collapse 3+ consecutive blank lines to 2; strip trailing spaces."""
    # Strip trailing whitespace per line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _apply_patches(text: str, patches: dict[str, str]) -> str:
    """Apply targeted OCR text replacement patches."""
    if not patches:
        return text
    for broken, fixed in patches.items():
        text = text.replace(broken, fixed)
    return text


def clean_marker_output(source: Path, dest: Path, patches: dict[str, str] | None = None) -> dict[str, int]:
    """Read *source* markdown, clean it, write to *dest*.

    Returns a stats dict with counts of changes made.

    The original file at *source* is **never modified**.
    """
    raw = source.read_text(encoding="utf-8")
    stats: dict[str, int] = {
        "original_chars": len(raw),
        "original_lines": raw.count("\n"),
        "br_tags_before": len(_BR_TAG.findall(raw)),
    }

    text = raw
    text = _remove_signature_noise(text)
    text = _fix_br_in_table_cells(text)
    text = _apply_patches(text, patches or {})
    text = _normalize_whitespace(text)

    stats["cleaned_chars"] = len(text)
    stats["cleaned_lines"] = text.count("\n")
    stats["br_tags_after"] = len(_BR_TAG.findall(text))
    stats["br_tags_removed"] = stats["br_tags_before"] - stats["br_tags_after"]

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return stats
