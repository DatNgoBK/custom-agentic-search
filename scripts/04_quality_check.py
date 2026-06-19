#!/usr/bin/env python
"""Quality validation script for preprocessed markdown chunks."""

import re
import sys
from pathlib import Path

CHUNKS_DIR = Path("output/source/chunks")
VIETNAMESE_CHARS = re.compile(r"[àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳýỹỷỵ]")
BR_TAG = re.compile(r"<br>", re.IGNORECASE)

def main() -> int:
    if not CHUNKS_DIR.exists():
        print(f"❌ Chunks directory not found: {CHUNKS_DIR}")
        return 1

    chunks = list(CHUNKS_DIR.glob("*.md"))
    if not chunks:
        print(f"❌ No markdown chunks found in {CHUNKS_DIR}")
        return 1

    print(f"🔍 Validating {len(chunks)} chunks...\n")

    total_issues = 0
    total_chars = 0

    for chunk_file in sorted(chunks):
        content = chunk_file.read_text(encoding="utf-8")
        total_chars += len(content)
        issues = []

        # Check size
        if len(content) < 50:
            issues.append("Too short (< 50 chars)")

        # Check Vietnamese diacritics — only flag for substantial chunks
        # (short chunks or pure numeric tables don't need diacritics).
        if not VIETNAMESE_CHARS.search(content) and len(content) > 200:
            issues.append("Missing Vietnamese diacritics in long chunk")

        # Check for uncleaned <br> inside tables
        table_lines = [line for line in content.split("\n") if line.lstrip().startswith("|")]
        br_in_tables = sum(len(BR_TAG.findall(line)) for line in table_lines)
        if br_in_tables > 0:
            issues.append(f"Found {br_in_tables} <br> tags inside tables")

        # Check for known signature noise
        if "DN: C=VN" in content or "Foxit PDF Reader" in content:
            issues.append("Digital signature noise detected")

        if issues:
            total_issues += 1
            print(f"⚠️ {chunk_file.name}: {', '.join(issues)}")

    print("-" * 50)
    print(f"📊 Summary: {len(chunks)} chunks, {total_chars} total characters.")
    if total_issues == 0:
        print("✅ All chunks passed quality validation.")
        return 0
    print(f"❌ {total_issues}/{len(chunks)} chunks have quality issues.")
    # We don't fail the build for warnings, only for catastrophic failures
    # like no chunks or missing directory, so we exit 0 here to allow make demo to continue,
    # but the warnings will be visible.
    return 0

if __name__ == "__main__":
    sys.exit(main())
