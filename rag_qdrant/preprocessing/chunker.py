"""Heading-aware markdown chunker for OpenViking ingestion.

OpenViking can parse large files, but splitting a 250-page financial report
(1MB markdown) into logical sections (by headings) before ingestion yields
better semantic chunks and avoids losing context mid-sentence or mid-table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Match markdown headings from H1 to H4
_HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$")


@dataclass
class Chunk:
    level: int
    title: str
    lines: list[str] = field(default_factory=list)

    def content(self) -> str:
        return "\n".join(self.lines).strip()


def chunk_markdown(source: Path, output_dir: Path) -> dict[str, int]:
    """Split a large markdown file into smaller files based on headings.

    Returns a stats dict.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    raw_lines = source.read_text(encoding="utf-8").split("\n")

    chunks: list[Chunk] = []
    current_chunk = Chunk(level=1, title="Root")

    in_code_block = False

    for line in raw_lines:
        # Don't split on headings inside code blocks
        if line.startswith("```"):
            in_code_block = not in_code_block

        match = _HEADING_PATTERN.match(line)
        if match and not in_code_block:
            # Save previous chunk if it has content
            if current_chunk.content():
                chunks.append(current_chunk)

            level = len(match.group(1))
            title = match.group(2).strip()
            # Start new chunk
            current_chunk = Chunk(level=level, title=title)
            current_chunk.lines.append(line)
        else:
            current_chunk.lines.append(line)

    # Add the last chunk
    if current_chunk.content():
        chunks.append(current_chunk)

    # Write chunks to output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean output dir first
    for f in output_dir.glob("*.md"):
        f.unlink()

    files_created = 0
    for i, chunk in enumerate(chunks):
        # Only write chunks that have actual content beyond the heading
        content = chunk.content()
        if not content:
            continue

        # Clean title for filename: replace non-alphanumeric with underscore
        safe_title = re.sub(r"[^\w\s-]", "", chunk.title).strip()
        safe_title = re.sub(r"[-\s]+", "_", safe_title)
        if not safe_title:
            safe_title = "section"

        filename = f"{i:03d}_{safe_title[:50]}.md"
        filepath = output_dir / filename

        filepath.write_text(content + "\n", encoding="utf-8")
        files_created += 1

    return {
        "total_chunks": files_created,
        "original_lines": len(raw_lines)
    }
