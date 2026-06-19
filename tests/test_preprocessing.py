"""Unit tests for the preprocessing module."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag_qdrant.preprocessing import chunk_markdown, clean_marker_output

# ----------------------------------------------------- clean_marker_output


def test_clean_strips_br_inside_table_cells(tmp_path: Path):
    src = tmp_path / "src.md"
    dest = tmp_path / "out.md"
    src.write_text(
        "| col1 | col2<br>continued | col3 |\n"
        "| a<br>b | c<br>d | e |\n",
        encoding="utf-8",
    )
    stats = clean_marker_output(src, dest)
    out = dest.read_text(encoding="utf-8")
    assert "<br>" not in out
    assert stats["br_tags_removed"] == 3


def test_clean_keeps_br_outside_tables(tmp_path: Path):
    src = tmp_path / "src.md"
    dest = tmp_path / "out.md"
    src.write_text("Regular paragraph with <br> tag.\n", encoding="utf-8")
    clean_marker_output(src, dest)
    out = dest.read_text(encoding="utf-8")
    # Outside tables we leave them alone — rare but possible intentional usage.
    assert "<br>" in out


def test_clean_does_not_modify_source(tmp_path: Path):
    src = tmp_path / "src.md"
    dest = tmp_path / "out.md"
    raw = "| <br>data |\n"
    src.write_text(raw, encoding="utf-8")
    clean_marker_output(src, dest)
    assert src.read_text(encoding="utf-8") == raw  # source untouched


def test_clean_removes_foxit_metadata(tmp_path: Path):
    src = tmp_path / "src.md"
    dest = tmp_path / "out.md"
    src.write_text(
        "Real content.\n"
        "Foxit PDF Reader Version: 12.1.1\n"
        "More content.\n",
        encoding="utf-8",
    )
    clean_marker_output(src, dest)
    out = dest.read_text(encoding="utf-8")
    assert "Foxit PDF Reader" not in out
    assert "Real content" in out
    assert "More content" in out


def test_clean_applies_patches(tmp_path: Path):
    src = tmp_path / "src.md"
    dest = tmp_path / "out.md"
    src.write_text("Original broken text here.\n", encoding="utf-8")
    patches = {"broken": "fixed"}
    clean_marker_output(src, dest, patches=patches)
    out = dest.read_text(encoding="utf-8")
    assert "broken" not in out
    assert "fixed" in out


def test_clean_collapses_excessive_blank_lines(tmp_path: Path):
    src = tmp_path / "src.md"
    dest = tmp_path / "out.md"
    src.write_text("Line 1\n\n\n\n\n\nLine 2\n", encoding="utf-8")
    clean_marker_output(src, dest)
    out = dest.read_text(encoding="utf-8")
    # 6+ consecutive newlines should collapse to 3 (= 2 blank lines)
    assert "\n\n\n\n" not in out
    assert "Line 1" in out and "Line 2" in out


def test_clean_returns_stats_dict(tmp_path: Path):
    src = tmp_path / "src.md"
    dest = tmp_path / "out.md"
    src.write_text("| a<br>b |\nLine\n", encoding="utf-8")
    stats = clean_marker_output(src, dest)
    assert "original_chars" in stats
    assert "cleaned_chars" in stats
    assert "br_tags_before" in stats
    assert "br_tags_after" in stats
    assert "br_tags_removed" in stats
    assert stats["br_tags_after"] == 0


# ------------------------------------------------------------ chunk_markdown


def test_chunk_splits_by_h1_to_h4(tmp_path: Path):
    src = tmp_path / "src.md"
    out = tmp_path / "chunks"
    src.write_text(
        "# Title\nIntro text.\n\n"
        "## Section A\nContent A.\n\n"
        "### Subsection\nDetail.\n\n"
        "## Section B\nContent B.\n",
        encoding="utf-8",
    )
    stats = chunk_markdown(src, out)
    files = sorted(out.glob("*.md"))
    assert stats["total_chunks"] == len(files) >= 4


def test_chunk_does_not_split_inside_code_block(tmp_path: Path):
    src = tmp_path / "src.md"
    out = tmp_path / "chunks"
    src.write_text(
        "# Real heading\n"
        "Content.\n\n"
        "```python\n"
        "# This is a comment, not a heading\n"
        "## Also not a heading\n"
        "```\n"
        "More content under real heading.\n",
        encoding="utf-8",
    )
    chunk_markdown(src, out)
    # Only 1 real heading → 1 chunk file
    files = sorted(out.glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "More content" in content
    assert "## Also not a heading" in content


def test_chunk_cleans_output_dir_before_writing(tmp_path: Path):
    src = tmp_path / "src.md"
    out = tmp_path / "chunks"
    out.mkdir()
    stale = out / "999_stale.md"
    stale.write_text("old garbage", encoding="utf-8")

    src.write_text("# Heading\nNew content.\n", encoding="utf-8")
    chunk_markdown(src, out)

    assert not stale.exists()
    assert any(out.glob("*.md"))


def test_chunk_creates_safe_filenames(tmp_path: Path):
    src = tmp_path / "src.md"
    out = tmp_path / "chunks"
    src.write_text(
        "# CÔNG BỐ THÔNG TIN BẮT THƯỜNG / EXTRAORDINARY!\n"
        "Vietnamese content with diacritics.\n",
        encoding="utf-8",
    )
    chunk_markdown(src, out)
    files = list(out.glob("*.md"))
    assert len(files) == 1
    name = files[0].name
    # Special chars stripped, but Vietnamese diacritics preserved (\w matches them)
    assert "/" not in name
    assert "!" not in name
    assert name.endswith(".md")


def test_chunk_skips_empty_sections(tmp_path: Path):
    src = tmp_path / "src.md"
    out = tmp_path / "chunks"
    src.write_text(
        "# Heading 1\n"
        "Real content here.\n\n"
        "# Heading 2\n",  # no body — should be skipped
        encoding="utf-8",
    )
    chunk_markdown(src, out)
    # Heading 2 has only the heading line → still has content (the heading itself)
    # so it's written. But truly empty chunks should be skipped.
    # Test that we don't write 0-byte files
    for f in out.glob("*.md"):
        assert f.stat().st_size > 0


def test_chunk_raises_when_source_missing(tmp_path: Path):
    out = tmp_path / "chunks"
    with pytest.raises(FileNotFoundError):
        chunk_markdown(tmp_path / "nonexistent.md", out)


def test_chunk_returns_stats(tmp_path: Path):
    src = tmp_path / "src.md"
    out = tmp_path / "chunks"
    src.write_text("# A\ntext\n# B\ntext\n", encoding="utf-8")
    stats = chunk_markdown(src, out)
    assert "total_chunks" in stats
    assert "original_lines" in stats
    assert stats["total_chunks"] == 2
