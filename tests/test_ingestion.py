"""Tests for the ingestion pipeline components."""

from __future__ import annotations

import pytest
from models.schemas import PaperMetadata
from ingestion.text_cleaner import clean_text
from ingestion.chunker import chunk_text, ChunkingConfig


class TestTextCleaner:
    def test_collapse_whitespace(self):
        result = clean_text("hello   world\n\n\n\nfoo")
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_fix_hyphenation(self):
        result = clean_text("trans-\nformer")
        assert "transformer" in result

    def test_removes_page_numbers(self):
        result = clean_text("Some text.\n  42  \nMore text.")
        assert "42" not in result

    def test_strip_citation_numbers(self):
        result = clean_text("as shown in previous work [12, 3]")
        assert "[12, 3]" not in result

    def test_preserves_unicode(self):
        result = clean_text("Müller et al. studied résumé data")
        assert "Müller" in result
        assert "résumé" in result


class TestChunker:
    def test_produces_chunks(self, sample_metadata, sample_text):
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10, min_chunk_size=5, use_semantic_chunking=False)
        chunks = chunk_text(sample_text, "Methodology", sample_metadata, config)
        assert len(chunks) > 0
        assert all(c.paper_title == sample_metadata.title for c in chunks)
        assert all(c.section == "Methodology" for c in chunks)

    def test_chunk_ids_unique(self, sample_metadata, sample_text):
        config = ChunkingConfig(chunk_size=30, chunk_overlap=5, min_chunk_size=5, use_semantic_chunking=False)
        chunks = chunk_text(sample_text, "Test", sample_metadata, config)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_small_text_single_chunk(self, sample_metadata):
        config = ChunkingConfig(chunk_size=500, chunk_overlap=10, min_chunk_size=5, use_semantic_chunking=False)
        chunks = chunk_text("Short text for testing purposes here.", "Intro", sample_metadata, config)
        assert len(chunks) <= 1

    def test_metadata_preserved(self, sample_metadata, sample_text):
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10, min_chunk_size=5, use_semantic_chunking=False)
        chunks = chunk_text(sample_text, "Methods", sample_metadata, config)
        for c in chunks:
            assert c.year == 2017
            assert "Vaswani" in c.authors
