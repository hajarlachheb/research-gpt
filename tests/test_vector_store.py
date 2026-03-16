"""Tests for the ChromaDB vector store."""

from __future__ import annotations

import os
import shutil
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_vs")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from models.schemas import DocumentChunk
from retrieval.vector_store import VectorStore, _chunk_to_metadata


@pytest.fixture(autouse=True)
def _clean_chroma():
    path = "./data/test_chroma_vs"
    if os.path.exists(path):
        shutil.rmtree(path)
    yield
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def vector_store():
    return VectorStore()


@pytest.fixture
def chunks():
    return [
        DocumentChunk(
            chunk_id="vs_001",
            text="Transformers rely on self-attention mechanisms for sequence modeling.",
            paper_title="Attention Is All You Need",
            authors=["Vaswani"],
            year=2017,
            section="Introduction",
            source_url="https://arxiv.org/abs/1706.03762",
            conference="NeurIPS",
        ),
        DocumentChunk(
            chunk_id="vs_002",
            text="BERT is a bidirectional transformer pre-trained on masked language modeling.",
            paper_title="BERT",
            authors=["Devlin"],
            year=2018,
            section="Abstract",
        ),
    ]


class TestVectorStore:
    def test_add_and_count(self, vector_store, chunks):
        embeddings = [[0.1] * 384, [0.2] * 384]
        vector_store.add_chunks(chunks, embeddings)
        assert vector_store.count == 2

    def test_query_returns_results(self, vector_store, chunks):
        embeddings = [[0.1] * 384, [0.2] * 384]
        vector_store.add_chunks(chunks, embeddings)
        results = vector_store.query([0.1] * 384, top_k=5)
        assert len(results) == 2
        assert results[0].chunk.chunk_id in ("vs_001", "vs_002")

    def test_query_with_filter(self, vector_store, chunks):
        embeddings = [[0.1] * 384, [0.2] * 384]
        vector_store.add_chunks(chunks, embeddings)
        results = vector_store.query(
            [0.1] * 384, top_k=5, where={"year": {"$eq": 2017}},
        )
        assert all(r.chunk.year == 2017 for r in results)

    def test_get_all_papers(self, vector_store, chunks):
        embeddings = [[0.1] * 384, [0.2] * 384]
        vector_store.add_chunks(chunks, embeddings)
        papers = vector_store.get_all_papers()
        titles = {p["title"] for p in papers}
        assert "Attention Is All You Need" in titles
        assert "BERT" in titles

    def test_get_all_chunks(self, vector_store, chunks):
        embeddings = [[0.1] * 384, [0.2] * 384]
        vector_store.add_chunks(chunks, embeddings)
        all_chunks = vector_store.get_all_chunks()
        assert len(all_chunks) == 2
        ids = {c.chunk_id for c in all_chunks}
        assert ids == {"vs_001", "vs_002"}

    def test_empty_store(self, vector_store):
        assert vector_store.count == 0
        results = vector_store.query([0.0] * 384, top_k=5)
        assert results == []

    def test_metadata_roundtrip(self, chunks):
        meta = _chunk_to_metadata(chunks[0])
        assert meta["paper_title"] == "Attention Is All You Need"
        assert meta["authors"] == "Vaswani"
        assert meta["year"] == 2017
        assert meta["conference"] == "NeurIPS"
