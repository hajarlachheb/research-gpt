"""Tests for the hybrid retriever and RRF fusion."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_hr")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from models.schemas import DocumentChunk, RetrievedChunk
from retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(chunk_id="h1", text="attention mechanism", paper_title="Paper A", year=2017),
        DocumentChunk(chunk_id="h2", text="transformer architecture", paper_title="Paper B", year=2018),
        DocumentChunk(chunk_id="h3", text="bert pretraining", paper_title="Paper C", year=2019),
    ]


@pytest.fixture
def hybrid_retriever(sample_chunks):
    vector_store = MagicMock()
    bm25 = MagicMock()

    vector_results = [
        RetrievedChunk(chunk=sample_chunks[0], score=0.9),
        RetrievedChunk(chunk=sample_chunks[1], score=0.7),
    ]
    bm25_results = [
        RetrievedChunk(chunk=sample_chunks[1], score=5.0),
        RetrievedChunk(chunk=sample_chunks[2], score=3.0),
    ]

    vector_store.query.return_value = vector_results
    bm25.query.return_value = bm25_results

    return HybridRetriever(vector_store, bm25), vector_store, bm25


class TestHybridRetriever:
    @patch("retrieval.hybrid_retriever.embed_query", return_value=[0.1] * 384)
    def test_retrieve_merges_results(self, mock_embed, hybrid_retriever):
        retriever, _, _ = hybrid_retriever
        results = retriever.retrieve("test query", top_k=10)
        assert len(results) == 3
        chunk_ids = [r.chunk.chunk_id for r in results]
        assert "h1" in chunk_ids
        assert "h2" in chunk_ids
        assert "h3" in chunk_ids

    @patch("retrieval.hybrid_retriever.embed_query", return_value=[0.1] * 384)
    def test_rrf_boosts_shared_chunks(self, mock_embed, hybrid_retriever):
        retriever, _, _ = hybrid_retriever
        results = retriever.retrieve("test query", top_k=10)
        scores = {r.chunk.chunk_id: r.score for r in results}
        assert scores["h2"] > scores["h1"]
        assert scores["h2"] > scores["h3"]

    @patch("retrieval.hybrid_retriever.embed_query", return_value=[0.1] * 384)
    def test_retrieve_respects_top_k(self, mock_embed, hybrid_retriever):
        retriever, _, _ = hybrid_retriever
        results = retriever.retrieve("test query", top_k=2)
        assert len(results) == 2

    @patch("retrieval.hybrid_retriever.embed_query", return_value=[0.1] * 384)
    def test_passes_filters_to_both(self, mock_embed, hybrid_retriever):
        retriever, vs, bm25 = hybrid_retriever
        where = {"year": {"$eq": 2017}}
        retriever.retrieve("test", top_k=5, where=where)
        vs.query.assert_called_once()
        assert vs.query.call_args.kwargs.get("where") == where
        bm25.query.assert_called_once()
        assert bm25.query.call_args.kwargs.get("filters") is not None


class TestChromeDBWhereToFlat:
    def test_single_condition(self):
        flat = HybridRetriever._chromadb_where_to_flat({"year": {"$eq": 2020}})
        assert flat == {"year": 2020}

    def test_and_conditions(self):
        where = {"$and": [
            {"year": {"$eq": 2020}},
            {"conference": {"$eq": "NeurIPS"}},
        ]}
        flat = HybridRetriever._chromadb_where_to_flat(where)
        assert flat["year"] == 2020
        assert flat["conference"] == "NeurIPS"

    def test_none_input(self):
        assert HybridRetriever._chromadb_where_to_flat(None) is None
