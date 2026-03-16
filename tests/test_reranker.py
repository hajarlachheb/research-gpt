"""Tests for the cross-encoder reranker."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_rr")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from models.schemas import DocumentChunk, RetrievedChunk
from reranker.cross_encoder import CrossEncoderReranker


@pytest.fixture
def sample_retrieved():
    chunks = [
        DocumentChunk(chunk_id=f"rr_{i}", text=f"Chunk text {i}", paper_title="Paper", year=2020)
        for i in range(5)
    ]
    return [RetrievedChunk(chunk=c, score=0.5) for c in chunks]


class TestCrossEncoderReranker:
    @patch("reranker.cross_encoder._load_cross_encoder")
    def test_rerank_returns_top_k(self, mock_load, sample_retrieved):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        mock_load.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", sample_retrieved, top_k=3)
        assert len(results) == 3

    @patch("reranker.cross_encoder._load_cross_encoder")
    def test_rerank_sorted_by_score(self, mock_load, sample_retrieved):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        mock_load.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", sample_retrieved, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("reranker.cross_encoder._load_cross_encoder")
    def test_rerank_preserves_chunk_data(self, mock_load, sample_retrieved):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
        mock_load.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", sample_retrieved, top_k=1)
        assert results[0].chunk.chunk_id == "rr_0"

    def test_rerank_empty_input(self):
        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", [])
        assert results == []
