"""Tests for the embedding module."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_emb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")


class TestEmbedder:
    @patch("ingestion.embedder._load_model")
    def test_embed_texts_returns_list(self, mock_load):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_load.return_value = mock_model

        from ingestion.embedder import embed_texts
        result = embed_texts(["hello", "world"])
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert len(result[0]) == 2

    @patch("ingestion.embedder._load_model")
    def test_embed_query_returns_vector(self, mock_load):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.5, 0.6, 0.7])
        mock_load.return_value = mock_model

        from ingestion.embedder import embed_query
        result = embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 3

    @patch("ingestion.embedder._load_model")
    def test_embed_texts_batch_size(self, mock_load):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_load.return_value = mock_model

        from ingestion.embedder import embed_texts
        embed_texts(["test"], batch_size=32)
        mock_model.encode.assert_called_once()
        assert mock_model.encode.call_args.kwargs.get("batch_size") == 32
