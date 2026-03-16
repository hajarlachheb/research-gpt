"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("api.dependencies.AppState") as MockState:
        mock_state = MagicMock()
        mock_state.vector_store = MagicMock()
        mock_state.vector_store.count = 100
        mock_state.vector_store.get_all_papers.return_value = [
            {
                "title": "Test Paper",
                "authors": ["Author A"],
                "year": 2023,
                "source_url": None,
                "conference": "NeurIPS",
                "chunk_count": 5,
            }
        ]
        mock_state.cache = MagicMock()
        mock_state.cache._redis = None
        mock_state.cache.get.return_value = None
        mock_state.tracer = MagicMock()
        mock_state.tracer.start_trace.return_value = MagicMock()
        mock_state.tracer.start_span.return_value = MagicMock()
        mock_state.tracer.get_recent_traces.return_value = []

        mock_state.query_rewriter = MagicMock()
        mock_state.query_rewriter.rewrite = AsyncMock(return_value="rewritten query")
        mock_state.hybrid_retriever = MagicMock()
        mock_state.hybrid_retriever.retrieve.return_value = []
        mock_state.reranker = MagicMock()
        mock_state.reranker.rerank.return_value = []
        mock_state.context_builder = MagicMock()
        mock_state.context_builder.build.return_value = ("context", [])
        mock_state.llm_generator = MagicMock()
        mock_state.llm_generator.generate = AsyncMock(
            return_value=("Test answer", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        )
        mock_state.prompt_manager = MagicMock()
        mock_state.prompt_manager.list_prompts.return_value = {"answer": [1], "rewrite": [1]}

        with patch("api.routes.get_app_state", return_value=mock_state):
            with patch("api.main.get_app_state", return_value=mock_state):
                from api.main import create_app
                app = create_app()
                yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data


class TestPapersEndpoint:
    def test_list_papers(self, client):
        response = client.get("/papers")
        assert response.status_code == 200
        papers = response.json()
        assert isinstance(papers, list)


class TestAskEndpoint:
    def test_ask_returns_answer(self, client):
        response = client.post("/ask", json={"question": "What is attention?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "query" in data

    def test_ask_empty_question(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 200


class TestPromptsEndpoint:
    def test_list_prompts(self, client):
        response = client.get("/prompts")
        assert response.status_code == 200


class TestTracesEndpoint:
    def test_get_traces(self, client):
        response = client.get("/traces")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
