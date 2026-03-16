"""Tests for the LLM generator and query rewriter."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_llm")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from llm.generator import LLMGenerator
from llm.query_rewriter import QueryRewriter


@pytest.fixture
def mock_prompt_manager():
    pm = MagicMock()
    pm.get_prompt.return_value = "You are a test assistant."
    return pm


class TestLLMGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_answer_and_usage(self, mock_prompt_manager):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        with patch.object(LLMGenerator, "__init__", lambda self, pm=None: None):
            gen = LLMGenerator()
            gen.settings = MagicMock()
            gen.settings.openai_model = "gpt-3.5-turbo"
            gen.settings.max_response_tokens = 1024
            gen._prompt_manager = mock_prompt_manager
            gen._client = MagicMock()
            gen._client.chat.completions.create = AsyncMock(return_value=mock_response)

            answer, usage = await gen.generate("What is attention?", "context text")
            assert answer == "Test answer"
            assert usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_generate_handles_no_usage(self, mock_prompt_manager):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_response.usage = None

        with patch.object(LLMGenerator, "__init__", lambda self, pm=None: None):
            gen = LLMGenerator()
            gen.settings = MagicMock()
            gen.settings.openai_model = "gpt-3.5-turbo"
            gen.settings.max_response_tokens = 1024
            gen._prompt_manager = mock_prompt_manager
            gen._client = MagicMock()
            gen._client.chat.completions.create = AsyncMock(return_value=mock_response)

            answer, usage = await gen.generate("Q", "C")
            assert usage["total_tokens"] == 0


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_rewrite_returns_rewritten_query(self, mock_prompt_manager):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Expanded query about transformers"

        with patch.object(QueryRewriter, "__init__", lambda self, pm=None: None):
            rewriter = QueryRewriter()
            rewriter.settings = MagicMock()
            rewriter.settings.openai_model = "gpt-3.5-turbo"
            rewriter._prompt_manager = mock_prompt_manager
            rewriter._client = MagicMock()
            rewriter._client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await rewriter.rewrite("transformers")
            assert result == "Expanded query about transformers"

    @pytest.mark.asyncio
    async def test_rewrite_strips_quotes(self, mock_prompt_manager):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '"Expanded query"'

        with patch.object(QueryRewriter, "__init__", lambda self, pm=None: None):
            rewriter = QueryRewriter()
            rewriter.settings = MagicMock()
            rewriter.settings.openai_model = "gpt-3.5-turbo"
            rewriter._prompt_manager = mock_prompt_manager
            rewriter._client = MagicMock()
            rewriter._client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await rewriter.rewrite("test")
            assert result == "Expanded query"

    @pytest.mark.asyncio
    async def test_rewrite_returns_original_on_error(self, mock_prompt_manager):
        with patch.object(QueryRewriter, "__init__", lambda self, pm=None: None):
            rewriter = QueryRewriter()
            rewriter.settings = MagicMock()
            rewriter.settings.openai_model = "gpt-3.5-turbo"
            rewriter._prompt_manager = mock_prompt_manager
            rewriter._client = MagicMock()
            rewriter._client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

            result = await rewriter.rewrite("original query")
            assert result == "original query"
