"""Tests for the evaluation pipeline."""

from __future__ import annotations

import os
import json
import tempfile
import pytest
from unittest.mock import AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_eval")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from models.schemas import EvalSample, EvalResult
from evaluation.evaluator import RAGEvaluator
from evaluation.dataset import EvalDatasetManager, DEFAULT_DATASET


class TestEvalDatasetManager:
    def test_loads_default_dataset(self):
        manager = EvalDatasetManager()
        assert len(manager.samples) == len(DEFAULT_DATASET)

    def test_add_sample(self):
        manager = EvalDatasetManager()
        initial = len(manager.samples)
        manager.add_sample(EvalSample(
            question="Test?", ground_truth="Answer.", relevant_papers=["Paper"],
        ))
        assert len(manager.samples) == initial + 1

    def test_save_and_load(self):
        manager = EvalDatasetManager()
        manager.add_sample(EvalSample(
            question="Custom?", ground_truth="Custom answer.", relevant_papers=[],
        ))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            manager.save(path)
            loaded = EvalDatasetManager(path)
            assert len(loaded.samples) == len(manager.samples)
            assert loaded.samples[-1].question == "Custom?"
        finally:
            os.unlink(path)


class TestRAGEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_with_no_ask_fn(self):
        evaluator = RAGEvaluator(ask_fn=None)
        result = await evaluator.evaluate([
            EvalSample(question="Q?", ground_truth="A.", relevant_papers=[]),
        ])
        assert result.num_samples == 0

    @pytest.mark.asyncio
    async def test_evaluate_empty_samples(self):
        evaluator = RAGEvaluator(ask_fn=AsyncMock())
        result = await evaluator.evaluate([])
        assert result.num_samples == 0

    @pytest.mark.asyncio
    async def test_basic_metrics_fallback(self):
        ask_fn = AsyncMock(return_value=("Transformers use attention mechanisms.", ["context"]))
        evaluator = RAGEvaluator(ask_fn=ask_fn)

        samples = [
            EvalSample(
                question="What is a transformer?",
                ground_truth="Transformers rely on attention mechanisms.",
                relevant_papers=["Paper A"],
            ),
        ]

        result = await evaluator.evaluate(samples)
        assert result.num_samples == 1
        assert result.answer_relevancy is not None or result.details
