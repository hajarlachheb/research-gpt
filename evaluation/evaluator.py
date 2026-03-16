"""Automated RAG evaluation pipeline using Ragas metrics."""

from __future__ import annotations

from typing import Optional

from models.schemas import EvalResult, EvalSample
from monitoring.logger import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    """Evaluate RAG pipeline quality using Ragas framework."""

    def __init__(self, ask_fn=None):
        """
        Args:
            ask_fn: Async callable that takes a question string and returns
                     (answer, contexts_list) tuple.
        """
        self._ask_fn = ask_fn

    async def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """Run evaluation on a batch of samples and compute metrics."""
        if not samples:
            return EvalResult(num_samples=0)

        questions = []
        ground_truths = []
        answers = []
        contexts = []

        for sample in samples:
            if self._ask_fn is None:
                logger.warning("no_ask_fn_provided_skipping_evaluation")
                return EvalResult(num_samples=0)

            answer, retrieved_contexts = await self._ask_fn(sample.question)
            questions.append(sample.question)
            ground_truths.append(sample.ground_truth)
            answers.append(answer)
            contexts.append(retrieved_contexts)

        try:
            result = await self._compute_ragas_metrics(
                questions, answers, contexts, ground_truths,
            )
        except ImportError:
            logger.warning("ragas_not_installed_computing_basic_metrics")
            result = self._compute_basic_metrics(
                questions, answers, contexts, ground_truths,
            )

        result.num_samples = len(samples)
        logger.info("evaluation_complete", **result.model_dump())
        return result

    async def _compute_ragas_metrics(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> EvalResult:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        return EvalResult(
            faithfulness=result.get("faithfulness"),
            answer_relevancy=result.get("answer_relevancy"),
            context_precision=result.get("context_precision"),
            context_recall=result.get("context_recall"),
        )

    @staticmethod
    def _compute_basic_metrics(
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> EvalResult:
        """Fallback metrics when Ragas is not available."""
        details = []
        total_relevance = 0.0

        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
            gt_words = set(gt.lower().split())
            a_words = set(a.lower().split())
            overlap = len(gt_words & a_words) / max(len(gt_words), 1)

            has_context = len(ctx) > 0
            not_hallucinated = "cannot find" not in a.lower() if has_context else True

            total_relevance += overlap
            details.append({
                "question": q[:80],
                "word_overlap": round(overlap, 3),
                "has_context": has_context,
                "not_hallucinated": not_hallucinated,
            })

        n = len(questions)
        return EvalResult(
            answer_relevancy=round(total_relevance / max(n, 1), 3),
            details=details,
        )
