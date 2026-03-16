"""Manage evaluation datasets for the RAG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from models.schemas import EvalSample
from monitoring.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = [
    EvalSample(
        question="What problem does the transformer architecture solve?",
        ground_truth=(
            "Transformers remove the need for recurrence and convolutions, "
            "relying entirely on self-attention mechanisms to draw global "
            "dependencies between input and output."
        ),
        relevant_papers=["Attention Is All You Need"],
    ),
    EvalSample(
        question="What are the key differences between BERT and RoBERTa?",
        ground_truth=(
            "RoBERTa improves on BERT by training longer with bigger batches, "
            "removing the next-sentence prediction objective, using dynamic "
            "masking, and training on more data."
        ),
        relevant_papers=[
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        ],
    ),
    EvalSample(
        question="How do diffusion models generate images?",
        ground_truth=(
            "Diffusion models generate images by learning to reverse a gradual "
            "noising process. They start from pure noise and iteratively denoise "
            "to produce coherent images, guided by a learned score function."
        ),
        relevant_papers=["Denoising Diffusion Probabilistic Models"],
    ),
    EvalSample(
        question="What is the advantage of the attention mechanism?",
        ground_truth=(
            "The attention mechanism allows models to focus on relevant parts "
            "of the input regardless of distance, enabling better modeling of "
            "long-range dependencies compared to recurrent architectures."
        ),
        relevant_papers=["Attention Is All You Need"],
    ),
    EvalSample(
        question="What training data was used for GPT-3?",
        ground_truth=(
            "GPT-3 was trained on a filtered version of Common Crawl, WebText2, "
            "Books1, Books2, and English Wikipedia, totaling about 570GB of text."
        ),
        relevant_papers=["Language Models are Few-Shot Learners"],
    ),
]


class EvalDatasetManager:
    def __init__(self, dataset_path: Optional[str] = None):
        self._samples: list[EvalSample] = []
        if dataset_path and Path(dataset_path).exists():
            self.load(dataset_path)
        else:
            self._samples = list(DEFAULT_DATASET)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._samples = [EvalSample(**item) for item in data]
        logger.info("eval_dataset_loaded", count=len(self._samples))

    def save(self, path: str) -> None:
        data = [s.model_dump() for s in self._samples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_sample(self, sample: EvalSample) -> None:
        self._samples.append(sample)

    @property
    def samples(self) -> list[EvalSample]:
        return self._samples
