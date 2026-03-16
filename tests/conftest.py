"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from models.schemas import PaperMetadata, DocumentChunk, RetrievedChunk, EvalSample


@pytest.fixture
def sample_metadata() -> PaperMetadata:
    return PaperMetadata(
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer", "Parmar"],
        year=2017,
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        source_url="https://arxiv.org/abs/1706.03762",
        conference="NeurIPS",
    )


@pytest.fixture
def sample_text() -> str:
    return (
        "The Transformer model architecture relies entirely on self-attention mechanisms "
        "to compute representations of its input and output without using sequence-aligned "
        "recurrence. The encoder maps an input sequence to a sequence of continuous representations. "
        "Given these representations, the decoder then generates an output sequence of symbols "
        "one element at a time. The model is auto-regressive, consuming the previously generated "
        "symbols as additional input when generating the next. "
        "Multi-head attention allows the model to jointly attend to information from different "
        "representation subspaces at different positions. With a single attention head, averaging "
        "inhibits this. The Transformer uses multi-head attention in three different ways. "
        "First, in encoder-decoder attention layers, the queries come from the previous decoder layer, "
        "and the memory keys and values come from the output of the encoder. This allows every "
        "position in the decoder to attend over all positions in the input sequence."
    )


@pytest.fixture
def sample_chunks(sample_metadata) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_001",
            text="The Transformer model architecture relies entirely on self-attention mechanisms.",
            paper_title=sample_metadata.title,
            authors=sample_metadata.authors,
            year=sample_metadata.year,
            section="Methodology",
            source_url=sample_metadata.source_url,
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            text="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
            paper_title=sample_metadata.title,
            authors=sample_metadata.authors,
            year=sample_metadata.year,
            section="Multi-Head Attention",
            source_url=sample_metadata.source_url,
        ),
    ]


@pytest.fixture
def sample_retrieved(sample_chunks) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=sample_chunks[0], score=0.92),
        RetrievedChunk(chunk=sample_chunks[1], score=0.87),
    ]


@pytest.fixture
def eval_samples() -> list[EvalSample]:
    return [
        EvalSample(
            question="What problem does the transformer solve?",
            ground_truth="Transformers remove recurrence and rely on attention mechanisms.",
            relevant_papers=["Attention Is All You Need"],
        ),
    ]
