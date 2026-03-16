"""Cross-encoder reranker to refine retrieved results."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from sentence_transformers import CrossEncoder

from config.settings import get_settings
from models.schemas import RetrievedChunk
from monitoring.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model, max_length=512)


class CrossEncoderReranker:
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks using a cross-encoder model."""
        top_k = top_k or get_settings().rerank_top_k

        if not chunks:
            return []

        model = _load_cross_encoder()
        pairs = [[query, rc.chunk.text] for rc in chunks]
        scores = model.predict(pairs)

        reranked = [
            RetrievedChunk(chunk=rc.chunk, score=float(score))
            for rc, score in zip(chunks, scores)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "reranking_complete",
            input_count=len(chunks),
            output_count=min(top_k, len(reranked)),
        )
        return reranked[:top_k]
