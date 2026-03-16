"""Generate embeddings using Sentence Transformers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a batch of texts and return a list of float vectors."""
    model = _load_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = _load_model()
    return model.encode(query, show_progress_bar=False).tolist()
