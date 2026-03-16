"""BM25 keyword retrieval over stored document chunks."""

from __future__ import annotations

import re
import threading
from typing import Optional

from rank_bm25 import BM25Okapi

from models.schemas import DocumentChunk, RetrievedChunk
from monitoring.logger import get_logger

logger = get_logger(__name__)

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "not", "no", "as", "if", "then", "so",
    "than", "too", "very", "just", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "each", "all", "both",
    "such", "only", "own", "same", "other", "which", "who", "whom", "what",
    "when", "where", "why", "how", "up", "out", "also",
})

_WORD_RE = re.compile(r"[a-z0-9]+")


class BM25Retriever:
    """In-memory BM25 index over document chunk texts."""

    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._tokenized: list[list[str]] = []
        self._index: Optional[BM25Okapi] = None
        self._lock = threading.Lock()

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        with self._lock:
            self._chunks.extend(chunks)
            new_tokenized = [self._tokenize(c.text) for c in chunks]
            self._tokenized.extend(new_tokenized)
            self._rebuild_index()

    def query(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        with self._lock:
            if self._index is None or not self._chunks:
                return []

            tokenized_query = self._tokenize(query)
            scores = self._index.get_scores(tokenized_query)

            scored_pairs = sorted(
                zip(self._chunks, scores), key=lambda x: x[1], reverse=True
            )

            results: list[RetrievedChunk] = []
            for chunk, score in scored_pairs:
                if score <= 0:
                    continue
                if filters and not self._matches_filters(chunk, filters):
                    continue
                results.append(RetrievedChunk(chunk=chunk, score=float(score)))
                if len(results) >= top_k:
                    break

            return results

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, filters: dict) -> bool:
        if "year" in filters and chunk.year != filters["year"]:
            return False
        if "author" in filters:
            author_str = "|".join(chunk.authors).lower()
            if filters["author"].lower() not in author_str:
                return False
        if "conference" in filters and (chunk.conference or "").lower() != filters["conference"].lower():
            return False
        if "title" in filters and filters["title"].lower() not in (chunk.paper_title or "").lower():
            return False
        return True

    def _rebuild_index(self) -> None:
        if self._tokenized:
            self._index = BM25Okapi(self._tokenized)
            logger.info("bm25_index_rebuilt", doc_count=len(self._tokenized))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = _WORD_RE.findall(text.lower())
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
