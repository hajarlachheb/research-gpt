"""Hybrid retriever combining vector similarity and BM25 keyword search."""

from __future__ import annotations

from typing import Optional

from config.settings import get_settings
from ingestion.embedder import embed_query
from models.schemas import RetrievedChunk
from retrieval.vector_store import VectorStore
from retrieval.bm25_retriever import BM25Retriever
from monitoring.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_retriever: BM25Retriever):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid retrieval: vector + BM25, fused with weighted RRF."""
        top_k = top_k or self.settings.retrieval_top_k

        query_embedding = embed_query(query)
        vector_results = self.vector_store.query(query_embedding, top_k=top_k, where=where)
        bm25_results = self.bm25.query(query, top_k=top_k, filters=self._chromadb_where_to_flat(where))

        fused = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            vector_weight=self.settings.vector_weight,
            bm25_weight=self.settings.bm25_weight,
        )

        logger.info(
            "hybrid_retrieval_complete",
            query_len=len(query),
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            fused_count=len(fused),
        )
        return fused[:top_k]

    @staticmethod
    def _chromadb_where_to_flat(where: Optional[dict]) -> Optional[dict]:
        """Convert ChromaDB where clause back to flat {field: value} for BM25 filtering."""
        if not where:
            return None
        flat: dict = {}
        conditions = where.get("$and", [where])
        field_map = {"paper_title": "title", "authors": "author"}
        for cond in conditions:
            for field_key, op in cond.items():
                if field_key.startswith("$"):
                    continue
                if isinstance(op, dict):
                    val = next(iter(op.values()))
                else:
                    val = op
                flat_key = field_map.get(field_key, field_key)
                flat[flat_key] = val
        return flat if flat else None

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """Weighted Reciprocal Rank Fusion of two result sets."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, rc in enumerate(vector_results):
            cid = rc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + vector_weight / (k + rank + 1)
            chunk_map[cid] = rc

        for rank, rc in enumerate(bm25_results):
            cid = rc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + bm25_weight / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = rc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(chunk=chunk_map[cid].chunk, score=score)
            for cid, score in ranked
            if cid in chunk_map
        ]
