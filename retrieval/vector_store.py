"""ChromaDB-backed vector store for document chunks."""

from __future__ import annotations

from typing import Optional

import chromadb

from config.settings import get_settings
from models.schemas import DocumentChunk, RetrievedChunk
from monitoring.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vector_store_initialized", collection=settings.chroma_collection)

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Add document chunks with pre-computed embeddings."""
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_chunk_to_metadata(c) for c in chunks],
        )
        logger.info("chunks_added", count=len(chunks))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Query the vector store and return scored chunks."""
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        retrieved: list[RetrievedChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for idx, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][idx] if results["metadatas"] else {}
            text = results["documents"][0][idx] if results["documents"] else ""
            distance = results["distances"][0][idx] if results["distances"] else 1.0
            score = 1.0 - distance  # cosine distance -> similarity

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                paper_title=meta.get("paper_title", ""),
                authors=meta.get("authors", "").split("|") if meta.get("authors") else [],
                year=meta.get("year"),
                section=meta.get("section"),
                source_url=meta.get("source_url"),
                conference=meta.get("conference"),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        return retrieved

    def get_all_papers(self) -> list[dict]:
        """Return unique papers currently stored."""
        all_data = self._collection.get(include=["metadatas"])
        papers: dict[str, dict] = {}
        for meta in (all_data["metadatas"] or []):
            title = meta.get("paper_title", "Unknown")
            if title not in papers:
                papers[title] = {
                    "title": title,
                    "authors": meta.get("authors", "").split("|") if meta.get("authors") else [],
                    "year": meta.get("year"),
                    "source_url": meta.get("source_url"),
                    "conference": meta.get("conference"),
                    "chunk_count": 0,
                }
            papers[title]["chunk_count"] += 1
        return list(papers.values())

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Load all stored chunks for index rebuilding (e.g. BM25 on startup)."""
        total = self._collection.count()
        if total == 0:
            return []

        all_data = self._collection.get(include=["documents", "metadatas"])
        chunks: list[DocumentChunk] = []
        for idx, chunk_id in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][idx] if all_data["metadatas"] else {}
            text = all_data["documents"][idx] if all_data["documents"] else ""
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                paper_title=meta.get("paper_title", ""),
                authors=meta.get("authors", "").split("|") if meta.get("authors") else [],
                year=meta.get("year"),
                section=meta.get("section"),
                source_url=meta.get("source_url"),
                conference=meta.get("conference"),
            ))
        return chunks

    @property
    def count(self) -> int:
        return self._collection.count()


def _chunk_to_metadata(chunk: DocumentChunk) -> dict:
    return {
        "paper_title": chunk.paper_title,
        "authors": "|".join(chunk.authors),
        "year": chunk.year or 0,
        "section": chunk.section or "",
        "source_url": chunk.source_url or "",
        "conference": chunk.conference or "",
    }
