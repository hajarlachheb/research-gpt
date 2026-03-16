"""End-to-end ingestion pipeline orchestrating extraction, cleaning, chunking, and storage."""

from __future__ import annotations

import structlog
from pathlib import Path

from config.settings import get_settings
from models.schemas import PaperMetadata, DocumentChunk, IngestResponse
from ingestion.pdf_extractor import extract_pdf, ExtractedSection
from ingestion.text_cleaner import clean_text
from ingestion.chunker import chunk_text, ChunkingConfig
from ingestion.embedder import embed_texts
from retrieval.vector_store import VectorStore
from monitoring.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.settings = get_settings()

    def ingest_pdf(self, file_path: str | Path, metadata: PaperMetadata) -> IngestResponse:
        """Ingest a PDF file: extract -> clean -> chunk -> embed -> store."""
        logger.info("ingesting_pdf", file_path=str(file_path), title=metadata.title)

        sections = extract_pdf(file_path)
        chunks = self._process_sections(sections, metadata)
        self._store_chunks(chunks)

        logger.info("ingestion_complete", title=metadata.title, chunk_count=len(chunks))
        return IngestResponse(paper_title=metadata.title, chunks_created=len(chunks))

    def ingest_text(self, text: str, metadata: PaperMetadata, section: str = "Full Text") -> IngestResponse:
        """Ingest raw text directly."""
        logger.info("ingesting_text", title=metadata.title)

        cleaned = clean_text(text)
        config = ChunkingConfig(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = chunk_text(cleaned, section, metadata, config)
        self._store_chunks(chunks)

        logger.info("ingestion_complete", title=metadata.title, chunk_count=len(chunks))
        return IngestResponse(paper_title=metadata.title, chunks_created=len(chunks))

    def _process_sections(
        self,
        sections: list[ExtractedSection],
        metadata: PaperMetadata,
    ) -> list[DocumentChunk]:
        config = ChunkingConfig(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        all_chunks: list[DocumentChunk] = []
        for section in sections:
            cleaned = clean_text(section.text)
            if len(cleaned.split()) < 20:
                continue
            chunks = chunk_text(cleaned, section.title, metadata, config)
            all_chunks.extend(chunks)
        return all_chunks

    def _store_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        self.vector_store.add_chunks(chunks, embeddings)
