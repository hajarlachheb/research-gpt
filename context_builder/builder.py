"""Construct the prompt context from retrieved and reranked chunks."""

from __future__ import annotations

import tiktoken

from config.settings import get_settings
from models.schemas import RetrievedChunk, SourceCitation
from monitoring.logger import get_logger

logger = get_logger(__name__)


class ContextBuilder:
    def __init__(self):
        self.settings = get_settings()
        try:
            self._enc = tiktoken.encoding_for_model(self.settings.openai_model)
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def build(self, chunks: list[RetrievedChunk]) -> tuple[str, list[SourceCitation]]:
        """
        Build a formatted context string and collect source citations.

        Returns (context_text, citations) where context_text fits inside
        the configured max_context_tokens budget.
        """
        max_tokens = self.settings.max_context_tokens
        context_parts: list[str] = []
        citations: list[SourceCitation] = []
        seen_papers: set[str] = set()
        current_tokens = 0

        for rc in chunks:
            chunk = rc.chunk
            block = self._format_block(chunk.paper_title, chunk.year, chunk.section, chunk.text)
            block_tokens = len(self._enc.encode(block))

            if current_tokens + block_tokens > max_tokens:
                break

            context_parts.append(block)
            current_tokens += block_tokens

            citation_key = f"{chunk.paper_title}::{chunk.section}"
            if citation_key not in seen_papers:
                seen_papers.add(citation_key)
                citations.append(SourceCitation(
                    paper=chunk.paper_title,
                    authors=chunk.authors,
                    year=chunk.year,
                    section=chunk.section,
                    source_url=chunk.source_url,
                    relevance_score=rc.score,
                ))

        context_text = "\n\n---\n\n".join(context_parts)
        logger.info(
            "context_built",
            chunks_used=len(context_parts),
            token_count=current_tokens,
            citations=len(citations),
        )
        return context_text, citations

    @staticmethod
    def _format_block(title: str, year: int | None, section: str | None, text: str) -> str:
        header_parts = [f"Paper: {title}"]
        if year:
            header_parts[0] += f" ({year})"
        if section:
            header_parts.append(f"Section: {section}")
        header = "\n".join(header_parts)
        return f"{header}\n\n{text}"
