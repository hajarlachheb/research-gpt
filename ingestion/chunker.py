"""Chunk documents with section awareness, overlap, and optional semantic grouping."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from models.schemas import DocumentChunk, PaperMetadata


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    use_semantic_chunking: bool = True
    similarity_threshold: float = 0.5


def _generate_chunk_id(paper_title: str, section: str, index: int) -> str:
    raw = f"{paper_title}::{section}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_text(
    text: str,
    section: str,
    metadata: PaperMetadata,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Split text into overlapping chunks preserving sentence boundaries.

    When ``use_semantic_chunking`` is enabled the splitter first groups
    consecutive sentences whose embeddings are similar, then falls back to
    size-based splitting within each group.
    """
    config = config or ChunkingConfig()
    sentences = _split_sentences(text)

    if config.use_semantic_chunking and len(sentences) > 2:
        groups = _semantic_group_sentences(sentences, config)
    else:
        groups = [sentences]

    chunks: list[DocumentChunk] = []
    for group in groups:
        chunks.extend(_size_based_chunking(group, section, metadata, config, offset=len(chunks)))

    return chunks


def _semantic_group_sentences(
    sentences: list[str],
    config: ChunkingConfig,
) -> list[list[str]]:
    """Group consecutive sentences whose embeddings are above a similarity threshold."""
    try:
        from ingestion.embedder import embed_texts
        embeddings = embed_texts(sentences, batch_size=128)
        emb_array = np.array(embeddings)
    except Exception:
        return [sentences]

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(emb_array[i - 1], emb_array[i])
        if sim >= config.similarity_threshold:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    return groups


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _size_based_chunking(
    sentences: list[str],
    section: str,
    metadata: PaperMetadata,
    config: ChunkingConfig,
    offset: int = 0,
) -> list[DocumentChunk]:
    """Fixed-size chunking with overlap on a list of sentences."""
    chunks: list[DocumentChunk] = []
    current_tokens: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > config.chunk_size and current_tokens:
            chunk_text_str = " ".join(current_tokens)
            if len(chunk_text_str.split()) >= config.min_chunk_size:
                chunks.append(_make_chunk(
                    chunk_text_str, section, metadata, offset + len(chunks),
                ))
            overlap_tokens = _get_overlap(current_tokens, config.chunk_overlap)
            current_tokens = overlap_tokens
            current_len = sum(len(t.split()) for t in current_tokens)

        current_tokens.append(sentence)
        current_len += sentence_len

    if current_tokens:
        chunk_text_str = " ".join(current_tokens)
        if len(chunk_text_str.split()) >= config.min_chunk_size:
            chunks.append(_make_chunk(
                chunk_text_str, section, metadata, offset + len(chunks),
            ))

    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _get_overlap(tokens: list[str], overlap_words: int) -> list[str]:
    """Return trailing sentences whose cumulative words reach overlap_words."""
    result: list[str] = []
    count = 0
    for sentence in reversed(tokens):
        wc = len(sentence.split())
        count += wc
        result.insert(0, sentence)
        if count >= overlap_words:
            break
    return result


def _make_chunk(
    text: str,
    section: str,
    metadata: PaperMetadata,
    index: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=_generate_chunk_id(metadata.title, section, index),
        text=text,
        paper_title=metadata.title,
        authors=metadata.authors,
        year=metadata.year,
        section=section,
        source_url=metadata.source_url,
        conference=metadata.conference,
    )
