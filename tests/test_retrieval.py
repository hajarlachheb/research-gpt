"""Tests for retrieval and reranking components."""

from __future__ import annotations

import pytest
from models.schemas import DocumentChunk, RetrievedChunk
from retrieval.bm25_retriever import BM25Retriever
from context_builder.builder import ContextBuilder


class TestBM25Retriever:
    def test_add_and_query(self, sample_chunks):
        bm25 = BM25Retriever()
        bm25.add_chunks(sample_chunks)
        results = bm25.query("self-attention mechanism", top_k=5)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_empty_query(self):
        bm25 = BM25Retriever()
        results = bm25.query("anything", top_k=5)
        assert results == []

    def test_ranking_order(self, sample_chunks):
        bm25 = BM25Retriever()
        bm25.add_chunks(sample_chunks)
        results = bm25.query("multi-head attention subspaces", top_k=5)
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_filter_by_year(self, sample_chunks):
        bm25 = BM25Retriever()
        bm25.add_chunks(sample_chunks)
        results = bm25.query("attention", top_k=5, filters={"year": 9999})
        assert results == []

    def test_filter_by_author(self, sample_chunks):
        bm25 = BM25Retriever()
        bm25.add_chunks(sample_chunks)
        results = bm25.query("attention", top_k=5, filters={"author": "Vaswani"})
        assert all("Vaswani" in r.chunk.authors for r in results)


class TestContextBuilder:
    def test_build_context(self, sample_retrieved):
        builder = ContextBuilder()
        context, citations = builder.build(sample_retrieved)
        assert "Attention Is All You Need" in context
        assert len(citations) > 0
        assert citations[0].paper == "Attention Is All You Need"

    def test_citations_deduplicated(self, sample_chunks):
        duplicated = [
            RetrievedChunk(chunk=sample_chunks[0], score=0.9),
            RetrievedChunk(chunk=sample_chunks[0], score=0.8),
        ]
        builder = ContextBuilder()
        _, citations = builder.build(duplicated)
        papers = [c.paper for c in citations]
        sections = [c.section for c in citations]
        paired = list(zip(papers, sections))
        assert len(paired) == len(set(paired))

    def test_empty_chunks(self):
        builder = ContextBuilder()
        context, citations = builder.build([])
        assert context == ""
        assert citations == []
