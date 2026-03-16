"""Tests for monitoring tracer and logger."""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_mon")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from monitoring.tracer import RAGTracer, RAGTrace, NoOpTracer


class TestRAGTracer:
    def test_start_trace(self):
        tracer = RAGTracer()
        trace = tracer.start_trace("test query")
        assert trace.query == "test query"
        assert trace.trace_id

    def test_start_span(self):
        tracer = RAGTracer()
        trace = tracer.start_trace("q")
        span = tracer.start_span(trace, "retrieval")
        assert span.name == "retrieval"
        assert len(trace.spans) == 1

    def test_finish_span(self):
        tracer = RAGTracer()
        trace = tracer.start_trace("q")
        span = tracer.start_span(trace, "test_span")
        span.finish(doc_count=5)
        assert span.end_time is not None
        assert span.metadata["doc_count"] == 5
        assert span.duration_ms >= 0

    def test_finish_trace_computes_latency(self):
        tracer = RAGTracer()
        trace = tracer.start_trace("q")
        span = tracer.start_span(trace, "s1")
        span.finish()
        tracer.finish_trace(trace)
        assert trace.total_latency_ms >= 0

    def test_get_recent_traces(self):
        tracer = RAGTracer()
        for i in range(5):
            t = tracer.start_trace(f"q{i}")
            tracer.finish_trace(t)
        recent = tracer.get_recent_traces(3)
        assert len(recent) == 3

    def test_export_trace(self):
        tracer = RAGTracer()
        trace = tracer.start_trace("q")
        trace.prompt_tokens = 100
        trace.completion_tokens = 50
        trace.total_tokens = 150
        exported = tracer.export_trace(trace)
        assert exported["query"] == "q"
        assert exported["token_usage"]["total_tokens"] == 150


class TestNoOpTracer:
    def test_noop_records_nothing(self):
        tracer = NoOpTracer()
        trace = tracer.start_trace("q")
        span = tracer.start_span(trace, "s")
        span.finish()
        tracer.finish_trace(trace)
        assert tracer.get_recent_traces() == []
