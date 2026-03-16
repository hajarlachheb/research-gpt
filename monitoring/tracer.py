"""RAG pipeline tracer for full observability of each query lifecycle."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TraceSpan:
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, **extra: Any) -> None:
        self.end_time = time.time()
        self.metadata.update(extra)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


@dataclass
class RAGTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    query: str = ""
    rewritten_query: Optional[str] = None
    retrieved_chunks: int = 0
    reranked_chunks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    answer_length: int = 0
    total_latency_ms: float = 0.0
    spans: list[TraceSpan] = field(default_factory=list)
    error: Optional[str] = None


class RAGTracer:
    """Captures the full lifecycle of a RAG query for monitoring."""

    def __init__(self):
        self._traces: list[RAGTrace] = []

    def start_trace(self, query: str) -> RAGTrace:
        trace = RAGTrace(query=query)
        self._traces.append(trace)
        return trace

    def start_span(self, trace: RAGTrace, name: str) -> TraceSpan:
        span = TraceSpan(name=name)
        trace.spans.append(span)
        return span

    def finish_trace(self, trace: RAGTrace) -> None:
        if trace.spans:
            first_start = min(s.start_time for s in trace.spans)
            last_end = max(s.end_time or s.start_time for s in trace.spans)
            trace.total_latency_ms = (last_end - first_start) * 1000

        logger.info(
            "rag_trace_complete",
            trace_id=trace.trace_id,
            query=trace.query[:100],
            latency_ms=round(trace.total_latency_ms, 2),
            retrieved=trace.retrieved_chunks,
            reranked=trace.reranked_chunks,
            total_tokens=trace.total_tokens,
            error=trace.error,
        )

    def get_recent_traces(self, limit: int = 50) -> list[RAGTrace]:
        return self._traces[-limit:]

    def export_trace(self, trace: RAGTrace) -> dict:
        """Export a trace as a serializable dictionary."""
        return {
            "trace_id": trace.trace_id,
            "timestamp": trace.timestamp,
            "query": trace.query,
            "rewritten_query": trace.rewritten_query,
            "retrieved_chunks": trace.retrieved_chunks,
            "reranked_chunks": trace.reranked_chunks,
            "token_usage": {
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "total_tokens": trace.total_tokens,
            },
            "answer_length": trace.answer_length,
            "total_latency_ms": round(trace.total_latency_ms, 2),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "metadata": s.metadata,
                }
                for s in trace.spans
            ],
            "error": trace.error,
        }


class NoOpTracer(RAGTracer):
    """A tracer that records nothing, used when tracing is disabled."""

    def start_trace(self, query: str) -> RAGTrace:
        return RAGTrace(query=query)

    def start_span(self, trace: RAGTrace, name: str) -> TraceSpan:
        span = TraceSpan(name=name)
        return span

    def finish_trace(self, trace: RAGTrace) -> None:
        pass

    def get_recent_traces(self, limit: int = 50) -> list[RAGTrace]:
        return []
