"""FastAPI route handlers for the ResearchGPT API."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from models.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    PaperMetadata,
    PaperRecord,
    HealthResponse,
    SourceCitation,
    EvalResult,
    EvalSample,
)
from api.dependencies import get_app_state
from monitoring.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    state = get_app_state()
    return HealthResponse(
        status="healthy",
        components={
            "vector_store": {"count": state.vector_store.count},
            "cache": {"type": "redis" if state.cache._redis else "memory"},
        },
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question across the research paper corpus."""
    state = get_app_state()
    start = time.time()
    trace = state.tracer.start_trace(request.question)

    try:
        cache_key = _build_cache_key(request.question, request.filters, request.top_k)
        cached = state.cache.get(cache_key)
        if cached:
            logger.info("answer_served_from_cache", question=request.question[:80])
            return AskResponse(**cached)

        rewrite_span = state.tracer.start_span(trace, "query_rewrite")
        rewritten_query = await state.query_rewriter.rewrite(request.question)
        trace.rewritten_query = rewritten_query
        rewrite_span.finish()

        retrieval_span = state.tracer.start_span(trace, "retrieval")
        where = _build_filters(request.filters)
        retrieved = state.hybrid_retriever.retrieve(
            rewritten_query, top_k=request.top_k, where=where,
        )
        trace.retrieved_chunks = len(retrieved)
        retrieval_span.finish(chunk_count=len(retrieved))

        rerank_span = state.tracer.start_span(trace, "reranking")
        reranked = state.reranker.rerank(rewritten_query, retrieved)
        trace.reranked_chunks = len(reranked)
        rerank_span.finish(chunk_count=len(reranked))

        context_span = state.tracer.start_span(trace, "context_build")
        context_text, citations = state.context_builder.build(reranked)
        context_span.finish(token_count=len(context_text.split()))

        if request.stream:
            return _stream_response(
                request.question, rewritten_query, context_text, citations, start, trace,
            )

        gen_span = state.tracer.start_span(trace, "llm_generation")
        answer, usage = await state.llm_generator.generate(request.question, context_text)
        gen_span.finish(**usage)

        trace.prompt_tokens = usage.get("prompt_tokens", 0)
        trace.completion_tokens = usage.get("completion_tokens", 0)
        trace.total_tokens = usage.get("total_tokens", 0)
        trace.answer_length = len(answer)

        latency_ms = (time.time() - start) * 1000

        response = AskResponse(
            answer=answer,
            sources=citations,
            query=request.question,
            rewritten_query=rewritten_query,
            latency_ms=round(latency_ms, 2),
            token_usage=usage,
        )

        state.cache.set(cache_key, response.model_dump())
        state.tracer.finish_trace(trace)

        return response

    except Exception as exc:
        trace.error = str(exc)
        state.tracer.finish_trace(trace)
        logger.error("ask_failed", error=str(exc), question=request.question[:80])
        raise HTTPException(status_code=500, detail=str(exc))


def _stream_response(
    question: str,
    rewritten_query: str,
    context_text: str,
    citations: list[SourceCitation],
    start_time: float,
    trace=None,
):
    """Return a streaming response with server-sent events."""

    async def event_generator():
        state = get_app_state()
        yield _sse_event("sources", json.dumps([c.model_dump() for c in citations]))

        gen_span = state.tracer.start_span(trace, "llm_generation") if trace else None

        full_answer = []
        async for token in state.llm_generator.generate_stream(question, context_text):
            full_answer.append(token)
            yield _sse_event("token", token)

        if gen_span:
            gen_span.finish()

        if trace:
            trace.answer_length = sum(len(t) for t in full_answer)
            state.tracer.finish_trace(trace)

        latency = (time.time() - start_time) * 1000
        yield _sse_event("done", json.dumps({
            "latency_ms": round(latency, 2),
            "rewritten_query": rewritten_query,
        }))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@router.post("/ingest", response_model=IngestResponse)
async def ingest_paper(request: IngestRequest):
    """Ingest a research paper from text or file path."""
    state = get_app_state()

    try:
        if request.file_path:
            result = state.ingestion_pipeline.ingest_pdf(request.file_path, request.metadata)
        elif request.text:
            result = state.ingestion_pipeline.ingest_text(request.text, request.metadata)
        else:
            raise HTTPException(status_code=400, detail="Provide either file_path or text")

        state.bm25_retriever.add_chunks(
            [c for c in _get_chunks_for_paper(state, request.metadata.title)]
        )

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ingest_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    authors: str = Form(""),
    year: Optional[int] = Form(None),
    abstract: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    conference: Optional[str] = Form(None),
):
    """Upload a PDF file for ingestion."""
    import tempfile
    from pathlib import Path

    state = get_app_state()

    metadata = PaperMetadata(
        title=title,
        authors=[a.strip() for a in authors.split(",") if a.strip()],
        year=year,
        abstract=abstract,
        source_url=source_url,
        conference=conference,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = state.ingestion_pipeline.ingest_pdf(tmp_path, metadata)

        state.bm25_retriever.add_chunks(
            [c for c in _get_chunks_for_paper(state, metadata.title)]
        )

        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/papers", response_model=list[PaperRecord])
async def list_papers():
    """List all ingested papers."""
    state = get_app_state()
    papers = state.vector_store.get_all_papers()
    return [PaperRecord(**p) for p in papers]


@router.get("/papers/{paper_title:path}")
async def get_paper_detail(paper_title: str):
    """Get a paper's metadata and its text chunks grouped by section."""
    state = get_app_state()
    from ingestion.embedder import embed_query as _emb
    emb = _emb(paper_title)
    results = state.vector_store.query(
        emb, top_k=200, where={"paper_title": {"$eq": paper_title}},
    )
    if not results:
        raise HTTPException(status_code=404, detail="Paper not found")

    first = results[0].chunk
    sections: dict[str, list[str]] = {}
    for rc in results:
        sec = rc.chunk.section or "General"
        sections.setdefault(sec, []).append(rc.chunk.text)

    return {
        "title": first.paper_title,
        "authors": first.authors,
        "year": first.year,
        "conference": first.conference,
        "source_url": first.source_url,
        "chunk_count": len(results),
        "sections": {k: v for k, v in sections.items()},
    }


@router.get("/stats")
async def get_stats():
    """Aggregate monitoring stats for the admin dashboard."""
    state = get_app_state()
    traces = state.tracer.get_recent_traces(200)
    total = len(traces)
    if total == 0:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "total_tokens": 0,
            "avg_tokens": 0,
            "error_count": 0,
            "error_rate": 0,
            "avg_retrieved": 0,
            "avg_reranked": 0,
            "recent_queries": [],
        }

    latencies = [t.total_latency_ms for t in traces]
    tokens = [t.total_tokens for t in traces]
    errors = [t for t in traces if t.error]

    return {
        "total_queries": total,
        "avg_latency_ms": round(sum(latencies) / total, 1),
        "total_tokens": sum(tokens),
        "avg_tokens": round(sum(tokens) / total),
        "error_count": len(errors),
        "error_rate": round(len(errors) / total * 100, 1),
        "avg_retrieved": round(sum(t.retrieved_chunks for t in traces) / total, 1),
        "avg_reranked": round(sum(t.reranked_chunks for t in traces) / total, 1),
        "recent_queries": [
            {"query": t.query[:80], "latency_ms": round(t.total_latency_ms, 1), "timestamp": t.timestamp}
            for t in traces[-10:]
        ],
    }


@router.get("/traces")
async def get_traces(limit: int = 20):
    """Return recent RAG traces for monitoring."""
    state = get_app_state()
    traces = state.tracer.get_recent_traces(limit)
    return [state.tracer.export_trace(t) for t in traces]


@router.get("/prompts")
async def list_prompts():
    """List available prompt templates and versions."""
    state = get_app_state()
    return state.prompt_manager.list_prompts()


@router.post("/evaluate", response_model=EvalResult)
async def run_evaluation(samples: Optional[list[EvalSample]] = None):
    """Run the RAG evaluation pipeline and return metric results."""
    from evaluation.evaluator import RAGEvaluator
    from evaluation.dataset import EvalDatasetManager

    state = get_app_state()

    async def _ask_fn(question: str) -> tuple[str, list[str]]:
        """Adapter bridging the API's RAG pipeline to the evaluator's interface."""
        rewritten = await state.query_rewriter.rewrite(question)
        retrieved = state.hybrid_retriever.retrieve(rewritten)
        reranked = state.reranker.rerank(rewritten, retrieved)
        context_text, _ = state.context_builder.build(reranked)
        answer, _ = await state.llm_generator.generate(question, context_text)
        contexts = [rc.chunk.text for rc in reranked]
        return answer, contexts

    eval_samples = samples if samples else EvalDatasetManager().samples
    evaluator = RAGEvaluator(ask_fn=_ask_fn)
    result = await evaluator.evaluate(eval_samples)
    return result


def _build_cache_key(question: str, filters: dict | None, top_k: int | None) -> str:
    raw = json.dumps({"q": question, "f": filters, "k": top_k}, sort_keys=True)
    return f"ask:{hashlib.sha256(raw.encode()).hexdigest()[:32]}"


def _build_filters(filters: dict | None) -> dict | None:
    """Convert user-facing filters to ChromaDB where clauses."""
    if not filters:
        return None

    conditions = []
    if "year" in filters:
        conditions.append({"year": {"$eq": filters["year"]}})
    if "author" in filters:
        conditions.append({"authors": {"$contains": filters["author"]}})
    if "conference" in filters:
        conditions.append({"conference": {"$eq": filters["conference"]}})
    if "title" in filters:
        conditions.append({"paper_title": {"$contains": filters["title"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _get_chunks_for_paper(state, paper_title: str):
    """Retrieve all chunks for a given paper from the vector store."""
    from ingestion.embedder import embed_query
    dummy_emb = embed_query(paper_title)
    results = state.vector_store.query(
        dummy_emb, top_k=200, where={"paper_title": {"$eq": paper_title}},
    )
    return [r.chunk for r in results]
