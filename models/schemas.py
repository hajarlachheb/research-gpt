from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    source_url: Optional[str] = None
    conference: Optional[str] = None


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    paper_title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    section: Optional[str] = None
    source_url: Optional[str] = None
    conference: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float


# ── API schemas ──────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    file_path: Optional[str] = None
    text: Optional[str] = None
    metadata: PaperMetadata


class IngestResponse(BaseModel):
    paper_title: str
    chunks_created: int
    message: str = "Ingestion successful"


class AskRequest(BaseModel):
    question: str
    filters: Optional[dict] = None
    top_k: Optional[int] = None
    stream: bool = False


class SourceCitation(BaseModel):
    paper: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    section: Optional[str] = None
    source_url: Optional[str] = None
    relevance_score: float = 0.0


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    query: str
    rewritten_query: Optional[str] = None
    latency_ms: float = 0.0
    token_usage: Optional[dict] = None


class PaperRecord(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    source_url: Optional[str] = None
    conference: Optional[str] = None
    chunk_count: int = 0


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict = Field(default_factory=dict)


# ── Evaluation schemas ───────────────────────────────────────────────


class EvalSample(BaseModel):
    question: str
    ground_truth: str
    relevant_papers: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevancy: Optional[float] = None
    num_samples: int = 0
    details: list[dict] = Field(default_factory=list)
