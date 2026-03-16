"""Shared application state and dependency injection for FastAPI."""

from __future__ import annotations

from functools import lru_cache

from retrieval.vector_store import VectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranker.cross_encoder import CrossEncoderReranker
from context_builder.builder import ContextBuilder
from llm.generator import LLMGenerator
from llm.query_rewriter import QueryRewriter
from ingestion.pipeline import IngestionPipeline
from prompts.manager import PromptManager
from monitoring.tracer import RAGTracer, NoOpTracer
from config.settings import get_settings as _get_settings
from cache.redis_cache import CacheManager
from monitoring.logger import get_logger

logger = get_logger(__name__)


class AppState:
    """Holds all singleton service instances."""

    def __init__(self):
        self.prompt_manager = PromptManager()
        self.vector_store = VectorStore()
        self.bm25_retriever = BM25Retriever()
        self._rebuild_bm25_index()
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25_retriever)
        self.reranker = CrossEncoderReranker()
        self.context_builder = ContextBuilder()
        self.llm_generator = LLMGenerator(self.prompt_manager)
        self.query_rewriter = QueryRewriter(self.prompt_manager)
        self.ingestion_pipeline = IngestionPipeline(self.vector_store)
        settings = _get_settings()
        self.tracer = RAGTracer() if settings.enable_tracing else NoOpTracer()
        self.cache = CacheManager()

    def _rebuild_bm25_index(self) -> None:
        """Populate BM25 from persisted ChromaDB data so keyword search works after restart."""
        chunks = self.vector_store.get_all_chunks()
        if chunks:
            self.bm25_retriever.add_chunks(chunks)
            logger.info("bm25_rebuilt_from_vectorstore", chunk_count=len(chunks))


_state: AppState | None = None


def get_app_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
