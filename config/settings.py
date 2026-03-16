from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"

    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "research_papers"

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600

    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    chunk_size: int = 512
    chunk_overlap: int = 64

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    enable_tracing: bool = True

    max_context_tokens: int = 3000
    max_response_tokens: int = 1024

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
