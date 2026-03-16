# ResearchGPT — AI Research Paper Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that enables users to ask complex questions across thousands of research papers, returning grounded answers with citations.

## Architecture

```
                        ┌────────────────────┐
                        │   User / Client    │
                        └────────┬───────────┘
                                 │ POST /ask
                                 ▼
                        ┌────────────────────┐
                        │  FastAPI Server    │◄──── GET /health, /papers, /traces
                        │  (api/main.py)     │      POST /ingest, /evaluate
                        └────────┬───────────┘
                                 │
          ┌──────────────────────┤
          ▼                      ▼
 ┌─────────────────┐   ┌─────────────────┐
 │  Redis Cache     │   │  Query Rewriter │ (LLM-based expansion)
 │  (cache/)        │   │  (llm/)         │
 └─────────────────┘   └────────┬────────┘
                                 │ rewritten query
                                 ▼
                        ┌─────────────────────────┐
                        │    Hybrid Retriever      │
                        │    (retrieval/)           │
                        ├─────────────┬────────────┤
                        ▼             ▼            │
               ┌──────────────┐ ┌──────────┐      │
               │ Vector Store │ │  BM25    │      │
               │ (ChromaDB)   │ │ (in-mem) │      │
               └──────────────┘ └──────────┘      │
                        │  Reciprocal Rank Fusion  │
                        └────────────┬─────────────┘
                                     ▼
                        ┌────────────────────┐
                        │ Cross-Encoder      │ (reranker/)
                        │ Reranker           │
                        └────────┬───────────┘
                                 │ top-K chunks
                                 ▼
                        ┌────────────────────┐
                        │ Context Builder    │ (token-budget aware)
                        │ (context_builder/) │
                        └────────┬───────────┘
                                 │ formatted context + citations
                                 ▼
                        ┌────────────────────┐
                        │ LLM Generator      │ (OpenAI / streaming)
                        │ (llm/)             │
                        └────────┬───────────┘
                                 │
                                 ▼
                        ┌────────────────────┐
                        │ Answer + Citations │
                        └────────────────────┘

        ┌──────────────────────────────────────────────┐
        │           Cross-cutting concerns              │
        ├──────────────┬───────────────┬───────────────┤
        │ Monitoring   │  Evaluation   │  Prompts      │
        │ (structlog   │  (Ragas       │  (versioned   │
        │  + tracer)   │   metrics)    │   templates)  │
        └──────────────┴───────────────┴───────────────┘
```

## Key Features

- **Hybrid retrieval** — combines dense vector search with BM25 keyword matching via Reciprocal Rank Fusion
- **Cross-encoder reranking** — refines initial retrieval with a fine-tuned reranker model
- **Citation grounding** — every answer references specific papers, sections, and years
- **Response streaming** — SSE-based token streaming for real-time answers
- **Query rewriting** — LLM-powered query expansion for better retrieval
- **Full observability** — structured logging, per-query tracing with span-level timing
- **Caching** — Redis with in-memory fallback for embeddings and responses
- **Prompt versioning** — file-based template management with version switching
- **Automated evaluation** — Ragas-based metrics (faithfulness, relevance, precision, recall)
- **PDF ingestion** — section-aware extraction, semantic chunking with overlap

## Project Structure

```
research-gpt/
├── api/                  # FastAPI routes and app setup
├── cache/                # Redis + in-memory caching
├── config/               # Settings and environment config
├── context_builder/      # Token-aware prompt context assembly
├── evaluation/           # Ragas evaluation pipeline
├── ingestion/            # PDF extraction, cleaning, chunking, embedding
├── llm/                  # Answer generation and query rewriting
├── models/               # Pydantic data schemas
├── monitoring/           # Structured logging and RAG tracing
├── prompts/              # Versioned prompt templates
├── reranker/             # Cross-encoder reranking
├── retrieval/            # Vector store, BM25, hybrid retriever
├── tests/                # Test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 2. Run with Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### 3. Run Locally

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## API Endpoints

### `POST /ask`

Ask a question across the paper corpus.

```json
{
  "question": "Compare BERT and RoBERTa training strategies",
  "filters": {"year": 2019},
  "stream": false
}
```

Response:

```json
{
  "answer": "According to RoBERTa (2019), the key improvements over BERT include...",
  "sources": [
    {
      "paper": "BERT: Pre-training of Deep Bidirectional Transformers",
      "year": 2018,
      "section": "Training"
    },
    {
      "paper": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
      "year": 2019,
      "section": "Methodology"
    }
  ],
  "query": "Compare BERT and RoBERTa training strategies",
  "rewritten_query": "Comparison of BERT and RoBERTa pretraining methodologies...",
  "latency_ms": 2340.5,
  "token_usage": {"prompt_tokens": 850, "completion_tokens": 200, "total_tokens": 1050}
}
```

### `POST /ingest`

Ingest a research paper from text or file path.

```json
{
  "text": "Full paper text here...",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Vaswani", "Shazeer"],
    "year": 2017,
    "source_url": "https://arxiv.org/abs/1706.03762"
  }
}
```

### `POST /ingest/upload`

Upload a PDF file for ingestion (multipart form).

### `GET /papers`

List all ingested papers with chunk counts.

### `GET /health`

Health check with component status.

### `GET /traces`

Recent RAG traces for monitoring.

### `GET /prompts`

List prompt templates and versions.

## Retrieval Pipeline

1. **Query rewriting** — LLM expands the user query for better recall
2. **Hybrid search** — parallel vector (cosine) and BM25 (keyword) retrieval
3. **Reciprocal Rank Fusion** — weighted fusion of both result sets
4. **Cross-encoder reranking** — top-K candidates re-scored with a cross-encoder
5. **Context building** — token-budget-aware assembly of passages with metadata

## Configuration

All configuration is managed via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | LLM model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RETRIEVAL_TOP_K` | `20` | Initial retrieval count |
| `RERANK_TOP_K` | `5` | Chunks kept after reranking |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap words between chunks |
| `BM25_WEIGHT` | `0.3` | BM25 weight in hybrid fusion |
| `VECTOR_WEIGHT` | `0.7` | Vector weight in hybrid fusion |

## Design Decisions

1. **ChromaDB** chosen for easy local development; swap for Pinecone/Weaviate in production at scale.
2. **Reciprocal Rank Fusion** over linear score combination — more robust to score distribution differences.
3. **Cross-encoder reranker** dramatically improves precision over bi-encoder alone.
4. **Token-budget context builder** prevents prompt overflow and prioritizes highest-scored chunks.
5. **In-memory BM25** kept separate from vector store for flexibility; can be replaced with Elasticsearch.
6. **Two-tier caching** (Redis + memory) ensures responsiveness even without Redis.

## Evaluation

The evaluation pipeline uses Ragas metrics:

| Metric | Description |
|---|---|
| Faithfulness | Is the answer grounded in the context? |
| Context Precision | Are the retrieved chunks relevant? |
| Context Recall | Were all necessary chunks retrieved? |
| Answer Relevancy | Does the answer address the question? |

Run evaluation via CLI:

```bash
python -m evaluation.run_eval
python -m evaluation.run_eval --dataset custom_data.json --output report.json
```

Or via the API:

```bash
curl -X POST http://localhost:8000/evaluate
```

## Limitations

- BM25 index is in-memory (rebuilt from ChromaDB on every restart)
- Single-node deployment; horizontal scaling would require shared state
- PDF section detection relies on heuristic heading patterns
- Evaluation requires OpenAI API access for Ragas metrics
- No user authentication (add middleware for production)
- Semantic chunking adds ingestion latency due to per-sentence embedding

## Running Tests

```bash
pytest
```
