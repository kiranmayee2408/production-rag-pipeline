# Production RAG Pipeline with Evaluation

An end-to-end **Retrieval-Augmented Generation (RAG) pipeline** built for production — with multiple chunking strategies, pluggable vector stores, embedding model comparison, and built-in RAGAS-style evaluation.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-purple)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What Makes This Different

Most RAG tutorials stop at "load PDF → embed → ask questions." This pipeline goes further:

- **3 chunking strategies** with quantitative benchmarking (recursive, sliding window, semantic)
- **2 vector backends** (ChromaDB local, Qdrant distributed) — swap with one line
- **2 embedding providers** (OpenAI API, HuggingFace local) — compare quality vs cost
- **RAGAS-style evaluation** built-in on every query: faithfulness, relevancy, context precision, context recall
- **Production API** with document upload, batch evaluation, and index stats

---

## Architecture

```
Documents (PDF, TXT, MD, JSON, Web)
         │
         ▼
┌─────────────────┐
│   Ingestion     │  DirectoryLoader, PDFLoader, WebLoader
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  RecursiveChunker | SlidingWindowChunker | SemanticChunker
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │  OpenAI text-embedding-3-small | HuggingFace all-MiniLM-L6-v2
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  ChromaDB (local) | Qdrant (distributed)
└────────┬────────┘
         │  query embedding
         ▼
┌─────────────────┐
│   Retrieval     │  top-k cosine similarity search
└────────┬────────┘
         │ contexts
         ▼
┌─────────────────┐
│   Generation    │  LLM (GPT-4o-mini or any LLM via function)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │  Faithfulness | Answer Relevancy | Context Precision | Context Recall
└─────────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/kiranmayee2408/production-rag-pipeline.git
cd production-rag-pipeline
pip install -r requirements.txt

# Set your OpenAI key (optional — mock embeddings work without it)
export OPENAI_API_KEY=sk-...

# Start the API
uvicorn api.main:app --reload
```

API docs: http://localhost:8000/docs

---

## Usage

### Index documents

```bash
# Index a directory
curl -X POST "http://localhost:8000/index/upload" \
  -F "file=@my_document.pdf"

# Check index size
curl "http://localhost:8000/index/stats"
```

### Query with evaluation

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the AC Advisor project?", "top_k": 5, "evaluate": true}'
```

Response:
```json
{
  "question": "What is the AC Advisor project?",
  "answer": "AC Advisor is an ML system that predicts automobile A/C energy usage...",
  "contexts": ["[retrieved chunk 1]", "[retrieved chunk 2]"],
  "sources": [{"chunk_id": "abc123", "score": 0.91, "source": "report.pdf"}],
  "latency_ms": 342.1,
  "eval_scores": {
    "faithfulness": 0.87,
    "answer_relevancy": 0.82,
    "context_precision": 0.80
  }
}
```

### Python API

```python
from pipeline.rag_pipeline import RAGPipeline

rag = RAGPipeline.from_config(embedding_provider="openai", vector_backend="chroma")
rag.index_directory("./my_docs")

result = rag.query("What are the key findings?")
print(result.answer)
print(result.eval_scores)  # {'faithfulness': 0.89, 'answer_relevancy': 0.84, ...}
```

### Benchmark chunking strategies

```bash
python scripts/benchmark_chunking.py
```

Output:
```
Strategy: recursive
  Chunks produced : 47
  Avg chunk size  : 498.2 words
  Avg retrieval   : 0.812

Strategy: sliding_window
  Chunks produced : 63
  Avg chunk size  : 512.0 words
  Avg retrieval   : 0.774
```

---

## Evaluation Metrics

| Metric | What it measures | Score range |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | 0–1 ↑ |
| **Answer Relevancy** | Does the answer actually address the question? | 0–1 ↑ |
| **Context Precision** | Are retrieved chunks relevant to the question? | 0–1 ↑ |
| **Context Recall** | Does the context contain the needed information? | 0–1 ↑ |

---

## Chunking Strategy Guide

| Strategy | Best for | Chunk consistency | Semantic quality |
|---|---|---|---|
| `recursive` | General documents, mixed content | High | Medium |
| `sliding_window` | Dense technical text, code | Very high | Low |
| `semantic` | Long narratives, research papers | Low | High |

---

## Embedding Model Comparison

| Model | Provider | Dims | Cost | Quality |
|---|---|---|---|---|
| `text-embedding-3-small` | OpenAI API | 1536 | $0.02/1M tokens | ★★★★☆ |
| `text-embedding-3-large` | OpenAI API | 3072 | $0.13/1M tokens | ★★★★★ |
| `all-MiniLM-L6-v2` | HuggingFace local | 384 | Free | ★★★☆☆ |
| `all-mpnet-base-v2` | HuggingFace local | 768 | Free | ★★★★☆ |

---

## Project Structure

```
production-rag/
├── pipeline/
│   ├── ingestion/loaders.py        # PDF, TXT, MD, JSON, Web loaders
│   ├── chunking/strategies.py      # Recursive, SlidingWindow, Semantic chunkers
│   ├── embedding/models.py         # OpenAI, HuggingFace, Mock embeddings
│   ├── retrieval/vector_store.py   # ChromaDB + Qdrant abstractions
│   ├── evaluation/ragas_eval.py    # Faithfulness, Relevancy, Precision, Recall
│   └── rag_pipeline.py             # Main orchestrator
├── api/main.py                     # FastAPI endpoints
├── scripts/
│   └── benchmark_chunking.py       # Strategy comparison benchmark
├── data/sample_docs/               # Drop your documents here
└── requirements.txt
```

---

## Related Work

- [LLM Observability Dashboard](https://github.com/kiranmayee2408/llm-observability-dashboard) — Monitor RAG quality in production
- [AC Advisor (IEEE)](https://github.com/kiranmayee2408/Ac-advisor-ieee)

---

## License

MIT
