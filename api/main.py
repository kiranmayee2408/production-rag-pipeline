from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import tempfile, os, shutil

from pipeline.rag_pipeline import RAGPipeline
from pipeline.evaluation.ragas_eval import EvalSample

app = FastAPI(title="Production RAG Pipeline API", version="1.0.0")
pipeline: RAGPipeline | None = None


@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = RAGPipeline.from_config(
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "mock"),
        vector_backend=os.getenv("VECTOR_BACKEND", "chroma"),
    )
    # Index sample docs on startup if any exist
    sample_dir = "./data/sample_docs"
    if os.path.exists(sample_dir) and os.listdir(sample_dir):
        pipeline.index_directory(sample_dir)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    evaluate: Optional[bool] = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    sources: list[dict]
    latency_ms: float
    eval_scores: dict


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    pipeline.evaluate_responses = req.evaluate
    pipeline.top_k = req.top_k
    result = pipeline.query(req.question)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        contexts=result.contexts,
        sources=[{"chunk_id": r.chunk_id, "score": round(r.score, 3),
                  "source": r.metadata.get("source", "")} for r in result.source_chunks],
        latency_ms=result.latency_ms,
        eval_scores=result.eval_scores,
    )


@app.post("/index/upload")
async def upload_and_index(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        from pipeline.ingestion.loaders import TextLoader, PDFLoader, MarkdownLoader
        loaders = {".txt": TextLoader, ".md": MarkdownLoader, ".pdf": PDFLoader}
        loader = loaders.get(suffix, TextLoader)()
        docs = loader.load(tmp_path)
        count = pipeline.index_documents(docs)
        return {"indexed_chunks": count, "filename": file.filename}
    finally:
        os.unlink(tmp_path)


@app.get("/index/stats")
async def index_stats():
    return {"total_chunks": pipeline.vector_store.count()}


@app.post("/evaluate")
async def evaluate_pipeline(samples: list[dict]):
    eval_samples = [EvalSample(**s) for s in samples]
    report = pipeline.batch_evaluate(eval_samples)
    return report


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": pipeline.vector_store.count() if pipeline else 0}
