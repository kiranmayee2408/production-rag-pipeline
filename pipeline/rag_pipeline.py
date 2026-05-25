"""
Production RAG Pipeline — main orchestrator.

Ties together ingestion → chunking → embedding → indexing → retrieval → generation → evaluation.

Usage:
    from pipeline.rag_pipeline import RAGPipeline

    rag = RAGPipeline.from_config()
    rag.index_directory("./data/docs")

    result = rag.query("What is the AC Advisor project?")
    print(result["answer"])
    print(result["contexts"])
    print(result["eval_scores"])
"""
import os
import time
import json
from dataclasses import dataclass
from typing import Optional

from pipeline.ingestion.loaders import DirectoryLoader, Document
from pipeline.chunking.strategies import get_chunker, Chunk
from pipeline.embedding.models import get_embedding_model, EmbeddingModel
from pipeline.retrieval.vector_store import get_vector_store, VectorStore, SearchResult
from pipeline.evaluation.ragas_eval import RAGEvaluator, EvalSample


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[str]
    source_chunks: list[SearchResult]
    latency_ms: float
    eval_scores: dict


class RAGPipeline:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        llm_fn,
        chunking_strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        evaluate_responses: bool = True,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_fn = llm_fn
        self.chunker = get_chunker(chunking_strategy, chunk_size=chunk_size, overlap=chunk_overlap)
        self.top_k = top_k
        self.evaluator = RAGEvaluator(llm_fn=llm_fn, use_llm_judge=False)
        self.evaluate_responses = evaluate_responses

    @classmethod
    def from_config(cls, embedding_provider: str = "mock", vector_backend: str = "chroma"):
        """Factory that creates a pipeline from environment configuration."""
        embed_model = get_embedding_model(embedding_provider)
        store = get_vector_store(
            backend=vector_backend,
            collection_name="rag_docs",
        )

        def llm_fn(prompt: str) -> str:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "[LLM not configured — set OPENAI_API_KEY]"
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content

        return cls(embedding_model=embed_model, vector_store=store, llm_fn=llm_fn)

    def index_documents(self, documents: list[Document]) -> int:
        """Chunk and embed documents, then store in vector index."""
        chunks = self.chunker.chunk(documents)
        print(f"[pipeline] Embedding {len(chunks)} chunks...")
        embeddings = self.embedding_model.embed([c.content for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)
        print(f"[pipeline] Indexed {len(chunks)} chunks. Total in store: {self.vector_store.count()}")
        return len(chunks)

    def index_directory(self, directory: str) -> int:
        loader = DirectoryLoader()
        documents = loader.load(directory)
        return self.index_documents(documents)

    def retrieve(self, question: str) -> list[SearchResult]:
        query_embedding = self.embedding_model.embed_one(question)
        return self.vector_store.search(query_embedding, top_k=self.top_k)

    def _build_prompt(self, question: str, contexts: list[str]) -> str:
        ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        return f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.

Context:
{ctx_text}

Question: {question}

Answer:"""

    def query(self, question: str) -> RAGResult:
        t0 = time.time()

        # Retrieve
        results = self.retrieve(question)
        contexts = [r.content for r in results]

        # Generate
        prompt = self._build_prompt(question, contexts)
        answer = self.llm_fn(prompt)

        latency_ms = round((time.time() - t0) * 1000, 1)

        # Evaluate
        eval_scores = {}
        if self.evaluate_responses:
            sample = EvalSample(question=question, answer=answer, contexts=contexts)
            eval_scores = {
                "faithfulness": self.evaluator.faithfulness(sample),
                "answer_relevancy": self.evaluator.answer_relevancy(sample),
                "context_precision": self.evaluator.context_precision(sample),
            }

        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            source_chunks=results,
            latency_ms=latency_ms,
            eval_scores=eval_scores,
        )

    def batch_evaluate(self, samples: list[EvalSample]) -> dict:
        """Run full RAGAS evaluation on a test dataset."""
        results = self.evaluator.evaluate(samples)
        return self.evaluator.report(results)
