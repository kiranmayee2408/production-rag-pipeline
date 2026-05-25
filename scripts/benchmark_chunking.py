"""
Benchmark all three chunking strategies on a sample corpus.
Compares: chunk count, avg chunk size, and retrieval quality (RAGAS scores).

Run: python scripts/benchmark_chunking.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.ingestion.loaders import DirectoryLoader
from pipeline.chunking.strategies import RecursiveChunker, SlidingWindowChunker
from pipeline.embedding.models import get_embedding_model
from pipeline.retrieval.vector_store import ChromaStore
from pipeline.evaluation.ragas_eval import RAGEvaluator, EvalSample

SAMPLE_QUESTIONS = [
    EvalSample(
        question="What is the AC Advisor project?",
        answer="",
        contexts=[],
        ground_truth="AC Advisor predicts automobile air conditioning energy usage.",
    ),
    EvalSample(
        question="What ML models were benchmarked?",
        answer="",
        contexts=[],
        ground_truth="CatBoost and Transformer-based architectures were benchmarked.",
    ),
]


def benchmark():
    loader = DirectoryLoader()
    docs = loader.load("./data/sample_docs")
    if not docs:
        print("No sample docs found. Creating synthetic docs...")
        from pipeline.ingestion.loaders import Document
        docs = [Document(
            content="The AC Advisor project benchmarks CatBoost and Transformer models on OBD-II data. "
                    "It achieves MAPE under 10% for air conditioning energy prediction. "
                    "The system is deployed as a FastAPI service with SHAP explainability. " * 20,
            metadata={"source": "synthetic"},
        )]

    embed_model = get_embedding_model("mock")
    evaluator = RAGEvaluator(use_llm_judge=False)

    strategies = [
        ("recursive", RecursiveChunker(chunk_size=512, overlap=64)),
        ("sliding_window", SlidingWindowChunker(chunk_size=512, stride=256)),
    ]

    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY BENCHMARK")
    print("=" * 60)

    for name, chunker in strategies:
        chunks = chunker.chunk(docs)
        avg_size = sum(len(c.content.split()) for c in chunks) / len(chunks) if chunks else 0

        # Build a tiny index and search
        store = ChromaStore(collection_name=f"bench_{name}", persist_dir=f"/tmp/bench_{name}")
        embeddings = embed_model.embed([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)

        # Evaluate retrieval quality
        results_scores = []
        for sample in SAMPLE_QUESTIONS:
            q_emb = embed_model.embed_one(sample.question)
            hits = store.search(q_emb, top_k=3)
            contexts = [h.content for h in hits]
            scored_sample = EvalSample(
                question=sample.question, answer=contexts[0] if contexts else "",
                contexts=contexts, ground_truth=sample.ground_truth,
            )
            cp = evaluator.context_precision(scored_sample)
            cr = evaluator.context_recall(scored_sample)
            results_scores.append((cp + cr) / 2)
            store.delete_collection()

        print(f"\nStrategy: {name}")
        print(f"  Chunks produced : {len(chunks)}")
        print(f"  Avg chunk size  : {avg_size:.1f} words")
        print(f"  Avg retrieval   : {sum(results_scores)/len(results_scores):.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark()
