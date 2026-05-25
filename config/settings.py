from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Vector store
    vector_store: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_dir: str = "./data/chroma"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "rag_documents"

    # Chunking
    chunk_strategy: Literal["recursive", "semantic", "sliding_window"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_threshold: float = 0.85

    # Retrieval
    retrieval_top_k: int = 5
    rerank_top_n: int = 3
    use_reranker: bool = False

    # Evaluation
    eval_sample_size: int = 50

    class Config:
        env_file = ".env"

settings = Settings()
