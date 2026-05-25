"""
Embedding service supporting multiple providers.
Designed for easy A/B comparison between embedding models.

Supported:
  - OpenAI  (text-embedding-3-small, text-embedding-3-large, ada-002)
  - HuggingFace sentence-transformers (local, free)
  - Mock (for tests)
"""
from abc import ABC, abstractmethod
from typing import Protocol
import os


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbeddings(EmbeddingModel):
    """
    OpenAI embedding models.
    text-embedding-3-small: 1536 dims, $0.02/1M tokens — good default
    text-embedding-3-large: 3072 dims, $0.13/1M tokens — higher quality
    """
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("pip install openai")
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings


class HuggingFaceEmbeddings(EmbeddingModel):
    """
    Local sentence-transformers — no API key needed.
    Good models: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (quality)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("pip install sentence-transformers")
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return embeddings.tolist()


class MockEmbeddings(EmbeddingModel):
    """Deterministic mock for tests — no API calls."""
    def __init__(self, dims: int = 128):
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        import math
        result = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = [math.sin(h * (i + 1) * 0.001) for i in range(self.dims)]
            norm = sum(x**2 for x in vec) ** 0.5
            result.append([x / norm for x in vec])
        return result


def get_embedding_model(provider: str = "openai", **kwargs) -> EmbeddingModel:
    return {
        "openai": OpenAIEmbeddings,
        "huggingface": HuggingFaceEmbeddings,
        "mock": MockEmbeddings,
    }[provider](**kwargs)
