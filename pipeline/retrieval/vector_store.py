"""
Vector store abstraction layer.
Swap between ChromaDB and Qdrant without changing retrieval code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pipeline.chunking.strategies import Chunk
from pipeline.embedding.models import EmbeddingModel


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: dict
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def delete_collection(self) -> None: ...


class ChromaStore(VectorStore):
    def __init__(self, collection_name: str = "rag_docs", persist_dir: str = "./data/chroma"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("pip install chromadb")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c.content for c in batch_chunks],
                metadatas=[{k: str(v) for k, v in c.metadata.items()} for c in batch_chunks],
            )
        print(f"[vector_store] ChromaDB: stored {len(chunks)} chunks")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            output.append(SearchResult(
                chunk_id=results["ids"][0][i],
                content=doc,
                metadata=meta,
                score=1 - dist,  # cosine distance → similarity
            ))
        return output

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection.name)


class QdrantStore(VectorStore):
    def __init__(self, collection_name: str = "rag_docs", url: str = "http://localhost:6333", dims: int = 1536):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("pip install qdrant-client")
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        from qdrant_client.models import Distance, VectorParams
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=abs(hash(c.chunk_id)) % (2**63),
                vector=emb,
                payload={"content": c.content, "chunk_id": c.chunk_id, **c.metadata},
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[vector_store] Qdrant: stored {len(chunks)} chunks")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )
        return [
            SearchResult(
                chunk_id=r.payload.get("chunk_id", str(r.id)),
                content=r.payload.get("content", ""),
                metadata={k: v for k, v in r.payload.items() if k not in ("content", "chunk_id")},
                score=r.score,
            )
            for r in results
        ]

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).points_count

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)


def get_vector_store(backend: str = "chroma", **kwargs) -> VectorStore:
    return {"chroma": ChromaStore, "qdrant": QdrantStore}[backend](**kwargs)
