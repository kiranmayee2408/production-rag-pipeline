"""
Chunking strategies — the single most impactful RAG hyperparameter.

Three strategies implemented:
1. RecursiveChunker  — split on paragraph > sentence > word boundaries
2. SlidingWindowChunker — fixed-size windows with overlap (simple, reliable)
3. SemanticChunker   — merge sentences until cosine similarity drops (quality-aware)

Usage:
    chunker = RecursiveChunker(chunk_size=512, overlap=64)
    chunks = chunker.chunk(documents)
"""
import re
from dataclasses import dataclass, field
from typing import Protocol
from pipeline.ingestion.loaders import Document


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class Chunker(Protocol):
    def chunk(self, documents: list[Document]) -> list[Chunk]: ...


class RecursiveChunker:
    """
    Split text hierarchically: paragraphs → sentences → words.
    Produces semantically coherent chunks that respect natural boundaries.
    """
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split(self, text: str, separators: list[str]) -> list[str]:
        sep = separators[0]
        remaining = separators[1:]

        if not sep:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]

        splits = text.split(sep)
        chunks, current = [], ""
        for s in splits:
            candidate = current + (sep if current else "") + s
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(s) > self.chunk_size and remaining:
                    chunks.extend(self._split(s, remaining))
                else:
                    current = s
        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            splits = self._split(doc.content, self.SEPARATORS)
            # Add overlap by prepending the tail of the previous chunk
            overlapped = []
            for i, split in enumerate(splits):
                if i > 0 and self.overlap > 0:
                    prev_tail = splits[i - 1][-self.overlap:]
                    split = prev_tail + " " + split
                overlapped.append(split)

            for i, content in enumerate(overlapped):
                all_chunks.append(Chunk(
                    content=content.strip(),
                    metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(overlapped)},
                    doc_id=doc.doc_id,
                ))
        print(f"[chunking] RecursiveChunker → {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


class SlidingWindowChunker:
    """
    Fixed-size sliding window over tokens (approximated by words).
    Simple and predictable — good baseline.
    """
    def __init__(self, chunk_size: int = 512, stride: int = 256):
        self.chunk_size = chunk_size
        self.stride = stride

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            words = doc.content.split()
            i = 0
            idx = 0
            while i < len(words):
                window = words[i:i + self.chunk_size]
                content = " ".join(window)
                all_chunks.append(Chunk(
                    content=content,
                    metadata={**doc.metadata, "chunk_index": idx, "window_start": i},
                    doc_id=doc.doc_id,
                ))
                i += self.stride
                idx += 1
        print(f"[chunking] SlidingWindowChunker → {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


class SemanticChunker:
    """
    Group sentences into chunks based on embedding similarity.
    When consecutive sentences diverge semantically, start a new chunk.
    Requires an embedding function: (list[str]) -> list[list[float]]
    """
    def __init__(self, embed_fn, threshold: float = 0.85, max_chunk_size: int = 800):
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            sentences = re.split(r"(?<=[.!?])\s+", doc.content)
            sentences = [s for s in sentences if s.strip()]
            if len(sentences) <= 1:
                all_chunks.append(Chunk(content=doc.content, metadata=doc.metadata, doc_id=doc.doc_id))
                continue

            embeddings = self.embed_fn(sentences)
            groups, current_group = [], [sentences[0]]

            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                current_text = " ".join(current_group + [sentences[i]])
                if sim >= self.threshold and len(current_text) <= self.max_chunk_size:
                    current_group.append(sentences[i])
                else:
                    groups.append(current_group)
                    current_group = [sentences[i]]
            if current_group:
                groups.append(current_group)

            for i, group in enumerate(groups):
                all_chunks.append(Chunk(
                    content=" ".join(group),
                    metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(groups)},
                    doc_id=doc.doc_id,
                ))

        print(f"[chunking] SemanticChunker → {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


def get_chunker(strategy: str, **kwargs) -> Chunker:
    return {
        "recursive": RecursiveChunker,
        "sliding_window": SlidingWindowChunker,
        "semantic": SemanticChunker,
    }[strategy](**kwargs)
