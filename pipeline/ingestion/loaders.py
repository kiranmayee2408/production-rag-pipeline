"""
Document ingestion pipeline.
Supports PDF, TXT, Markdown, JSON, and web URLs.
Each loader returns a list of Document objects for the chunking stage.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import httpx


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            import hashlib
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class TextLoader:
    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return [Document(content=text, metadata={"source": str(path), "type": "text"})]


class MarkdownLoader:
    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        # Split by H2 headers to create section-level documents
        sections = re.split(r"\n## ", text)
        docs = []
        for i, section in enumerate(sections):
            if section.strip():
                docs.append(Document(
                    content=section.strip(),
                    metadata={"source": str(path), "type": "markdown", "section": i},
                ))
        return docs if docs else [Document(content=text, metadata={"source": str(path), "type": "markdown"})]


class JSONLoader:
    def __init__(self, content_key: str = "text", metadata_keys: list[str] | None = None):
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        data = json.loads(path.read_text())
        if isinstance(data, list):
            docs = []
            for item in data:
                content = item.get(self.content_key, str(item))
                meta = {k: item.get(k) for k in self.metadata_keys if k in item}
                meta["source"] = str(path)
                docs.append(Document(content=content, metadata=meta))
            return docs
        content = data.get(self.content_key, json.dumps(data))
        return [Document(content=content, metadata={"source": str(path), "type": "json"})]


class PDFLoader:
    def load(self, path: str | Path) -> list[Document]:
        try:
            import pypdf
            path = Path(path)
            reader = pypdf.PdfReader(str(path))
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    docs.append(Document(
                        content=text,
                        metadata={"source": str(path), "type": "pdf", "page": i + 1},
                    ))
            return docs
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")


class WebLoader:
    def load(self, url: str) -> list[Document]:
        response = httpx.get(url, timeout=15, follow_redirects=True)
        response.raise_for_status()
        # Strip HTML tags for clean text
        text = re.sub(r"<[^>]+>", " ", response.text)
        text = re.sub(r"\s+", " ", text).strip()
        return [Document(content=text, metadata={"source": url, "type": "web"})]


class DirectoryLoader:
    LOADERS = {
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".json": JSONLoader,
        ".pdf": PDFLoader,
    }

    def load(self, directory: str | Path, glob: str = "**/*") -> list[Document]:
        directory = Path(directory)
        docs = []
        for path in directory.glob(glob):
            if path.is_file() and path.suffix in self.LOADERS:
                loader = self.LOADERS[path.suffix]()
                try:
                    docs.extend(loader.load(path))
                except Exception as e:
                    print(f"[ingestion] Warning: failed to load {path}: {e}")
        print(f"[ingestion] Loaded {len(docs)} documents from {directory}")
        return docs
