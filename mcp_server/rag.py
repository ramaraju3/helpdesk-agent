"""RAG retrieval over the runbooks in data/docs/ (fictional company docs).

The pipeline, one function per stage:

    load_docs -> chunk -> embed -> index (in memory) -> search(query) -> top-k chunks

"Generation" (the G in RAG) happens in the agent: the chunks go back to the model
as a tool result and the reply is drafted from them.

Embeddings run locally with fastembed (BAAI/bge-small-en-v1.5, ONNX). They're free,
need no key, and give the same vectors every run. The model (~70 MB) downloads on
first use; set FASTEMBED_CACHE_PATH to choose where it is cached.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

DOCS_DIR = Path(__file__).parent / "data" / "docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Chunk:
    source: str  # file name, e.g. "vpn-and-remote-access.md"
    title: str  # the document's "# " heading
    section: str  # the "## " heading this chunk came from
    text: str  # the section body

    def for_embedding(self) -> str:
        # Prefix the body with its titles so a section like "Gateway outages" still
        # carries the context "this is about the VPN" (a contextual chunk header).
        return f"{self.title} - {self.section}\n{self.text}"


# ---------- 1. Load + 2. Chunk ----------


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    """Split a document into one chunk per '## ' section.

    Our docs are short and well structured, so sections are natural chunks. For long,
    unstructured text you'd use fixed-size windows with some overlap instead.
    """
    title, chunks = "", []
    section, body = None, []

    def flush():
        if section is not None and body:
            chunks.append(Chunk(source, title, section, " ".join(body).strip()))

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush()
            section, body = line[3:].strip(), []
        elif line.startswith("# "):
            title = line[2:].strip()
        elif line.strip():
            body.append(line.strip())
    flush()
    return chunks


def load_chunks() -> list[Chunk]:
    return [
        chunk
        for path in sorted(DOCS_DIR.glob("*.md"))
        for chunk in chunk_markdown(path.name, path.read_text())
    ]


# ---------- 3. Embed + 4. Index ----------


@cache  # load the model once per process
def embedder() -> TextEmbedding:
    return TextEmbedding(EMBEDDING_MODEL)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector to length 1, so a dot product equals cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


@cache  # build the index once per process (~30 chunks, takes about a second)
def index() -> tuple[list[Chunk], np.ndarray]:
    chunks = load_chunks()
    # passage_embed vs query_embed: bge models were trained with a special prefix on
    # queries, and fastembed adds it for us. Documents get no prefix.
    vectors = np.array(list(embedder().passage_embed([c.for_embedding() for c in chunks])))
    return chunks, normalize(vectors)


# ---------- 5. Search ----------


def search(query: str, k: int = 3) -> list[dict]:
    """Return the k chunks most similar in meaning to the query, best first."""
    chunks, vectors = index()
    query_vector = normalize(np.array(list(embedder().query_embed([query])))[0])
    scores = vectors @ query_vector  # cosine similarity with every chunk at once
    best = np.argsort(scores)[::-1][:k]
    return [
        {
            "source": chunks[i].source,
            "section": chunks[i].section,
            "text": chunks[i].text,
            "score": round(float(scores[i]), 3),
        }
        for i in best
    ]
