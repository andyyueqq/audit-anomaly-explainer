"""
RAG Pipeline for Audit Anomaly Explainer
=========================================
Handles: chunking policy documents, embedding via Google Gemini, indexing in
FAISS, and retrieving top-k policy chunks for a given anomaly query.
"""

import os
import re
import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_policy_by_section(filepath: str) -> list[dict]:
    """
    Split a Markdown policy document into chunks by ## headings.
    Each chunk contains the section heading + body text.
    Returns list of {"source": filename, "section": heading, "text": full_text}.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = Path(filepath).stem
    # Extract document title (first # heading)
    title_match = re.match(r"^#\s+(.+)", content, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else filename

    # Split by ## headings
    sections = re.split(r"\n(?=## )", content)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section heading
        heading_match = re.match(r"##\s+(.+)", section)
        if heading_match:
            heading = heading_match.group(1).strip()
        else:
            heading = "Introduction"

        # Clean up the text (remove excessive whitespace but keep structure)
        text = section.strip()

        # Only include chunks with meaningful content (> 50 chars)
        if len(text) > 50:
            chunks.append({
                "source": filename,
                "doc_title": doc_title,
                "section": heading,
                "text": text,
            })

    return chunks


def load_all_policies(policy_dir: str) -> list[dict]:
    """Load and chunk all .md policy files from the given directory."""
    policy_dir = Path(policy_dir)
    all_chunks = []
    for md_file in sorted(policy_dir.glob("*.md")):
        chunks = chunk_policy_by_section(str(md_file))
        all_chunks.extend(chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding (Google Gemini)
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str], api_key: str, model: str = "gemini-embedding-001") -> np.ndarray:
    """Get embeddings for a list of texts using Google's embedding API, with retry on 429."""
    import time
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model=f"models/{model}",
                content=texts,
                task_type="RETRIEVAL_DOCUMENT",
            )
            return np.array(result["embedding"], dtype="float32")
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait = (attempt + 1) * 10  # 10s, 20s, 30s, 40s, 50s
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            raise e
    raise RuntimeError("Embedding failed after retries")


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------

class PolicyIndex:
    """FAISS-based vector index for policy chunks."""

    def __init__(self):
        self.chunks: list[dict] = []
        self.index = None
        self.dimension: int = 0

    def build(self, chunks: list[dict], api_key: str):
        """Build the index from policy chunks."""
        import faiss

        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings(texts, api_key)
        self.dimension = embeddings.shape[1]

        # Use L2 index (works well for normalized embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query: str, api_key: str, top_k: int = 3) -> list[dict]:
        """Search for the most relevant policy chunks given a query string."""
        import faiss

        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        query_embedding = get_embeddings([query], api_key)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[0][i])
            results.append(chunk)

        return results

    def save(self, path: str):
        """Save index and chunks to disk."""
        import faiss

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "policy.index"))
        with open(directory / "chunks.json", "w") as f:
            json.dump(self.chunks, f, indent=2)

    def load(self, path: str):
        """Load index and chunks from disk."""
        import faiss

        directory = Path(path)
        self.index = faiss.read_index(str(directory / "policy.index"))
        self.dimension = self.index.d
        with open(directory / "chunks.json", "r") as f:
            self.chunks = json.load(f)


# ---------------------------------------------------------------------------
# Convenience: Build or Load
# ---------------------------------------------------------------------------

def build_or_load_index(
    policy_dir: str,
    index_dir: str,
    api_key: str,
    force_rebuild: bool = False
) -> PolicyIndex:
    """Build the index from scratch or load from disk if it exists."""
    idx = PolicyIndex()
    index_path = Path(index_dir)

    if not force_rebuild and (index_path / "policy.index").exists():
        idx.load(index_dir)
        return idx

    chunks = load_all_policies(policy_dir)
    if not chunks:
        raise ValueError(f"No policy chunks found in {policy_dir}")

    idx.build(chunks, api_key)
    idx.save(index_dir)
    return idx
