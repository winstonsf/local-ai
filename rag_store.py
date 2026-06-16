"""
rag_store.py — minimal local RAG store: chunk -> embed (NPU) -> cosine retrieve.

A transparent, zero-extra-deps vector store (numpy) to prove the ingest ->
retrieve slice end-to-end on the Intel NPU. Swappable for Chroma/FAISS when we
wire OpenWebUI; the Embedder and chunking carry straight over.

    # one-shot demo: ingest a doc and run a query
    python rag_store.py --doc sample.md --query "which outlets need attention?"

    # build a persistent index, then query it later
    python rag_store.py --doc sample.md --save kb.npz
    python rag_store.py --load kb.npz --query "when is the follow-up?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ov_embed import Embedder


def chunk_text(text: str, target_chars: int = 500, overlap: int = 80) -> list[str]:
    """Greedy paragraph-packing into ~target_chars chunks with a little overlap."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > target_chars:
            chunks.append(buf)
            buf = (buf[-overlap:] + "\n\n" + p) if overlap else p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


class RagStore:
    def __init__(self, embedder: Embedder):
        self.emb = embedder
        self.chunks: list[str] = []
        self.vecs: np.ndarray | None = None

    def add_document(self, path: str) -> int:
        new = chunk_text(Path(path).read_text(encoding="utf-8"))
        vecs = self.emb.encode(new)
        self.chunks.extend(new)
        self.vecs = vecs if self.vecs is None else np.vstack([self.vecs, vecs])
        return len(new)

    def query(self, q: str, k: int = 3) -> list[tuple[float, str]]:
        if self.vecs is None:
            return []
        qv = self.emb.encode([q])[0]
        scores = self.vecs @ qv                      # cosine (vectors are normalized)
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]

    def save(self, path: str) -> None:
        np.savez(path, vecs=self.vecs)
        Path(path).with_suffix(".chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        self.vecs = np.load(path)["vecs"]
        self.chunks = json.loads(
            Path(path).with_suffix(".chunks.json").read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Minimal NPU-backed RAG store.")
    p.add_argument("--doc", help="document to ingest (.md/.txt)")
    p.add_argument("--query", help="query to retrieve against")
    p.add_argument("--save", help="save index to this .npz path")
    p.add_argument("--load", help="load index from this .npz path (skip embedding model on ingest-less runs)")
    p.add_argument("--model", default="models/bge-small-ov")
    p.add_argument("--device", default="NPU")
    p.add_argument("-k", type=int, default=3)
    args = p.parse_args()

    store = RagStore(Embedder(args.model, device=args.device))
    if args.load:
        store.load(args.load)
        print(f"Loaded {len(store.chunks)} chunks from {args.load}")
    if args.doc:
        n = store.add_document(args.doc)
        print(f"Ingested {args.doc}: {n} chunks on device={store.emb.device}")
    if args.save:
        store.save(args.save)
        print(f"Saved index -> {args.save}")
    if args.query:
        print(f"\nQuery: {args.query}")
        for rank, (score, chunk) in enumerate(store.query(args.query, args.k), 1):
            preview = " ".join(chunk.split())[:160]
            print(f"  #{rank}  cos={score:.3f}  {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
