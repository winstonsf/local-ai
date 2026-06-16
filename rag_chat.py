"""
rag_chat.py — retrieve (NPU) -> generate (iGPU): grounded answers over your docs.

Ties the NPU embedder + numpy retriever (rag_store) to the iGPU LLM (llm_ov):
retrieve the top-k chunks, build a context-grounded prompt, and generate an
answer that's restricted to the retrieved context.

    python rag_chat.py --doc sample.md --query "which outlets need attention?"
    python rag_chat.py --load kb.npz --query "when is the follow-up?" --stream
"""

from __future__ import annotations

import argparse

from llm_ov import LLM
from ov_embed import Embedder
from rag_store import RagStore

SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the answer is not in the context, say you don't know. "
    "Be concise."
)


def build_user_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(contexts))
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer using only the context above."


def main() -> int:
    p = argparse.ArgumentParser(description="NPU-retrieve + iGPU-generate RAG chat.")
    p.add_argument("--doc", help="document to ingest (.md/.txt)")
    p.add_argument("--load", help="load a prebuilt index (.npz)")
    p.add_argument("--query", required=True)
    p.add_argument("--embed-model", default="models/bge-small-ov")
    p.add_argument("--llm-model", default="models/qwen2.5-3b-ov")
    p.add_argument("--embed-device", default="NPU")
    p.add_argument("--llm-device", default="GPU")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--export", help="also save the answer to a file (.pdf/.docx/.html)")
    args = p.parse_args()

    store = RagStore(Embedder(args.embed_model, device=args.embed_device))
    if args.load:
        store.load(args.load)
    if args.doc:
        store.add_document(args.doc)
    if store.vecs is None:
        p.error("nothing to retrieve from — pass --doc or --load")

    hits = store.query(args.query, args.k)
    print(f"Retrieved {len(hits)} chunks on {store.emb.device} "
          f"(top cos={hits[0][0]:.3f}):")
    for rank, (score, chunk) in enumerate(hits, 1):
        print(f"  [{rank}] cos={score:.3f}  {' '.join(chunk.split())[:90]}")

    llm = LLM(args.llm_model, device=args.llm_device)
    prompt = build_user_prompt(args.query, [c for _, c in hits])

    print(f"\nAnswer (LLM on {llm.device}):")
    answer = llm.chat(SYSTEM, prompt, max_new_tokens=args.max_new_tokens,
                      stream=args.stream)
    if not args.stream:
        print(answer)

    if args.export:
        from export_doc import markdown_to_file
        sources = "\n".join(f"{i}. {' '.join(c.split())[:120]}"
                            for i, (_, c) in enumerate(hits, 1))
        doc_md = (f"# {args.query}\n\n{answer}\n\n"
                  f"---\n\n*Sources retrieved from the knowledge base:*\n\n{sources}\n")
        out = markdown_to_file(doc_md, args.export)
        print(f"\nExported answer -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
