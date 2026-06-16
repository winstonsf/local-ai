"""
embed_npu.py — run a real embedding model on the Intel NPU via OpenVINO.

Second-stage NPU verification (a real transformer, not a toy op) and the first
real RAG building block: turns text into normalized vectors.

The NPU needs STATIC shapes, so we reshape the model to a fixed [batch, seq_len]
and pad every input to seq_len. Falls back NPU -> GPU -> CPU if a device refuses.

Prereqs:
    optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
        --task feature-extraction --weight-format fp16 models/bge-small-ov

Usage:
    python embed_npu.py                       # uses models/bge-small-ov on NPU
    python embed_npu.py --device GPU
    python embed_npu.py --model models/bge-small-ov --seq-len 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _pool_and_normalize(last_hidden, attention_mask):
    """Mean-pool token vectors over real tokens, then L2-normalize."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    vec = summed / counts
    return vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9, None)


def main() -> int:
    p = argparse.ArgumentParser(description="Embed text on the Intel NPU via OpenVINO.")
    p.add_argument("--model", default="models/bge-small-ov", help="exported OpenVINO IR dir")
    p.add_argument("--device", default="NPU", help="NPU | GPU | CPU")
    p.add_argument("--seq-len", type=int, default=128, help="static sequence length")
    args = p.parse_args()

    if not Path(args.model).exists():
        p.error(f"model dir '{args.model}' not found — run the optimum-cli export first")

    from optimum.intel import OVModelForFeatureExtraction
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    sentences = [
        "The NPU accelerates neural network inference at low power.",   # 0
        "Intel's AI Boost engine speeds up model inference efficiently.",  # 1 ~ 0
        "I grilled vegetables for dinner last night.",                  # 2 unrelated
    ]
    enc = tok(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=args.seq_len,
        return_tensors="np",
    )

    order = [args.device, "GPU", "CPU"]
    tried = []
    for dev in dict.fromkeys(order):  # de-dup, preserve order
        tried.append(dev)
        try:
            model = OVModelForFeatureExtraction.from_pretrained(args.model)
            # Static shapes for NPU; harmless (and fast) on GPU/CPU too.
            model.reshape(len(sentences), args.seq_len)
            model.to(dev)
            model.compile()
            out = model(**{k: enc[k] for k in ("input_ids", "attention_mask")})
            vecs = _pool_and_normalize(out.last_hidden_state, enc["attention_mask"])
            print(f"Device: {dev}")
            print(f"Embeddings: shape={vecs.shape}, dtype={vecs.dtype}")
            sim_related = float(vecs[0] @ vecs[1])
            sim_unrelated = float(vecs[0] @ vecs[2])
            print(f"  cos(related)   sent0~sent1 = {sim_related:.3f}")
            print(f"  cos(unrelated) sent0~sent2 = {sim_unrelated:.3f}")
            verdict = "PASS" if sim_related > sim_unrelated else "SUSPECT"
            print(f"  sanity: related > unrelated -> {verdict}")
            return 0
        except Exception as e:
            print(f"Device {dev} failed: {type(e).__name__}: {e}")

    print(f"All devices failed (tried {tried}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
