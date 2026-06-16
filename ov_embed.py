"""
ov_embed.py — reusable OpenVINO text embedder, NPU-first.

Wraps an exported OpenVINO IR feature-extraction model (e.g. bge-small) behind a
small Embedder.encode(texts) -> (N, dim) float32 API. The NPU needs static
shapes, so the model is compiled at batch=1 x fixed seq_len and texts are
encoded one row at a time; this single code path also works on GPU/CPU.

    from ov_embed import Embedder
    emb = Embedder("models/bge-small-ov", device="NPU")
    vecs = emb.encode(["hello world", "another sentence"])   # (2, 384), L2-normalized
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_dir: str = "models/bge-small-ov", device: str = "NPU",
                 seq_len: int = 128, fallback: bool = True):
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"model dir '{model_dir}' not found — run the optimum-cli export first")

        from optimum.intel import OVModelForFeatureExtraction
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.tok = AutoTokenizer.from_pretrained(model_dir)

        order = list(dict.fromkeys([device] + (["GPU", "CPU"] if fallback else [])))
        last_err = None
        for dev in order:
            try:
                model = OVModelForFeatureExtraction.from_pretrained(model_dir)
                model.reshape(1, seq_len)          # static [batch=1, seq_len] for NPU
                model.to(dev)
                model.compile()
                self.model, self.device = model, dev
                break
            except Exception as e:                  # try the next device
                last_err = e
        else:
            raise RuntimeError(f"no device worked (tried {order}): {last_err}")

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(texts, padding="max_length", truncation=True,
                       max_length=self.seq_len, return_tensors="np")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        out = np.empty((len(texts), self._dim(ids[:1], mask[:1])), dtype=np.float32)
        for i in range(len(texts)):
            out[i] = self._embed_one(ids[i:i + 1], mask[i:i + 1])
        return out

    # --- internals -------------------------------------------------------
    def _embed_one(self, ids, mask) -> np.ndarray:
        res = self.model(input_ids=ids, attention_mask=mask)
        return self._pool(res.last_hidden_state, mask)[0]

    def _dim(self, ids, mask) -> int:
        return self._embed_one(ids, mask).shape[0]

    @staticmethod
    def _pool(last_hidden, attention_mask) -> np.ndarray:
        m = attention_mask[..., None].astype(np.float32)
        vec = (last_hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        return vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9, None)
