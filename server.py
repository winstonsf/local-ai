"""
server.py — OpenAI-compatible API over the local OpenVINO models, for OpenWebUI.

Exposes the iGPU LLM and NPU embedder as an OpenAI-style API so OpenWebUI (or any
OpenAI client) can use them:

    GET  /v1/models
    POST /v1/chat/completions   (streaming + non-streaming)
    POST /v1/embeddings

Run:
    pip install fastapi "uvicorn[standard]"
    python server.py                 # http://localhost:8000/v1

OpenWebUI: Settings -> Connections -> OpenAI API
    Base URL: http://localhost:8000/v1     API key: anything (unused)

Models are loaded lazily on first use (compiling for NPU/iGPU takes a few seconds).
A lock serializes generation — this is a single-user, on-box server.
"""

from __future__ import annotations

import json
import queue
import threading
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

LLM_ID = "qwen2.5-3b"
EMB_ID = "bge-small"

app = FastAPI(title="local-ai OpenVINO shim")
_lock = threading.Lock()
_llm = None
_emb = None


def get_llm():
    global _llm
    if _llm is None:
        from llm_ov import LLM
        _llm = LLM("models/qwen2.5-3b-ov", device="GPU")
    return _llm


def get_emb():
    global _emb
    if _emb is None:
        from ov_embed import Embedder
        _emb = Embedder("models/bge-small-ov", device="NPU")
    return _emb


def messages_to_qwen(messages: list[dict]) -> str:
    """OpenAI chat messages -> a single Qwen2.5 prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        parts.append(f"<|im_start|>{role}\n{m.get('content', '')}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---- models ----------------------------------------------------------------
@app.get("/v1/models")
def list_models():
    now = int(time.time())
    return {"object": "list", "data": [
        {"id": LLM_ID, "object": "model", "created": now, "owned_by": "local"},
        {"id": EMB_ID, "object": "model", "created": now, "owned_by": "local"},
    ]}


# ---- chat completions ------------------------------------------------------
class ChatReq(BaseModel):
    model: str = LLM_ID
    messages: list[dict]
    max_tokens: int | None = 256
    temperature: float | None = 0.0
    stream: bool | None = False


def _gen_config(llm, req: ChatReq):
    cfg = llm._og.GenerationConfig()
    cfg.max_new_tokens = req.max_tokens or 256
    if req.temperature and req.temperature > 0:
        cfg.do_sample = True
        cfg.temperature = req.temperature
    else:
        cfg.do_sample = False
    return cfg


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    llm = get_llm()
    prompt = messages_to_qwen(req.messages)
    cfg = _gen_config(llm, req)
    cid = f"chatcmpl-{int(time.time()*1000)}"
    created = int(time.time())

    if req.stream:
        def event_stream():
            tokens: "queue.Queue[str | None]" = queue.Queue()

            def run():
                def cb(chunk: str) -> bool:
                    tokens.put(chunk)
                    return False
                try:
                    with _lock:
                        llm.pipe.generate(prompt, cfg, cb)
                finally:
                    tokens.put(None)

            threading.Thread(target=run, daemon=True).start()
            while True:
                tok = tokens.get()
                if tok is None:
                    break
                chunk = {"id": cid, "object": "chat.completion.chunk",
                         "created": created, "model": req.model,
                         "choices": [{"index": 0, "delta": {"content": tok},
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    with _lock:
        text = str(llm.pipe.generate(prompt, cfg))
    return {"id": cid, "object": "chat.completion", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


# ---- embeddings ------------------------------------------------------------
class EmbReq(BaseModel):
    model: str = EMB_ID
    input: str | list[str]


@app.post("/v1/embeddings")
def embeddings(req: EmbReq):
    texts = [req.input] if isinstance(req.input, str) else req.input
    with _lock:
        vecs = get_emb().encode(texts)
    data = [{"object": "embedding", "index": i, "embedding": v.tolist()}
            for i, v in enumerate(vecs)]
    return {"object": "list", "data": data, "model": req.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
