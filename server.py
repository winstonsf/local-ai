"""
server.py — OpenAI-compatible API over the local OpenVINO models, for OpenWebUI.

Exposes the iGPU LLM and NPU embedder as an OpenAI-style API:

    GET  /v1/models
    POST /v1/chat/completions   (streaming + non-streaming, + tool/function calling)
    POST /v1/embeddings

Tool calling: when a request includes `tools`, the prompt is built with Qwen2.5's
own chat template (which renders the tool defs), and `<tool_call>` blocks the model
emits are parsed back into OpenAI-style `tool_calls`. This is what lets OpenWebUI's
"Native" function calling work — i.e. the model deciding on its own to search/etc.

Run:
    pip install fastapi "uvicorn[standard]"
    python server.py                 # http://localhost:8000/v1

OpenWebUI: Settings -> Connections -> OpenAI API
    Base URL: http://localhost:8000/v1     API key: anything (unused)
    For autonomous tool use: set the model's Function Calling to "Native".

Models load lazily; a lock serializes generation — single-user, on-box server.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

LLM_ID = "qwen2.5-3b"
EMB_ID = "bge-small"
LLM_DIR = "models/qwen2.5-3b-ov"
EMB_DIR = "models/bge-small-ov"

app = FastAPI(title="local-ai OpenVINO shim")
_lock = threading.Lock()
_llm = None
_emb = None
_tok = None


def get_llm():
    global _llm
    if _llm is None:
        from llm_ov import LLM
        _llm = LLM(LLM_DIR, device="GPU")
    return _llm


def get_emb():
    global _emb
    if _emb is None:
        from ov_embed import Embedder
        _emb = Embedder(EMB_DIR, device="NPU")
    return _emb


def get_tok():
    """HF tokenizer — used only to render the chat template (incl. tool defs)."""
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(LLM_DIR)
    return _tok


# ---- prompt building -------------------------------------------------------
def _content_to_str(content: Any) -> str:
    if isinstance(content, list):  # multimodal-style parts -> join text
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Make OpenAI messages digestible by the Qwen chat template.

    Mainly: assistant tool_calls carry `arguments` as a JSON *string* in OpenAI,
    but the template expects a dict; and content must be a plain string.
    """
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = _content_to_str(m["content"])
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                calls.append({"type": "function",
                              "function": {"name": fn.get("name"), "arguments": args}})
            m["tool_calls"] = calls
            m.setdefault("content", "")
        out.append(m)
    return out


def build_prompt(messages: list[dict], tools: list | None) -> str:
    return get_tok().apply_chat_template(
        normalize_messages(messages), tools=tools or None,
        add_generation_prompt=True, tokenize=False)


def _json_objects(text: str) -> list[dict]:
    """Extract balanced {...} JSON objects (handles nested braces, ignores tags/prose)."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start:i + 1]))
                except Exception:
                    pass
                start = None
    return objs


def parse_tool_calls(text: str) -> list[dict]:
    # Generation is stopped at </tool_call>; the model's tag opening can be mangled,
    # so we don't rely on it — just pull balanced JSON objects shaped like tool calls.
    head = text.split("</tool_call>")[0]
    calls = []
    for i, obj in enumerate(_json_objects(head)):
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            calls.append({"id": f"call_{int(time.time()*1000)}_{i}", "type": "function",
                          "function": {"name": obj["name"],
                                       "arguments": json.dumps(obj["arguments"])}})
    return calls


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
    max_tokens: int | None = 512
    temperature: float | None = 0.0
    stream: bool | None = False
    tools: list | None = None
    tool_choice: Any | None = None


def _gen_config(llm, req: ChatReq):
    cfg = llm._og.GenerationConfig()
    cfg.max_new_tokens = req.max_tokens or 512
    if req.temperature and req.temperature > 0:
        cfg.do_sample = True
        cfg.temperature = req.temperature
    else:
        cfg.do_sample = False
    # Stop right after a tool call so the (small) model can't ramble into fake turns.
    if req.tools and hasattr(cfg, "stop_strings"):
        cfg.stop_strings = {"</tool_call>"}
        if hasattr(cfg, "include_stop_str_in_output"):
            cfg.include_stop_str_in_output = True
    return cfg


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    llm = get_llm()
    prompt = build_prompt(req.messages, req.tools)
    cfg = _gen_config(llm, req)
    cid = f"chatcmpl-{int(time.time()*1000)}"
    created = int(time.time())
    has_tools = bool(req.tools)

    # Plain token streaming (no tools): nicest UX path.
    if req.stream and not has_tools:
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
                yield _sse({"id": cid, "object": "chat.completion.chunk",
                            "created": created, "model": req.model,
                            "choices": [{"index": 0, "delta": {"content": tok},
                                         "finish_reason": None}]})
            yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Generate fully, then (if tools) parse out any tool calls.
    with _lock:
        text = str(llm.pipe.generate(prompt, cfg))
    tool_calls = parse_tool_calls(text) if has_tools else []
    if tool_calls:
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish = "stop"

    if req.stream:  # tools + stream: deliver the resolved message as SSE
        def one_shot():
            yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": message, "finish_reason": None}]})
            yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            yield "data: [DONE]\n\n"
        return StreamingResponse(one_shot(), media_type="text/event-stream")

    return {"id": cid, "object": "chat.completion", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
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
