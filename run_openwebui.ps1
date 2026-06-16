# run_openwebui.ps1 — launch OpenWebUI pre-wired to the local OpenVINO shim.
#
# Prereqs: the shim is running (python server.py -> http://localhost:8000/v1) and
# OpenWebUI is installed in its own venv (kept separate to avoid dependency
# conflicts with the OpenVINO stack).
#
#   .\run_openwebui.ps1            # then open http://localhost:8080
#
# Env vars below point OpenWebUI's chat models AND its RAG embeddings at the shim,
# so generation runs on the iGPU and embeddings run on the NPU.

param(
    [string]$OwuiVenv = "C:\Users\winst\openwebui-venv",
    [int]$Port = 8080,
    [string]$ShimBase = "http://localhost:8000/v1"
)

$env:WEBUI_AUTH = "False"                  # single-user local; skip login
$env:ENABLE_OLLAMA_API = "False"

# Chat models come from the shim (Qwen2.5-3B on the iGPU)
$env:OPENAI_API_BASE_URL = $ShimBase
$env:OPENAI_API_KEY = "local"

# RAG embeddings come from the shim too (bge-small on the NPU) — no local model download
$env:RAG_EMBEDDING_ENGINE = "openai"
$env:RAG_OPENAI_API_BASE_URL = $ShimBase
$env:RAG_OPENAI_API_KEY = "local"
$env:RAG_EMBEDDING_MODEL = "bge-small"

$env:DATA_DIR = "C:\Users\winst\openwebui-data"

& "$OwuiVenv\Scripts\open-webui.exe" serve --port $Port
