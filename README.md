# local-ai

A fully local, on-box AI pipeline for **Intel Core Ultra** hardware (NPU + Arc iGPU)
using [OpenVINO](https://docs.openvino.ai/). It recognizes PDFs/images, answers
questions over them with RAG, and exports answers back to real documents — **no cloud**.

```
PDF / image  ──recognize (iGPU VL)──▶  markdown  ──embed (NPU)──▶  retrieve
                                                                       │
                          PDF / DOCX / HTML  ◀──export──  answer  ◀──generate (iGPU)
```

## Hardware / device split

Verified on an Intel **Core Ultra 7 356H** (NPU "AI Boost" + Intel Arc iGPU):

| Stage | Model | Device |
|-------|-------|--------|
| Embeddings | `BAAI/bge-small-en-v1.5` | **NPU** (static shapes) |
| LLM generation | `Qwen/Qwen2.5-3B-Instruct` (int4) | **iGPU** |
| Vision/OCR recognition | `Qwen/Qwen2-VL-2B-Instruct` | **iGPU** (runs as one graph) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows
pip install -r requirements-export.txt       # markdown -> PDF/DOCX/HTML
pip install "openvino>=2025.0" "openvino-genai>=2025.0" "optimum-intel[openvino]" pymupdf pillow

# Export the models to OpenVINO IR (one-time; large downloads)
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 --task feature-extraction \
    --weight-format fp16 models/bge-small-ov
optimum-cli export openvino --model Qwen/Qwen2.5-3B-Instruct --task text-generation-with-past \
    --weight-format int4 models/qwen2.5-3b-ov
optimum-cli export openvino --model Qwen/Qwen2-VL-2B-Instruct --weight-format int4 models/qwen2-vl-2b-ov
```

## Usage

```bash
# 0. Confirm OpenVINO sees the NPU + iGPU
python npu_smoke_test.py

# 1. Recognize a PDF/image into markdown (iGPU vision model)
python recognize.py sample.pdf --out sample_recognized.md

# 2. Ask a grounded question (retrieve on NPU, generate on iGPU), export the answer
python rag_chat.py --doc sample_recognized.md \
    --query "What are the recommended actions?" --export answer.pdf

# Or just the export step (markdown -> document)
python export_doc.py sample.md --to pdf
```

## Modules

| File | Role |
|------|------|
| `npu_smoke_test.py` | Confirm OpenVINO can compile + run on NPU / iGPU / CPU |
| `ov_embed.py` | Reusable NPU-first text embedder (`Embedder`) |
| `rag_store.py` | Chunk → embed → numpy cosine retrieval, with save/load |
| `llm_ov.py` | LLM on the iGPU via `openvino_genai.LLMPipeline` |
| `vlm_ov.py` | Vision-language model on the iGPU via `VLMPipeline` |
| `recognize.py` | PDF/image → markdown (renders PDF pages with PyMuPDF) |
| `rag_chat.py` | End-to-end: retrieve → generate → optional export |
| `export_doc.py` | Markdown → PDF / DOCX / HTML (pure-Python) |

The numpy vector store is intentionally simple; swap for Chroma/FAISS when wiring
OpenWebUI.
