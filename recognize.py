"""
recognize.py — PDF/image -> markdown via the VL model on the iGPU.

The recognition (input) side of the local-AI pipeline: renders PDF pages (or
loads an image), runs each through the VL model, and emits markdown ready to
feed the RAG store or the export pipeline.

    python recognize.py sample.pdf                      # -> stdout
    python recognize.py scan.png --out scan.md
    python recognize.py report.pdf --out report.md --dpi 200
    # then: python rag_store.py --doc report.md --query "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PROMPT = (
    "Read this document page and transcribe ALL of its content as clean GitHub-"
    "flavored markdown. Preserve headings, lists, and tables. Output only the "
    "markdown, no commentary."
)


def load_pages(path: str, dpi: int):
    """Yield PIL images: rendered PDF pages, or the single image file."""
    from PIL import Image

    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        import fitz  # pymupdf

        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    elif ext in IMAGE_EXTS:
        yield Image.open(path)
    else:
        raise ValueError(f"unsupported input '{ext}' (use PDF or {sorted(IMAGE_EXTS)})")


def main() -> int:
    p = argparse.ArgumentParser(description="Recognize a PDF/image into markdown.")
    p.add_argument("input", help="PDF or image file")
    p.add_argument("--out", help="write markdown here (default: stdout)")
    p.add_argument("--model", default="models/qwen2-vl-2b-ov")
    p.add_argument("--device", default="GPU")
    p.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args()

    from vlm_ov import VLM

    pages = list(load_pages(args.input, args.dpi))
    vlm = VLM(args.model, device=args.device)
    print(f"Recognizing {len(pages)} page(s) from {args.input} on {vlm.device}...")

    parts = []
    for i, img in enumerate(pages, 1):
        md = vlm.recognize(img, DEFAULT_PROMPT, max_new_tokens=args.max_new_tokens).strip()
        parts.append(md if len(pages) == 1 else f"## Page {i}\n\n{md}")
        print(f"  page {i}/{len(pages)}: {len(md)} chars")

    result = "\n\n".join(parts)
    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
        print(f"Wrote {Path(args.out).resolve()}")
    else:
        print("\n----- MARKDOWN -----\n")
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
