"""
export_doc.py — turn model output (markdown) into PDF / DOCX / HTML files.

This is the "document output" step of the local-AI setup: your model (Qwen3-VL,
etc.) returns markdown text, and this script renders it into a real file.

Pure-Python, no system binaries required (works cleanly on Windows):
    pip install -r requirements-export.txt

Usage:
    # from a markdown file
    python export_doc.py report.md --to pdf
    python export_doc.py report.md --to docx --out summary.docx

    # from stdin (pipe model output straight in)
    echo "# Hello\n\nWorld" | python export_doc.py - --to pdf --out hello.pdf

    # programmatically (e.g. right after an LLM call)
    from export_doc import markdown_to_file
    markdown_to_file(model_output, "result.pdf")   # format inferred from extension
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import markdown as md_lib

# Light, print-friendly styling for the PDF/HTML output.
_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt;
       line-height: 1.45; color: #1a1a1a; margin: 2.5em; }
h1 { font-size: 20pt; border-bottom: 2px solid #444; padding-bottom: 4px; }
h2 { font-size: 15pt; margin-top: 1.2em; }
h3 { font-size: 12.5pt; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, monospace; font-size: 9.5pt; }
pre { background: #f2f2f2; padding: 10px; border-radius: 5px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #bbb; padding: 6px 10px; text-align: left; }
th { background: #efefef; }
blockquote { border-left: 4px solid #ccc; margin: 1em 0; padding-left: 1em;
             color: #555; }
"""

_MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "nl2br"]


def markdown_to_html(text: str, title: str = "Document") -> str:
    """Render markdown text to a full, styled HTML document."""
    body = md_lib.markdown(text, extensions=_MD_EXTENSIONS)
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def _write_html(html: str, out: Path) -> None:
    out.write_text(html, encoding="utf-8")


def _write_pdf(html: str, out: Path) -> None:
    # xhtml2pdf: pure-Python HTML->PDF, no system deps. Good enough for reports.
    from xhtml2pdf import pisa

    with out.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh)
    if result.err:
        raise RuntimeError(f"PDF generation failed ({result.err} errors)")


def _write_docx(html: str, out: Path) -> None:
    # htmldocx converts the rendered HTML (tables, headings, lists) into a docx.
    from docx import Document
    from htmldocx import HtmlToDocx

    document = Document()
    HtmlToDocx().add_html_to_document(html, document)
    document.save(str(out))


_WRITERS = {"pdf": _write_pdf, "docx": _write_docx, "html": _write_html}


def markdown_to_file(text: str, out_path: str | Path, fmt: str | None = None) -> Path:
    """Render markdown `text` to `out_path`. Format inferred from extension if omitted."""
    out = Path(out_path)
    fmt = (fmt or out.suffix.lstrip(".")).lower()
    if fmt not in _WRITERS:
        raise ValueError(f"Unsupported format '{fmt}'. Choose: {', '.join(_WRITERS)}")
    html = markdown_to_html(text, title=out.stem)
    _WRITERS[fmt](html, out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render markdown to PDF/DOCX/HTML.")
    p.add_argument("input", help="Markdown file, or '-' to read from stdin")
    p.add_argument("--to", choices=list(_WRITERS), help="Output format (else inferred from --out)")
    p.add_argument("--out", help="Output path (default: input name with new extension)")
    args = p.parse_args(argv)

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")

    fmt = args.to
    if args.out:
        out_path = Path(args.out)
        fmt = fmt or out_path.suffix.lstrip(".")
    else:
        if not fmt:
            p.error("specify --to or --out so the format is known")
        stem = "output" if args.input == "-" else Path(args.input).stem
        out_path = Path(f"{stem}.{fmt}")

    result = markdown_to_file(text, out_path, fmt)
    print(f"Wrote {result.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
