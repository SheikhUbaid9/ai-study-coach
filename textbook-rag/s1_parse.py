"""Stage 1 — parse PDFs into structured blocks with Docling.

Why Docling rather than pdfplumber: a textbook's *structure* is the thing we most
need to keep. pdfplumber gives you a wall of text per page; Docling gives you
labelled items — this is a section header, this is body text, this is a table —
plus the page each one came from. Stage 2 chunks along those headings and Stage 3
turns the page numbers into citations, so throwing structure away here would
quietly cap the quality of everything downstream.

Output: out/01_blocks.jsonl, one record per block:
    {doc_id, block_id, kind, level, text, page, book, subject}

Inspect it with:  python inspect.py parse
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from common import log, warn, write_jsonl
from config import (
    BOOKS_DIR,
    DO_TABLE_STRUCTURE,
    OCR_ENABLED,
    OCR_LANGUAGES,
    PAGE_LIMIT,
)

# Docling labels we treat as headings (they open a new section), body text, or
# things to drop outright. Running headers/footers repeat on every page and only
# add noise to embeddings, so they go.
HEADING_LABELS = {"title", "section_header", "subtitle-level-1"}
BODY_LABELS = {"text", "paragraph", "list_item", "formula", "caption", "code"}
TABLE_LABELS = {"table"}
DROP_LABELS = {"page_header", "page_footer", "picture", "furniture"}


def _build_converter():
    """Docling's option classes have moved around across 2.x releases, so build
    the converter defensively and degrade to defaults rather than crashing."""
    from docling.document_converter import DocumentConverter

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = OCR_ENABLED
        opts.do_table_structure = DO_TABLE_STRUCTURE
        # Only OCR pages that actually lack a text layer. Forcing OCR everywhere
        # on a digital textbook is slower AND worse than using the real text.
        if hasattr(opts, "ocr_options") and OCR_LANGUAGES:
            try:
                opts.ocr_options.lang = OCR_LANGUAGES
            except Exception:
                pass
        if hasattr(opts.ocr_options, "force_full_page_ocr"):
            opts.ocr_options.force_full_page_ocr = False

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    except Exception as exc:  # pragma: no cover - depends on docling version
        warn("parse", f"using Docling defaults ({type(exc).__name__}: {exc})")
        return DocumentConverter()


def _page_of(item) -> int | None:
    """Docling records provenance per item; the page number is what makes a
    citation like 'Physics Ch 12, p.84' possible later."""
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, (list, tuple)) and prov else prov
    return getattr(first, "page_no", None)


def _label_of(item) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def _heading_level(item, label: str) -> int:
    if label == "title":
        return 1
    lvl = getattr(item, "level", None)
    if isinstance(lvl, int) and lvl > 0:
        return min(lvl + 1, 6)
    return 2


def _infer_metadata(pdf_path: Path) -> tuple[str, str]:
    """Guess subject + book name from the filename, e.g.
    'fsc-physics-part1.pdf' -> ('Physics', 'fsc-physics-part1').
    These are only defaults — override them in books/manifest.json if wrong."""
    stem = pdf_path.stem
    subjects = ["physics", "chemistry", "biology", "mathematics", "math", "english"]
    found = next((s for s in subjects if s in stem.lower()), None)
    subject = {"math": "Mathematics"}.get(found, found.title()) if found else "Unknown"
    return subject, stem


def _table_to_text(item, doc) -> str:
    """Flatten a table to text WITHOUT losing the header association.

    Naive flattening ('12 84 3.4 …') destroys the row/column relationship and is
    a documented top failure mode. Emitting 'Header: value' pairs per row keeps
    each number attached to the thing it measures."""
    for attempt in ("export_to_markdown", "export_to_dataframe"):
        fn = getattr(item, attempt, None)
        if fn is None:
            continue
        try:
            out = fn(doc) if attempt == "export_to_markdown" else fn()
            if attempt == "export_to_markdown":
                return str(out)
            df = out
            lines = []
            for _, row in df.iterrows():
                cells = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
                if cells:
                    lines.append("; ".join(cells))
            return "\n".join(lines)
        except Exception:
            continue
    return str(getattr(item, "text", "") or "")


def parse_pdf(pdf_path: Path, converter) -> list[dict]:
    subject, book = _infer_metadata(pdf_path)
    doc_id = pdf_path.stem

    if PAGE_LIMIT:
        log("parse", f"PAGE_LIMIT={PAGE_LIMIT} — trial run, only the first "
                     f"{PAGE_LIMIT} pages will be parsed")
    log("parse", f"converting {pdf_path.name} (this is the slow step)...")
    try:
        result = (
            converter.convert(str(pdf_path), page_range=(1, PAGE_LIMIT))
            if PAGE_LIMIT
            else converter.convert(str(pdf_path))
        )
    except TypeError:
        # Older Docling releases don't accept page_range; fall back to the full doc.
        warn("parse", "this Docling version ignores PAGE_LIMIT — parsing the whole file")
        result = converter.convert(str(pdf_path))
    doc = result.document

    blocks: list[dict] = []
    dropped = 0

    for item, _level in doc.iterate_items():
        label = _label_of(item)

        if label in DROP_LABELS:
            dropped += 1
            continue

        if label in TABLE_LABELS:
            text, kind, level = _table_to_text(item, doc), "table", 0
        elif label in HEADING_LABELS:
            text = (getattr(item, "text", "") or "").strip()
            kind, level = "heading", _heading_level(item, label)
        elif label in BODY_LABELS:
            text, kind, level = (getattr(item, "text", "") or "").strip(), "text", 0
        else:
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                dropped += 1
                continue
            kind, level = "text", 0

        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            dropped += 1
            continue

        blocks.append(
            {
                "doc_id": doc_id,
                "block_id": f"{doc_id}#b{len(blocks):05d}",
                "kind": kind,
                "level": level,
                "text": text,
                "page": _page_of(item),
                "book": book,
                "subject": subject,
            }
        )

    n_pages = getattr(doc, "num_pages", None)
    n_pages = n_pages() if callable(n_pages) else n_pages
    log(
        "parse",
        f"{pdf_path.name}: {len(blocks)} blocks "
        f"({sum(b['kind'] == 'heading' for b in blocks)} headings, "
        f"{sum(b['kind'] == 'table' for b in blocks)} tables), "
        f"{dropped} dropped, {n_pages or '?'} pages",
    )
    return blocks


def run() -> list[dict]:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))

    if not pdfs:
        warn("parse", f"no PDFs found in {BOOKS_DIR}")
        warn("parse", "drop your FSc textbook PDFs there and re-run.")
        return []

    converter = _build_converter()
    all_blocks: list[dict] = []
    for pdf in pdfs:
        try:
            all_blocks.extend(parse_pdf(pdf, converter))
        except Exception as exc:
            warn("parse", f"FAILED on {pdf.name}: {type(exc).__name__}: {exc}")

    write_jsonl("01_blocks", all_blocks)
    return all_blocks


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
