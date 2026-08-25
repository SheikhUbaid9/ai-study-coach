"""Quality report for each stage — the whole point of building this stage-by-stage.

Every check below exists because it catches a specific, known way that RAG
ingestion silently goes wrong. None of them throw errors; bad ingestion almost
never throws. It just quietly retrieves the wrong thing forever. So each check
prints a verdict — OK / CHECK / BAD — and tells you what to do about it.

    python inspect_stage.py parse
    python inspect_stage.py chunk
    python inspect_stage.py tag
    python inspect_stage.py embed
    python inspect_stage.py all
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter

from common import read_jsonl
from config import MAX_TOKENS, MIN_CHARS_PER_PAGE, MIN_TOKENS, TARGET_TOKENS

OK, CHECK, BAD = "  OK  ", " CHECK", " BAD  "


def _h(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _v(verdict: str, label: str, detail: str = "") -> None:
    print(f"[{verdict}] {label}" + (f"\n         {detail}" if detail else ""))


def _pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.0f}%)" if total else "0"


def _dist(values: list[int]) -> str:
    if not values:
        return "n/a"
    s = sorted(values)
    p = lambda q: s[min(int(len(s) * q), len(s) - 1)]  # noqa: E731
    return (
        f"min={s[0]}  p10={p(0.10)}  median={statistics.median(s):.0f}  "
        f"p90={p(0.90)}  max={s[-1]}"
    )


# --------------------------------------------------------------------------

def inspect_parse() -> None:
    blocks = read_jsonl("01_blocks")
    _h(f"STAGE 1 — PARSE   ({len(blocks)} blocks)")

    docs = Counter(b["doc_id"] for b in blocks)
    kinds = Counter(b["kind"] for b in blocks)
    print(f"documents : {dict(docs)}")
    print(f"block kinds: {dict(kinds)}\n")

    # Did heading detection actually work? Without headings, structure-aware
    # chunking silently degrades into one giant blob per document.
    headings = kinds.get("heading", 0)
    ratio = headings / max(len(blocks), 1)
    if headings == 0:
        _v(BAD, "No headings detected.",
           "Structure-aware chunking cannot work. The PDF may be scanned, or "
           "Docling didn't recognise this book's heading style.")
    elif ratio < 0.01:
        _v(CHECK, f"Very few headings ({headings}, {ratio:.1%} of blocks).",
           "Expect large chunks split by size rather than by section.")
    else:
        _v(OK, f"{headings} headings detected ({ratio:.1%} of blocks).")

    # Pages with almost no text usually mean a scan OCR missed, or a full-page
    # diagram — either way, content you think you ingested but didn't.
    per_page: dict = {}
    for b in blocks:
        if b["page"] is not None:
            key = (b["doc_id"], b["page"])
            per_page[key] = per_page.get(key, 0) + len(b["text"])
    if per_page:
        thin = [k for k, v in per_page.items() if v < MIN_CHARS_PER_PAGE]
        if len(thin) > len(per_page) * 0.2:
            _v(BAD, f"{_pct(len(thin), len(per_page))} of pages are near-empty.",
               "Likely a scanned book where OCR failed. Check OCR_ENABLED and that "
               "Tesseract is installed.")
        elif thin:
            _v(CHECK, f"{_pct(len(thin), len(per_page))} of pages are near-empty.",
               "Usually full-page diagrams — fine, unless those pages have text you need.")
        else:
            _v(OK, f"All {len(per_page)} pages have text.")
    else:
        _v(CHECK, "No page numbers captured.", "Citations will lack page references.")

    lens = [len(b["text"]) for b in blocks]
    print(f"\nblock length (chars): {_dist(lens)}")

    print("\nfirst 12 detected headings:")
    for b in [b for b in blocks if b["kind"] == "heading"][:12]:
        print(f"   L{b['level']}  p.{b['page']}  {b['text'][:66]}")

    print("\n--- sample body block ---")
    body = next((b for b in blocks if b["kind"] == "text" and len(b["text"]) > 200), None)
    print((body["text"][:400] + "...") if body else "(none found)")


def inspect_chunk() -> None:
    chunks = read_jsonl("02_chunks")
    _h(f"STAGE 2 — CHUNK   ({len(chunks)} chunks)")

    tokens = [c["n_tokens"] for c in chunks]
    reasons = Counter(c["split_reason"] for c in chunks)
    print(f"tokens per chunk: {_dist(tokens)}")
    print(f"target={TARGET_TOKENS}  max={MAX_TOKENS}  min={MIN_TOKENS}")
    print(f"split reasons   : {dict(reasons)}\n")

    # A high share of clean structural chunks is the signal that the book's own
    # section boundaries carried through — that's the good outcome.
    structural = sum(1 for c in chunks if c["split_reason"].startswith("structure"))
    if structural / max(len(chunks), 1) > 0.6:
        _v(OK, f"{_pct(structural, len(chunks))} of chunks follow the book's own sections.")
    else:
        _v(CHECK, f"Only {_pct(structural, len(chunks))} follow real section boundaries.",
           "Most chunks were split by size/semantics — check heading detection in stage 1.")

    oversized = [c for c in chunks if c["n_tokens"] > MAX_TOKENS]
    if oversized:
        _v(BAD, f"{len(oversized)} chunks exceed MAX_TOKENS.",
           "These get TRUNCATED by the embedding model — no error, the tail is just "
           "lost. Lower MAX_TOKENS, or check it against your model's max sequence "
           "length (bge-small = 512).")
    else:
        _v(OK, "No chunk exceeds MAX_TOKENS.")

    runts = [c for c in chunks if c["n_tokens"] < MIN_TOKENS]
    if len(runts) > len(chunks) * 0.1:
        _v(CHECK, f"{_pct(len(runts), len(chunks))} of chunks are very short.",
           "Short chunks embed poorly and crowd out real content at retrieval time.")
    else:
        _v(OK, f"{len(runts)} undersized chunks.")

    orphans = [c for c in chunks if not c["heading_path"]]
    if orphans:
        _v(CHECK, f"{_pct(len(orphans), len(chunks))} have no heading path.",
           "These will get weak citations — usually front matter or an undetected heading.")
    else:
        _v(OK, "Every chunk sits under a heading.")

    # A chunk ending mid-sentence means a boundary landed badly.
    broken = [c for c in chunks if c["text"].rstrip()[-1:] not in ".!?:)]\"'" and c["n_parts"] > 1]
    if len(broken) > len(chunks) * 0.15:
        _v(CHECK, f"{len(broken)} chunks end mid-sentence.",
           "Raise OVERLAP_TOKENS so the cut-off idea survives in the next chunk.")
    else:
        _v(OK, f"{len(broken)} chunks end mid-sentence.")

    print("\n--- longest chunk (most likely to be a problem) ---")
    worst = max(chunks, key=lambda c: c["n_tokens"])
    print(f"{worst['n_tokens']} tokens | {' > '.join(worst['heading_path']) or '(no heading)'}")
    print(worst["text"][:400] + "...")

    print("\n--- a typical structural chunk ---")
    typical = min(
        (c for c in chunks if c["split_reason"].startswith("structure")),
        key=lambda c: abs(c["n_tokens"] - TARGET_TOKENS),
        default=None,
    )
    if typical:
        print(f"{typical['n_tokens']} tokens | {' > '.join(typical['heading_path'])}")
        print(typical["text"][:400] + "...")


def inspect_tag() -> None:
    tagged = read_jsonl("03_tagged")
    _h(f"STAGE 3 — TAG   ({len(tagged)} chunks)")

    no_ch = [t for t in tagged if not t["chapter_no"]]
    no_pg = [t for t in tagged if not t["page_range"]]
    subjects = Counter(t["subject"] for t in tagged)
    print(f"subjects: {dict(subjects)}\n")

    if "Unknown" in subjects:
        _v(CHECK, f"{subjects['Unknown']} chunks have subject='Unknown'.",
           "Subject is guessed from the filename — rename PDFs like 'fsc-physics-part1.pdf'.")
    else:
        _v(OK, "Every chunk has a subject.")

    if len(no_ch) > len(tagged) * 0.3:
        _v(CHECK, f"{_pct(len(no_ch), len(tagged))} have no chapter number.",
           "Citations will say the section but not the chapter.")
    else:
        _v(OK, f"{_pct(len(no_ch), len(tagged))} lack a chapter number.")

    if len(no_pg) > len(tagged) * 0.1:
        _v(BAD, f"{_pct(len(no_pg), len(tagged))} have no page number.",
           "You cannot recover this later — it must come from stage 1 provenance.")
    else:
        _v(OK, f"{_pct(len(no_pg), len(tagged))} lack a page number.")

    print("\nsample citations:")
    for t in tagged[:8]:
        print(f"   {t['citation']}")

    print("\nsample keywords:")
    for t in tagged[:5]:
        print(f"   {', '.join(t['keywords'][:6])}")

    print("\n--- what actually gets embedded (note the heading prefix) ---")
    print(tagged[0]["embed_text"][:400] + "...")


def inspect_embed() -> None:
    import numpy as np

    from config import OUT_DIR

    rows = read_jsonl("04_embedded")
    vecs = np.load(OUT_DIR / "04_vectors.npy")
    _h(f"STAGE 4 — EMBED   ({vecs.shape[0]} vectors, {vecs.shape[1]} dims)")

    models = {r["embed_model"] for r in rows}
    print(f"model(s): {models}\n")

    if len(models) > 1:
        _v(BAD, "Vectors from more than one embedding model in the same index.",
           "Similarity across models is meaningless. Re-run stage 4 on everything.")
    else:
        _v(OK, "Single embedding model throughout.")

    if vecs.shape[0] != len(rows):
        _v(BAD, f"{vecs.shape[0]} vectors vs {len(rows)} metadata rows — misaligned.")
    else:
        _v(OK, "Vector count matches metadata row count.")

    if not np.isfinite(vecs).all():
        _v(BAD, "Vectors contain NaN/inf.")
    else:
        _v(OK, "All vectors finite.")

    norms = np.linalg.norm(vecs, axis=1)
    if abs(float(norms.mean()) - 1.0) > 0.01:
        _v(CHECK, f"Vectors not unit-normalised (mean norm {norms.mean():.3f}).",
           "Retrieval assumes normalised vectors so dot product == cosine.")
    else:
        _v(OK, "Vectors are unit-normalised.")

    # Near-identical vectors mean duplicated content competing for retrieval slots.
    sample = vecs[:500]
    sims = sample @ sample.T
    np.fill_diagonal(sims, 0)
    dupes = int((sims > 0.98).sum() // 2)
    if dupes > len(sample) * 0.05:
        _v(CHECK, f"{dupes} near-duplicate pairs in the first {len(sample)} chunks.",
           "Often repeated boilerplate, or too much overlap. They crowd out real results.")
    else:
        _v(OK, f"{dupes} near-duplicate pairs in the sample.")

    print(f"\nsimilarity spread (sample): mean={sims.mean():.3f}  max={sims.max():.3f}")


def inspect_index() -> None:
    import json

    from config import OUT_DIR

    path = OUT_DIR / "05_index_manifest.json"
    _h("STAGE 5 — INDEX")
    if not path.exists():
        _v(CHECK, "No manifest — stage 5 hasn't run yet.")
        return
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))


STAGES = {
    "parse": inspect_parse,
    "chunk": inspect_chunk,
    "tag": inspect_tag,
    "embed": inspect_embed,
    "index": inspect_index,
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = list(STAGES) if which == "all" else [which]
    for name in targets:
        fn = STAGES.get(name)
        if fn is None:
            print(f"unknown stage {name!r}; choose from: {', '.join(STAGES)}, all")
            sys.exit(1)
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"\n[{name}] not run yet: {exc}")


if __name__ == "__main__":
    main()
