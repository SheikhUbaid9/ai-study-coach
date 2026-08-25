"""Stage 2 — structure-aware chunking, with semantic splitting as a fallback.

The strategy, and why:

A textbook already contains the best possible chunk boundaries — chapters and
numbered headings, written by humans who were deliberately grouping one idea per
section. Guessing boundaries statistically when real ones are printed on the page
is strictly worse and costs an embedding pass. So:

  1. Group blocks into sections under their heading (structure-aware).
  2. A section that fits under MAX_TOKENS becomes exactly one chunk. Most do.
  3. Only sections that overflow get split further — and there we use semantic
     breakpoints (embedding-similarity dips) rather than blind character counts,
     because that's where a smart boundary actually earns its cost.
  4. Runt sections (a heading with two lines under it) get merged into their
     neighbour, so we don't embed near-empty chunks.

Every chunk records *why* it ended where it did, in `split_reason` — so when you
inspect the output you can tell a clean structural chunk from one the splitter
had to guess at.

Output: out/02_chunks.jsonl
Inspect: python inspect.py chunk
"""

from __future__ import annotations

import re
import sys

from common import count_tokens, log, read_jsonl, warn, write_jsonl
from config import (
    MAX_TOKENS,
    MIN_TOKENS,
    OVERLAP_TOKENS,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_SPLIT_ENABLED,
    TARGET_TOKENS,
)

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


# --------------------------------------------------------------------------
# Group blocks into heading-scoped sections
# --------------------------------------------------------------------------

def build_sections(blocks: list[dict]) -> list[dict]:
    """Walk blocks in reading order, maintaining a stack of open headings.

    The heading stack is what produces `heading_path` — the breadcrumb
    ["Chapter 4 - Integration", "4.2 Definite Integrals"] that later becomes both
    retrieval context and the human-readable citation.
    """
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []   # (level, heading text)
    current: dict | None = None

    def close():
        nonlocal current
        if current and current["body"]:
            sections.append(current)
        current = None

    def open_new(doc_id, book, subject):
        return {
            "doc_id": doc_id,
            "book": book,
            "subject": subject,
            "heading_path": [h for _lvl, h in stack],
            "body": [],
        }

    for blk in blocks:
        if blk["kind"] == "heading":
            close()
            level = blk["level"] or 2
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, blk["text"]))
            current = open_new(blk["doc_id"], blk["book"], blk["subject"])
            continue

        if current is None:
            # Body text before any heading (front matter, or a book whose
            # headings Docling didn't detect).
            current = open_new(blk["doc_id"], blk["book"], blk["subject"])
        current["body"].append(blk)

    close()
    return sections


# --------------------------------------------------------------------------
# Splitting oversized sections
# --------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_END.split(text) if s.strip()]
    return parts or [text]


def _semantic_breakpoints(sentences: list[str]) -> list[int]:
    """Return indices where a new chunk should start, based on where adjacent
    sentence embeddings stop resembling each other. Returns [] if the embedding
    model isn't available, so the caller falls back to size-based splitting."""
    if len(sentences) < 4:
        return []
    try:
        import numpy as np

        from s4_embed import get_model

        model = get_model()
        vecs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
        cutoff = np.percentile(sims, 100 - SEMANTIC_BREAKPOINT_PERCENTILE)
        return [i + 1 for i, s in enumerate(sims) if s <= cutoff]
    except Exception as exc:
        warn("chunk", f"semantic split unavailable ({type(exc).__name__}) — using size split")
        return []


def _split_text(text: str) -> list[tuple[str, str]]:
    """Split an oversized section. Returns [(text, split_reason), ...]."""
    sentences = _sentences(text)
    breakpoints = set(_semantic_breakpoints(sentences)) if SEMANTIC_SPLIT_ENABLED else set()
    reason = "semantic" if breakpoints else "size_overflow"

    out: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_tokens = 0

    for i, sent in enumerate(sentences):
        n = count_tokens(sent)
        # Start a new chunk at a semantic boundary, or when we'd blow the ceiling.
        if buf and (
            (i in breakpoints and buf_tokens >= TARGET_TOKENS // 2)
            or buf_tokens + n > MAX_TOKENS
        ):
            out.append((" ".join(buf), reason))
            buf, buf_tokens = [], 0
        buf.append(sent)
        buf_tokens += n

    if buf:
        out.append((" ".join(buf), reason))
    return out


def _apply_overlap(pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Stitch the tail of each piece onto the front of the next, so a definition
    that straddles a boundary survives in at least one chunk intact."""
    if OVERLAP_TOKENS <= 0 or len(pieces) < 2:
        return pieces

    out = [pieces[0]]
    for text, reason in pieces[1:]:
        prev = out[-1][0]
        tail_sents = _sentences(prev)
        tail, tokens = [], 0
        for sent in reversed(tail_sents):
            t = count_tokens(sent)
            if tokens + t > OVERLAP_TOKENS:
                break
            tail.insert(0, sent)
            tokens += t
        out.append(((" ".join(tail) + " " + text).strip() if tail else text, reason))
    return out


# --------------------------------------------------------------------------
# Section -> chunks
# --------------------------------------------------------------------------

def section_to_chunks(section: dict, seq: int) -> list[dict]:
    body_text = "\n".join(b["text"] for b in section["body"]).strip()
    if not body_text:
        return []

    pages = sorted({b["page"] for b in section["body"] if b["page"] is not None})
    has_table = any(b["kind"] == "table" for b in section["body"])
    total = count_tokens(body_text)

    if total <= MAX_TOKENS:
        pieces = [(body_text, "structure")]
    else:
        pieces = _apply_overlap(_split_text(body_text))

    chunks = []
    for i, (text, reason) in enumerate(pieces):
        chunks.append(
            {
                "chunk_id": f"{section['doc_id']}#c{seq:05d}-{i}",
                "doc_id": section["doc_id"],
                "book": section["book"],
                "subject": section["subject"],
                "heading_path": section["heading_path"],
                "text": text,
                "pages": pages,
                "n_tokens": count_tokens(text),
                "n_chars": len(text),
                "split_reason": reason,
                "part": i,
                "n_parts": len(pieces),
                "has_table": has_table,
            }
        )
    return chunks


def merge_runts(chunks: list[dict]) -> list[dict]:
    """Fold undersized chunks into the previous chunk — but only within the same
    section.

    The subtlety that matters: a runt may only merge into a chunk it shares a
    heading with. Merging across headings would fuse unrelated topics into one
    blob, which destroys exactly the structure this whole stage exists to keep.
    A short section is better left short than glued to the section after it.

    Merging also stops at TARGET_TOKENS rather than MAX_TOKENS, so a run of tiny
    sibling chunks can't cascade into one oversized chunk.
    """
    out: list[dict] = []
    merged = 0
    for ch in chunks:
        prev = out[-1] if out else None
        if (
            prev is not None
            and ch["n_tokens"] < MIN_TOKENS
            and prev["doc_id"] == ch["doc_id"]
            # Same section only — this is the guard that keeps distinct topics apart.
            and prev["heading_path"] == ch["heading_path"]
            and prev["n_tokens"] + ch["n_tokens"] <= TARGET_TOKENS
        ):
            prev["text"] = prev["text"] + "\n" + ch["text"]
            prev["n_tokens"] = count_tokens(prev["text"])
            prev["n_chars"] = len(prev["text"])
            prev["pages"] = sorted(set(prev["pages"]) | set(ch["pages"]))
            if not prev["split_reason"].endswith("+merged"):
                prev["split_reason"] += "+merged"
            merged += 1
            continue
        out.append(ch)
    if merged:
        log("chunk", f"merged {merged} undersized chunks into their own section's neighbour")
    return out


def run() -> list[dict]:
    blocks = read_jsonl("01_blocks")
    sections = build_sections(blocks)
    log("chunk", f"{len(blocks)} blocks -> {len(sections)} heading-scoped sections")

    chunks: list[dict] = []
    for i, section in enumerate(sections):
        chunks.extend(section_to_chunks(section, i))

    chunks = merge_runts(chunks)

    by_reason: dict[str, int] = {}
    for c in chunks:
        by_reason[c["split_reason"]] = by_reason.get(c["split_reason"], 0) + 1
    log("chunk", f"{len(chunks)} chunks — " + ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())))

    write_jsonl("02_chunks", chunks)
    return chunks


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
