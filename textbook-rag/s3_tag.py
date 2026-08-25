"""Stage 3 — attach citation metadata and build the text we actually embed.

Two jobs here, and the second one is easy to miss:

1. **Citation metadata.** Subject, chapter, page range, formatted into a string a
   student can act on ("Physics — Ch 12: Electrostatics, p.84-85"). This has to
   happen at ingestion: once chunks are embedded and indexed, you cannot
   reconstruct which page a sentence came from.

2. **Contextual embedding text.** We embed the heading path *together with* the
   chunk body. A chunk whose body reads "It is measured in newton-metres." is
   nearly meaningless alone — embedded as "Physics > Chapter 5: Torque >> It is
   measured in newton-metres." it lands in the right neighbourhood. The stored
   `text` stays clean for display; `embed_text` is what Stage 4 vectorises.

Output: out/03_tagged.jsonl
Inspect: python inspect.py tag
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from common import log, read_jsonl, write_jsonl

# Words too common in any textbook to be useful as retrieval keywords.
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "which", "can", "has", "have", "had", "its", "not", "but", "all", "any",
    "when", "then", "than", "there", "their", "these", "those", "such", "into",
    "will", "would", "also", "been", "more", "most", "other", "some", "each",
    "may", "one", "two", "use", "used", "using", "shown", "figure", "fig",
    "example", "table", "chapter", "section", "page", "equation", "given",
}

CHAPTER_RE = re.compile(
    r"(?:chapter|unit|module|lesson|ch\.?)\s*[-–—:]?\s*(\d+)", re.IGNORECASE
)
NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+)(?:\.\d+)*\s")


def extract_chapter(heading_path: list[str]) -> tuple[str | None, str | None]:
    """Return (chapter_number, chapter_title) from the outermost headings.

    Textbooks label chapters inconsistently — 'Chapter 4', 'UNIT 4', '4.2 ...' —
    so try an explicit match first, then fall back to a leading section number.
    """
    for heading in heading_path:
        m = CHAPTER_RE.search(heading)
        if m:
            title = CHAPTER_RE.sub("", heading).strip(" -–—:.")
            return m.group(1), title or None
    for heading in heading_path:
        m = NUMBERED_HEADING_RE.match(heading)
        if m:
            return m.group(1), NUMBERED_HEADING_RE.sub("", heading).strip()
    return None, (heading_path[0] if heading_path else None)


def format_pages(pages: list[int]) -> str | None:
    if not pages:
        return None
    if len(pages) == 1:
        return f"p.{pages[0]}"
    return f"p.{min(pages)}-{max(pages)}"


def build_citation(subject: str, chapter_no, chapter_title, pages: list[int]) -> str:
    bits = [subject] if subject and subject != "Unknown" else []
    chapter = ""
    if chapter_no:
        chapter = f"Ch {chapter_no}"
        if chapter_title:
            chapter += f": {chapter_title}"
    elif chapter_title:
        chapter = chapter_title
    if chapter:
        bits.append(chapter)
    page_str = format_pages(pages)
    if page_str:
        bits.append(page_str)
    return " — ".join(bits) if bits else "source unknown"


def extract_keywords(text: str, k: int = 8) -> list[str]:
    """Cheap frequency-based keywords. Deliberately not an LLM call: the
    reference doc's layered pattern is to use free/deterministic methods at
    corpus scale and save model calls for cases that genuinely need them."""
    words = re.findall(r"[a-z][a-z\-]{3,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [w for w, _ in counts.most_common(k)]


def build_embed_text(subject: str, heading_path: list[str], text: str) -> str:
    trail = " > ".join([subject] + heading_path) if heading_path else subject
    return f"{trail}\n\n{text}" if trail else text


def tag_chunk(chunk: dict) -> dict:
    chapter_no, chapter_title = extract_chapter(chunk["heading_path"])
    citation = build_citation(
        chunk["subject"], chapter_no, chapter_title, chunk["pages"]
    )
    return {
        **chunk,
        "chapter_no": chapter_no,
        "chapter_title": chapter_title,
        "section": chunk["heading_path"][-1] if chunk["heading_path"] else None,
        "page_range": format_pages(chunk["pages"]),
        "citation": citation,
        "keywords": extract_keywords(chunk["text"]),
        "embed_text": build_embed_text(
            chunk["subject"], chunk["heading_path"], chunk["text"]
        ),
    }


def run() -> list[dict]:
    chunks = read_jsonl("02_chunks")
    tagged = [tag_chunk(c) for c in chunks]

    missing_ch = sum(1 for t in tagged if not t["chapter_no"])
    missing_pg = sum(1 for t in tagged if not t["page_range"])
    log("tag", f"tagged {len(tagged)} chunks")
    if missing_ch:
        log("tag", f"{missing_ch} chunks have no chapter number (check heading detection)")
    if missing_pg:
        log("tag", f"{missing_pg} chunks have no page number (citations will be partial)")

    write_jsonl("03_tagged", tagged)
    return tagged


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
