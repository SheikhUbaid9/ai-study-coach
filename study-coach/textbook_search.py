"""Bridge from Study Coach to the textbook-rag pipeline next door.

The two stay separate projects (textbook-rag has its own deps, its own config,
and is independently runnable), but Study Coach reaches into it for retrieval so
the student only ever opens one app.

This is deliberately a *soft* dependency. If textbook-rag isn't installed, or no
book has been ingested yet, `is_available()` returns False and Study Coach simply
doesn't offer the search tool — it keeps working exactly as it did before. A
missing textbook index should never break the chat.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAG_DIR = Path(__file__).parent.parent / "textbook-rag"

_status: str | None = None


def _ensure_path() -> bool:
    if not RAG_DIR.exists():
        return False
    # Appended, not prepended: Study Coach's own modules must keep priority.
    # (Both projects have an app.py, so import order matters here.)
    if str(RAG_DIR) not in sys.path:
        sys.path.append(str(RAG_DIR))
    return True


def is_available() -> bool:
    """True only if the pipeline is importable AND something has been ingested."""
    global _status
    if not _ensure_path():
        _status = "textbook-rag folder not found"
        return False
    try:
        from config import OUT_DIR, STORE_BACKEND
    except ImportError as exc:
        _status = f"textbook-rag not importable ({exc})"
        return False

    if STORE_BACKEND == "pgvector":
        _status = "pgvector backend"
        return True

    if not (OUT_DIR / "04_embedded.jsonl").exists():
        _status = "no books ingested yet"
        return False
    _status = "file backend"
    return True


def status() -> str:
    if _status is None:
        is_available()
    return _status or "unknown"


def ingested_doc_ids() -> set[str]:
    """Which documents are actually searchable.

    Has to ask whichever store is in use, not the local files. On a deployed
    app there IS no local `out/` directory — the books were ingested on someone's
    laptop and pushed to Postgres — so reading the file would report every book
    as un-ingested while search works perfectly. Two different answers to the
    same question is worse than either one alone.
    """
    if not _ensure_path():
        return set()

    from config import STORE_BACKEND

    if STORE_BACKEND == "pgvector":
        try:
            import psycopg

            from config import PG_DSN, PG_TABLE

            with psycopg.connect(PG_DSN, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT DISTINCT doc_id FROM {PG_TABLE}")
                    return {row[0] for row in cur.fetchall()}
        except Exception as exc:
            print(f"[textbook_search] couldn't list ingested docs: {exc}")
            return set()

    try:
        from common import read_jsonl

        return {r["doc_id"] for r in read_jsonl("04_embedded")}
    except Exception:
        return set()


def library() -> list[dict]:
    """What's on disk and what's been ingested — for the UI's Books tab."""
    if not _ensure_path():
        return []
    from config import BOOKS_DIR

    ingested = ingested_doc_ids()

    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    books = [
        {
            "name": p.name,
            "path": p,
            "size_mb": p.stat().st_size / 1e6,
            "ingested": p.stem in ingested,
            "local": True,
        }
        for p in sorted(BOOKS_DIR.glob("*.pdf"))
    ]

    # Books ingested elsewhere. On a deployed app this is usually ALL of them:
    # the PDFs sit on the laptop that ingested them, while the searchable chunks
    # live in Postgres. Listing only local files would show an empty shelf while
    # the coach happily quotes those same books.
    local_stems = {b["path"].stem for b in books}
    for doc_id in sorted(ingested - local_stems):
        books.append({
            "name": f"{doc_id}.pdf",
            "path": None,
            "size_mb": 0.0,
            "ingested": True,
            "local": False,
        })

    return books


def subjects() -> list[str]:
    if not _ensure_path():
        return []
    try:
        from common import read_jsonl

        return sorted({r["subject"] for r in read_jsonl("04_embedded")})
    except Exception:
        return []


last_error: str | None = None


def search(query: str, subject: str | None = None, top_k: int = 4) -> list[dict]:
    """Retrieve passages from the student's books.

    Returns [] on failure so the chat degrades gracefully rather than crashing —
    but the reason is recorded in `last_error` and printed. Swallowing the
    exception silently made a broken database look identical to a book that
    simply didn't mention the topic, which is the worst possible ambiguity here.
    """
    global last_error
    last_error = None

    if not is_available():
        last_error = f"textbook search unavailable: {status()}"
        print(f"[textbook_search] {last_error}")
        return []

    _ensure_path()
    try:
        from s6_retrieve import retrieve

        results = retrieve(query, top_k=top_k, subject=subject)
        if not results:
            last_error = "retrieval ran but matched nothing (is the index populated?)"
            print(f"[textbook_search] {last_error}")
        return results
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
        print(f"[textbook_search] RETRIEVAL FAILED — {last_error}")
        return []
