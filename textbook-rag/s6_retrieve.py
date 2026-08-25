"""Stage 6 — hybrid retrieval: dense + sparse, fused with Reciprocal Rank Fusion.

Why hybrid rather than picking one:

* **Dense** (embeddings) matches meaning. "How does a lens bend light?" finds a
  section titled "Refraction" that shares no keywords with the question.
* **Sparse** (BM25 / Postgres full-text) matches exact rare tokens. Formula
  names, units, "Le Chatelier", "sin θ" — precisely the vocabulary dense
  embeddings blur together.

Each alone leaves recall on the table; a published benchmark put hybrid at 91%
recall@10 against 78% dense-only and 65% BM25-only. Textbook queries mix both
kinds constantly, which is why this isn't a close call here.

**RRF** merges the two ranked lists by *rank position*, never raw score —
sidestepping the fact that a cosine similarity of 0.82 and a BM25 score of 14.3
aren't on comparable scales and can't be sensibly averaged:

    score(chunk) = Σ  1 / (k + rank_in_list_i)        k = 60

k=60 is the constant from the original 2009 RRF paper; it has held up as a robust
default for ~two decades.

Output (when run directly): out/06_retrieved.jsonl — the retrieved set for a
query, with per-retriever ranks kept so you can see *why* each chunk surfaced.
Inspect: python run.py ask "your question"
"""

from __future__ import annotations

import re
import sys

from common import log, read_jsonl, warn, write_jsonl
from config import (
    CANDIDATES_PER_RETRIEVER,
    OUT_DIR,
    PG_DSN,
    PG_TABLE,
    RRF_K,
    STORE_BACKEND,
    TOP_K,
)

_cache: dict = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# --------------------------------------------------------------------------
# file backend
# --------------------------------------------------------------------------

def _load_file_index():
    if "file" not in _cache:
        import numpy as np
        from rank_bm25 import BM25Okapi

        rows = read_jsonl("04_embedded")
        vectors = np.load(OUT_DIR / "04_vectors.npy")
        bm25 = BM25Okapi([_tokenize(r["embed_text"]) for r in rows])
        _cache["file"] = (rows, vectors, bm25)
        log("retrieve", f"loaded {len(rows)} chunks into memory")
    return _cache["file"]


def _search_file(query: str, n: int, subject: str | None):
    import numpy as np

    from s4_embed import embed_texts

    rows, vectors, bm25 = _load_file_index()

    # Subject filter is applied *before* scoring — filtering after the fact
    # silently collapses recall when the filter is selective.
    allowed = (
        [i for i, r in enumerate(rows) if r["subject"].lower() == subject.lower()]
        if subject
        else list(range(len(rows)))
    )
    if not allowed:
        warn("retrieve", f"no chunks for subject={subject!r}; searching everything")
        allowed = list(range(len(rows)))

    qvec = embed_texts([query], is_query=True)[0]
    dense_scores = vectors[allowed] @ qvec          # normalised -> dot == cosine
    dense_order = np.argsort(-dense_scores)[:n]
    dense_ranked = [(allowed[i], float(dense_scores[i])) for i in dense_order]

    sparse_all = bm25.get_scores(_tokenize(query))
    sparse_scores = np.array([sparse_all[i] for i in allowed])
    sparse_order = np.argsort(-sparse_scores)[:n]
    sparse_ranked = [(allowed[i], float(sparse_scores[i])) for i in sparse_order]

    return rows, dense_ranked, sparse_ranked


# --------------------------------------------------------------------------
# pgvector backend
# --------------------------------------------------------------------------

def _search_pg(query: str, n: int, subject: str | None):
    import psycopg

    from common import to_vector_literal
    from s4_embed import embed_texts

    qvec = embed_texts([query], is_query=True)[0]
    params = {"q": to_vector_literal(qvec), "n": n, "text": query}

    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            # An empty table and a genuine no-match look identical downstream, so
            # separate them here — "stage 5 never landed" is by far the more common
            # cause and needs a different fix than "tune the retriever".
            cur.execute(f"SELECT count(*) AS n FROM {PG_TABLE}")
            n_rows = cur.fetchone()["n"]
            if n_rows == 0:
                warn("retrieve", f"table '{PG_TABLE}' is EMPTY — nothing was indexed. "
                                 "Re-run stage 5 (it failed earlier if you saw the "
                                 "ndarray error).")
                return [], [], []

            # A subject filter that matches nothing must NOT silently return zero
            # results — that reads downstream as "your book doesn't cover this",
            # which is a lie when the passage is sitting right there under a
            # different subject label. Check the filter before trusting it.
            where = ""
            if subject:
                cur.execute(
                    f"SELECT count(*) AS n FROM {PG_TABLE} WHERE subject ILIKE %s",
                    (subject,),
                )
                if cur.fetchone()["n"] == 0:
                    warn(
                        "retrieve",
                        f"no chunks have subject={subject!r} — ignoring the filter and "
                        "searching everything (check your PDF filenames)",
                    )
                else:
                    where = "WHERE subject ILIKE %(subject)s"
                    params["subject"] = subject

            log("retrieve", f"searching {n_rows} chunks in pgvector"
                            + (f" filtered to subject={subject!r}" if where else ""))

            cur.execute(
                f"""SELECT chunk_id, doc_id, book, subject, citation, section,
                           chapter_no, page_range, text, n_tokens,
                           1 - (embedding <=> %(q)s::vector) AS score
                    FROM {PG_TABLE} {where}
                    ORDER BY embedding <=> %(q)s::vector
                    LIMIT %(n)s""",
                params,
            )
            dense = cur.fetchall()

            cur.execute(
                f"""SELECT chunk_id, doc_id, book, subject, citation, section,
                           chapter_no, page_range, text, n_tokens,
                           ts_rank(to_tsvector('english', text),
                                   plainto_tsquery('english', %(text)s)) AS score
                    FROM {PG_TABLE}
                    {where + ' AND' if where else 'WHERE'}
                        to_tsvector('english', text)
                        @@ plainto_tsquery('english', %(text)s)
                    ORDER BY score DESC
                    LIMIT %(n)s""",
                params,
            )
            sparse = cur.fetchall()

    # Log both counts separately. "Found nothing" has very different causes
    # depending on which search came back empty: an unfiltered vector search over
    # a non-empty table should ALWAYS return rows (it's just ORDER BY distance
    # LIMIT n), so dense=0 means something structural is wrong, whereas sparse=0
    # just means none of the query's words appear literally in the text.
    log("retrieve", f"dense={len(dense)} sparse={len(sparse)}")
    if not dense:
        warn("retrieve", "vector search returned 0 rows on a non-empty table — "
                         "check that the embedding column is populated and its "
                         "dimension matches the query vector")

    rows_by_id = {r["chunk_id"]: r for r in dense + sparse}
    rows = list(rows_by_id.values())
    idx = {cid: i for i, cid in enumerate(rows_by_id)}

    def ranked(hits):
        # score can come back NULL from ts_rank on an empty tsquery; treat that
        # as zero rather than letting float(None) blow up the whole retrieval.
        return [
            (idx[h["chunk_id"]], float(h["score"]) if h["score"] is not None else 0.0)
            for h in hits
        ]

    return rows, ranked(dense), ranked(sparse)


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def reciprocal_rank_fusion(ranked_lists: dict[str, list[tuple[int, float]]]) -> dict[int, dict]:
    """Merge ranked lists on rank position. Returns {row_index: {...}}."""
    fused: dict[int, dict] = {}
    for name, ranked in ranked_lists.items():
        for rank, (row_idx, raw) in enumerate(ranked, start=1):
            entry = fused.setdefault(
                row_idx, {"rrf_score": 0.0, "ranks": {}, "raw_scores": {}}
            )
            entry["rrf_score"] += 1.0 / (RRF_K + rank)
            entry["ranks"][name] = rank
            entry["raw_scores"][name] = round(raw, 4)
    return fused


def retrieve(query: str, top_k: int = TOP_K, subject: str | None = None) -> list[dict]:
    search = _search_pg if STORE_BACKEND == "pgvector" else _search_file
    rows, dense_ranked, sparse_ranked = search(query, CANDIDATES_PER_RETRIEVER, subject)

    fused = reciprocal_rank_fusion({"dense": dense_ranked, "sparse": sparse_ranked})
    ordered = sorted(fused.items(), key=lambda kv: -kv[1]["rrf_score"])[:top_k]

    results = []
    for final_rank, (row_idx, meta) in enumerate(ordered, start=1):
        row = rows[row_idx]
        results.append(
            {
                "rank": final_rank,
                "chunk_id": row["chunk_id"],
                "citation": row["citation"],
                "subject": row["subject"],
                "section": row.get("section"),
                "text": row["text"],
                "n_tokens": row.get("n_tokens"),
                "rrf_score": round(meta["rrf_score"], 6),
                # Kept deliberately: seeing that a chunk ranked #1 dense but
                # didn't appear in sparse at all tells you which retriever is
                # actually earning its place for this kind of question.
                "found_by": sorted(meta["ranks"]),
                "ranks": meta["ranks"],
                "raw_scores": meta["raw_scores"],
            }
        )
    return results


def run(query: str, subject: str | None = None) -> list[dict]:
    results = retrieve(query, TOP_K, subject)
    log("retrieve", f"{len(results)} chunks for: {query!r}")
    for r in results:
        log("retrieve", f"  #{r['rank']} [{'+'.join(r['found_by'])}] {r['citation']}")
    write_jsonl("06_retrieved", [{"query": query, **r} for r in results])
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python s6_retrieve.py "your question"')
        sys.exit(1)
    run(" ".join(sys.argv[1:]))
