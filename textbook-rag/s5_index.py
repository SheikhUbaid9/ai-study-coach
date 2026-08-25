"""Stage 5 — build the searchable index.

Two backends behind one interface, chosen by STORE_BACKEND in config.py:

* **file** (default) — vectors stay in the .npy from Stage 4, BM25 is built in
  memory at query time. Zero setup. At this scale that's not a compromise: a
  ~300-page textbook is roughly 500 chunks, so five books is ~2,500 vectors.
  Brute-force scoring over 2,500 rows is microseconds, and an approximate index
  (HNSW) would only trade exact recall for a speedup you cannot perceive.

* **pgvector** — the same data in Postgres, dense search via pgvector's HNSW and
  sparse search via Postgres' own full-text index. Not needed at this scale, but
  it's the same engine LangGraph uses for production persistence, so it's worth
  knowing. Choose it to learn it, not because the corpus demands it.

Output: out/05_index_manifest.json (what got indexed, where, with which model)
Inspect: python inspect.py index
"""

from __future__ import annotations

import json
import sys

from common import log, read_jsonl, warn
from config import EMBED_DIM, EMBED_MODEL, OUT_DIR, PG_DSN, PG_TABLE, STORE_BACKEND

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    chunk_id      text PRIMARY KEY,
    doc_id        text NOT NULL,
    book          text,
    subject       text,
    chapter_no    text,
    chapter_title text,
    section       text,
    page_range    text,
    citation      text,
    heading_path  text[],
    keywords      text[],
    text          text NOT NULL,
    embed_text    text NOT NULL,
    n_tokens      int,
    split_reason  text,
    embed_model   text NOT NULL,
    embedding     vector({dim}) NOT NULL
);

-- Dense: HNSW over cosine distance. Vectors are already L2-normalised.
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
    ON {table} USING hnsw (embedding vector_cosine_ops);

-- Sparse: Postgres' own full-text index, so hybrid search needs no second service.
CREATE INDEX IF NOT EXISTS {table}_fts_idx
    ON {table} USING gin (to_tsvector('english', text));

CREATE INDEX IF NOT EXISTS {table}_subject_idx ON {table} (subject);
"""


# --------------------------------------------------------------------------
# file backend
# --------------------------------------------------------------------------

def build_file_index(rows: list[dict]) -> dict:
    vec_path = OUT_DIR / "04_vectors.npy"
    if not vec_path.exists():
        raise FileNotFoundError(f"{vec_path} missing — run stage 4 first")
    log("index", f"file backend: {len(rows)} chunks, vectors at {vec_path.name}")
    return {
        "backend": "file",
        "n_chunks": len(rows),
        "vectors": vec_path.name,
        "metadata": "04_embedded.jsonl",
    }


# --------------------------------------------------------------------------
# pgvector backend
# --------------------------------------------------------------------------

def build_pg_index(rows: list[dict]) -> dict:
    import numpy as np
    import psycopg

    from common import to_vector_literal

    vectors = np.load(OUT_DIR / "04_vectors.npy")

    log("index", f"connecting to {PG_DSN.rsplit('@', 1)[-1]}...")
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL.format(table=PG_TABLE, dim=EMBED_DIM))
            conn.commit()

            # Re-ingesting a book should replace its chunks, never duplicate them.
            doc_ids = sorted({r["doc_id"] for r in rows})
            cur.execute(f"DELETE FROM {PG_TABLE} WHERE doc_id = ANY(%s)", (doc_ids,))
            log("index", f"cleared existing rows for {len(doc_ids)} doc(s)")

            cur.executemany(
                f"""INSERT INTO {PG_TABLE} (
                        chunk_id, doc_id, book, subject, chapter_no, chapter_title,
                        section, page_range, citation, heading_path, keywords,
                        text, embed_text, n_tokens, split_reason, embed_model, embedding
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector
                    )""",
                [
                    (
                        r["chunk_id"], r["doc_id"], r["book"], r["subject"],
                        r["chapter_no"], r["chapter_title"], r["section"],
                        r["page_range"], r["citation"], r["heading_path"],
                        r["keywords"], r["text"], r["embed_text"], r["n_tokens"],
                        r["split_reason"], r["embed_model"],
                        to_vector_literal(vectors[r["vector_row"]]),
                    )
                    for r in rows
                ],
            )
            conn.commit()
            cur.execute(f"SELECT count(*) FROM {PG_TABLE}")
            total = cur.fetchone()[0]

    log("index", f"pgvector: inserted {len(rows)} chunks, table now holds {total}")
    return {"backend": "pgvector", "n_chunks": len(rows), "table": PG_TABLE, "total_rows": total}


def test_connection() -> bool:
    """Verify the DB is reachable and pgvector is usable, before a long ingest.

    Worth doing separately: embedding a whole book and only then discovering the
    connection string is wrong wastes the slowest part of the pipeline.
    """
    try:
        import psycopg
    except ImportError:
        warn("index", "psycopg not installed — pip install 'psycopg[binary]' pgvector")
        return False

    host = PG_DSN.rsplit("@", 1)[-1].split("/")[0]
    log("index", f"connecting to {host} ...")
    try:
        with psycopg.connect(PG_DSN, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0].split(",")[0]
                log("index", f"connected: {version}")

                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                row = cur.fetchone()
                if not row:
                    warn("index", "pgvector extension is not available on this database")
                    return False
                log("index", f"pgvector {row[0]} ready")

                cur.execute(f"SELECT to_regclass('{PG_TABLE}')")
                existing = cur.fetchone()[0]
                if existing:
                    cur.execute(f"SELECT count(*) FROM {PG_TABLE}")
                    log("index", f"table '{PG_TABLE}' exists with {cur.fetchone()[0]} rows")
                else:
                    log("index", f"table '{PG_TABLE}' not created yet (stage 5 will create it)")
        log("index", "connection test PASSED")
        return True
    except Exception as exc:
        warn("index", f"connection FAILED: {type(exc).__name__}: {exc}")
        warn("index", "check PG_DSN. On Supabase use the *Session pooler* string "
                      "(port 5432) — the direct-connection host is IPv6-only on the "
                      "free tier and fails on many home networks.")
        return False


def run() -> dict:
    rows = read_jsonl("04_embedded")
    if not rows:
        warn("index", "nothing to index")
        return {}

    models = {r["embed_model"] for r in rows}
    if len(models) > 1:
        # Cosine similarity between vectors from two different models is
        # meaningless — and nothing raises an error, it just retrieves garbage.
        warn("index", f"MIXED EMBEDDING MODELS {models} — re-run stage 4 on everything")

    if STORE_BACKEND == "pgvector":
        manifest = build_pg_index(rows)
    else:
        manifest = build_file_index(rows)

    manifest["embed_model"] = EMBED_MODEL
    manifest["embed_dim"] = EMBED_DIM
    path = OUT_DIR / "05_index_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("index", f"manifest -> {path.name}")
    return manifest


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
