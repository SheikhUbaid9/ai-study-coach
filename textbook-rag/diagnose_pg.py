"""Direct check of what's actually in the pgvector table and whether it searches.

    python diagnose_pg.py            uses the query "nociceptors"
    python diagnose_pg.py "torque"

Talks straight to Postgres with no chunking, embedding-cache or agent code in the
way, so it separates "the data is wrong" from "the pipeline around it is wrong".
"""

from __future__ import annotations

import sys

from common import to_vector_literal
from config import EMBED_DIM, PG_DSN, PG_TABLE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    query = " ".join(sys.argv[1:]) or "nociceptors"

    import psycopg

    print(f"table    : {PG_TABLE}")
    print(f"host     : {PG_DSN.rsplit('@', 1)[-1].split('/')[0]}")
    print(f"query    : {query!r}\n")

    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(f"SELECT count(*) AS n FROM {PG_TABLE}")
            total = cur.fetchone()["n"]
            print(f"rows in table            : {total}")
            if total == 0:
                print("\n!! Table is empty — re-run ingestion stage 5.")
                return 1

            # A NULL embedding is the classic reason a vector search returns
            # nothing while the rows plainly exist.
            cur.execute(f"SELECT count(*) AS n FROM {PG_TABLE} WHERE embedding IS NULL")
            print(f"rows with NULL embedding : {cur.fetchone()['n']}")

            cur.execute(
                f"SELECT DISTINCT subject, embed_model FROM {PG_TABLE}"
            )
            for row in cur.fetchall():
                print(f"subject / model          : {row['subject']!r} / {row['embed_model']}")

            cur.execute(
                f"SELECT vector_dims(embedding) AS d FROM {PG_TABLE} LIMIT 1"
            )
            dims = cur.fetchone()["d"]
            print(f"stored vector dimension  : {dims}  (config expects {EMBED_DIM})")
            if dims != EMBED_DIM:
                print("\n!! Dimension mismatch — the query vector can never match. "
                      "Re-embed and re-index.")
                return 1

            # keyword search
            cur.execute(
                f"""SELECT citation FROM {PG_TABLE}
                    WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)""",
                (query,),
            )
            kw = [r["citation"] for r in cur.fetchall()]
            print(f"\nkeyword matches          : {len(kw)}")
            for c in kw:
                print(f"   {c}")

            cur.execute(
                f"SELECT citation FROM {PG_TABLE} WHERE text ILIKE %s", (f"%{query}%",)
            )
            like = [r["citation"] for r in cur.fetchall()]
            print(f"plain ILIKE matches      : {len(like)}")

            # vector search
            print("\nembedding the query...")
            from s4_embed import embed_texts

            qvec = embed_texts([query], is_query=True)[0]
            cur.execute(
                f"""SELECT citation, 1 - (embedding <=> %s::vector) AS score
                    FROM {PG_TABLE}
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5""",
                (to_vector_literal(qvec), to_vector_literal(qvec)),
            )
            hits = cur.fetchall()
            print(f"vector matches           : {len(hits)}")
            for h in hits:
                print(f"   {h['score']:.3f}  {h['citation']}")

            if not hits:
                print("\n!! Vector search returned nothing on a non-empty table. "
                      "That points at the embedding column, not the query.")
                return 1

    print("\nLooks healthy — if the app still finds nothing, the problem is in the "
          "app's wiring (wrong backend selected, or a filter), not the database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
