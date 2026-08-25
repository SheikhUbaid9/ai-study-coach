"""CLI driver — run the ingestion stages, or ask a question.

    python run.py ingest              # stages 1-5, full ingestion
    python run.py ingest --from 2     # re-run from chunking (skip slow parsing)
    python run.py ask "what is torque?"
    python run.py ask "..." --subject Physics
    python run.py retrieve "..."      # retrieval only, no LLM, no HF_TOKEN needed

Stages are separate commands on purpose. Parsing is by far the slowest step, so
when you're tuning chunk size — which is the parameter you'll actually iterate on
— you re-run from stage 2 and reuse the parsed blocks.
"""

from __future__ import annotations

import argparse
import sys
import time

from common import log, warn

STAGES = [
    ("1", "parse", "PDF -> structured blocks (Docling)"),
    ("2", "chunk", "blocks -> heading-scoped chunks"),
    ("3", "tag", "chunks -> citations + embed text"),
    ("4", "embed", "chunks -> vectors (bge-small)"),
    ("5", "index", "vectors -> searchable index"),
]


def run_ingest(start_from: int) -> int:
    t0 = time.time()
    for num, name, desc in STAGES:
        if int(num) < start_from:
            log("run", f"skipping stage {num} ({name})")
            continue

        print(f"\n{'─' * 72}\nSTAGE {num} — {name.upper()}: {desc}\n{'─' * 72}")
        module = __import__(f"s{num}_{name}")
        t = time.time()
        try:
            result = module.run()
        except FileNotFoundError as exc:
            warn("run", str(exc))
            return 1
        except Exception as exc:
            warn("run", f"stage {num} ({name}) failed: {type(exc).__name__}: {exc}")
            return 1

        log("run", f"stage {num} done in {time.time() - t:.1f}s")
        if not result:
            warn("run", f"stage {num} produced nothing — stopping.")
            return 1

    print(f"\n{'─' * 72}")
    log("run", f"ingestion complete in {time.time() - t0:.1f}s")
    log("run", "inspect the output:  python inspect_stage.py all")
    return 0


def run_ask(query: str, subject: str | None, retrieve_only: bool) -> int:
    from s6_retrieve import run as retrieve_run

    results = retrieve_run(query, subject)
    if not results:
        warn("run", "nothing retrieved — has ingestion run?")
        return 1

    if retrieve_only:
        print()
        for r in results:
            print(f"{'─' * 72}\n#{r['rank']}  {r['citation']}   "
                  f"[found by: {'+'.join(r['found_by'])}]  rrf={r['rrf_score']}")
            print(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))
        print("─" * 72)
        return 0

    from s7_generate import run as generate_run

    out = generate_run(query, results)
    return 1 if out.get("error") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="run ingestion stages 1-5")
    p_ing.add_argument("--from", dest="start", type=int, default=1,
                       help="start from this stage (default 1)")

    p_ask = sub.add_parser("ask", help="retrieve + generate an answer")
    p_ask.add_argument("query", nargs="+")
    p_ask.add_argument("--subject", help="restrict to one subject, e.g. Physics")

    p_ret = sub.add_parser("retrieve", help="retrieval only, no LLM call")
    p_ret.add_argument("query", nargs="+")
    p_ret.add_argument("--subject")

    sub.add_parser("testdb", help="check the Postgres/pgvector connection")

    args = parser.parse_args()

    if args.cmd == "testdb":
        from s5_index import test_connection

        return 0 if test_connection() else 1
    if args.cmd == "ingest":
        return run_ingest(args.start)
    return run_ask(" ".join(args.query), args.subject, args.cmd == "retrieve")


if __name__ == "__main__":
    sys.exit(main())
