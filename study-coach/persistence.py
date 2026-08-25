"""Where the agent's memory actually lives.

Two backends behind one call:

* **memory** (default) - MemorySaver + InMemoryStore. Zero setup, and everything
  vanishes when the process stops. Fine while developing, useless once deployed:
  a hosted app sleeps and restarts constantly, so every student would lose their
  profile and task list several times a day.

* **postgres** - the same Supabase database the textbooks already use. Survives
  restarts, and is what makes deployment meaningful.

The graph itself is identical either way. That's the point the LangGraph course
makes about deployment: persistence is supplied *to* the graph, never baked into
it, so moving from laptop to server changes this file and nothing else.

Set MEMORY_BACKEND=postgres in .env to switch.
"""

from __future__ import annotations

import os

# Kept module-level so the connections aren't rebuilt on every Streamlit rerun.
_pool = None


def backend_name() -> str:
    return os.environ.get("MEMORY_BACKEND", "memory").lower()


def _postgres_dsn() -> str | None:
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        return None
    # LangGraph's Postgres saver runs multi-statement setup and prepared
    # statements; Supabase's *transaction* pooler (port 6543) breaks both.
    # Warn rather than fail, since some setups do work.
    if ":6543" in dsn:
        print("[persistence] WARNING: port 6543 is the transaction pooler and "
              "usually fails here. Use the session pooler (5432).")
    return dsn


def get_persistence():
    """Return (checkpointer, store, description).

    Falls back to in-memory on any Postgres problem rather than refusing to
    start — a student locked out of their coach because a database hiccuped is
    worse than one running with temporary memory.
    """
    global _pool

    if backend_name() != "postgres":
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore

        return MemorySaver(), InMemoryStore(), "in-memory (resets on restart)"

    dsn = _postgres_dsn()
    if not dsn:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore

        return MemorySaver(), InMemoryStore(), "in-memory (PG_DSN not set)"

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        from psycopg_pool import ConnectionPool

        if _pool is None:
            _pool = ConnectionPool(
                conninfo=dsn,
                max_size=5,
                # autocommit is required: the setup() calls below run CREATE
                # statements that can't sit inside a transaction block.
                kwargs={"autocommit": True, "prepare_threshold": None},
                open=True,
            )

        checkpointer = PostgresSaver(_pool)
        store = PostgresStore(_pool)
        # Idempotent - creates the tables on first run, no-ops afterwards.
        checkpointer.setup()
        store.setup()

        return checkpointer, store, "postgres (survives restarts)"

    except Exception as exc:
        print(f"[persistence] Postgres unavailable ({type(exc).__name__}: {exc})")
        print("[persistence] falling back to in-memory for this run")
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore

        return MemorySaver(), InMemoryStore(), f"in-memory (postgres failed: {type(exc).__name__})"
