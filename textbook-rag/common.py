"""Shared plumbing: stage logging, JSONL read/write, token counting.

Everything a stage produces goes to disk as JSONL so you can open it and read it.
That's deliberate — the whole point of this pipeline is being able to inspect the
output of each step and judge whether the quality is good before building on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from config import OUT_DIR

# Windows consoles default to a legacy codepage that mangles em-dashes and any
# non-ASCII text pulled out of a textbook.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def log(stage: str, msg: str) -> None:
    print(f"[{stage:>9}] {msg}")


def warn(stage: str, msg: str) -> None:
    print(f"[{stage:>9}] !! {msg}")


# --------------------------------------------------------------------------
# JSONL — one record per line, so a 5,000-chunk file is still greppable
# --------------------------------------------------------------------------

def stage_path(name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{name}.jsonl"


def write_jsonl(name: str, records: Iterable[dict[str, Any]]) -> Path:
    path = stage_path(name)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    log("io", f"wrote {n} records -> {path.relative_to(OUT_DIR.parent)}")
    return path


def read_jsonl(name: str) -> list[dict[str, Any]]:
    path = stage_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the earlier stage first (see run.py --help)"
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl(name: str) -> Iterator[dict[str, Any]]:
    with stage_path(name).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------
# Chunk sizes are specified in tokens because that's what the embedding model
# actually consumes. We use the real tokenizer when it's installed so the numbers
# in the quality report mean something; the /4 fallback keeps the pipeline
# runnable before anyone has downloaded a model.

_tokenizer = None
_tokenizer_tried = False


def _get_tokenizer():
    global _tokenizer, _tokenizer_tried
    if _tokenizer_tried:
        return _tokenizer
    _tokenizer_tried = True
    try:
        from transformers import AutoTokenizer

        from config import EMBED_MODEL

        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    except Exception:
        _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    if tok is None:
        return max(1, len(text) // 4)  # rough but stable fallback
    return len(tok.encode(text, add_special_tokens=False))


def using_real_tokenizer() -> bool:
    return _get_tokenizer() is not None


# --------------------------------------------------------------------------
# pgvector parameter binding
# --------------------------------------------------------------------------

def to_vector_literal(vec) -> str:
    """Render a vector as pgvector's text form: '[0.1,0.2,0.3]'.

    psycopg has no built-in adapter for numpy arrays, and relying on
    pgvector's register_vector() is fragile — it has to resolve the extension's
    type OID on the connection, and that doesn't reliably carry through
    executemany() across versions. Passing a text literal with an explicit
    ::vector cast in the SQL sidesteps all of it and works everywhere.

    No spaces: pgvector's input parser is happiest with a compact literal.
    """
    values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return "[" + ",".join(repr(float(v)) for v in values) + "]"
