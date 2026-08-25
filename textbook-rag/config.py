"""Every tunable in one place.

The reference doc's most-repeated finding is that there is no universally correct
chunk size / threshold / top-k — those are per-corpus empirical questions. So the
numbers below are *starting points to tune*, not settled defaults. Change one at a
time, re-run, and inspect the stage output (see inspect.py).
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Where things live
# --------------------------------------------------------------------------
ROOT = Path(__file__).parent
BOOKS_DIR = ROOT / "books"          # drop your FSc PDFs here
OUT_DIR = ROOT / "out"              # every stage dumps its output here, inspectable

# --------------------------------------------------------------------------
# Stage 1 — parsing
# --------------------------------------------------------------------------
# Docling runs OCR only when a page has no usable text layer. Forcing OCR on a
# book that already has real text wastes a lot of time for worse results, so we
# let Docling decide per page rather than switching it on globally.
OCR_ENABLED = True                  # allow OCR fallback for scanned pages
OCR_LANGUAGES = ["eng"]             # add "urd" if your books have Urdu text
DO_TABLE_STRUCTURE = True           # recover table cell structure, not just text

# A page whose extracted text is shorter than this is treated as "suspiciously
# empty" and flagged in the quality report — usually means a scan that OCR missed,
# or a full-page diagram.
MIN_CHARS_PER_PAGE = 80

# Parse only the first N pages. Set this to ~10 for a trial run on a scanned book:
# OCR takes several seconds per page, so a 258-page scan is a 20-60 minute job, and
# you do NOT want to discover the OCR output is garbage at the end of it. Check the
# stage-1 quality report on a 10-page sample first, then set this back to None.
PAGE_LIMIT = int(os.environ["PAGE_LIMIT"]) if os.environ.get("PAGE_LIMIT") else None

# --------------------------------------------------------------------------
# Stage 2 — chunking
# --------------------------------------------------------------------------
# Structure-aware first: a chunk is one heading's section. Only sections that
# blow past MAX_TOKENS get split further — that's the "semantic only as fallback"
# decision, so we don't pay embedding cost on sections that are already fine.
# The ceiling is dictated by the EMBEDDING MODEL, not by taste: bge-small-en-v1.5
# has a 512-token maximum sequence length and silently TRUNCATES anything longer —
# no error, you just lose the tail of the chunk. 480 leaves headroom for the
# special tokens and the heading prefix added in stage 3. If you switch to a model
# with a different limit, change this too.
TARGET_TOKENS = 320                 # ideal chunk size
MAX_TOKENS = 480                    # hard ceiling — must stay under the model's 512
MIN_TOKENS = 60                     # below this, merge into the neighbouring chunk
OVERLAP_TOKENS = 60                 # ~15% of target, per the reference doc's heuristic

# When an oversized section must be split, use embedding-similarity breakpoints
# rather than blind character counts. Costs one embedding pass over that section
# only. Set False to always fall back to plain recursive splitting.
SEMANTIC_SPLIT_ENABLED = True
SEMANTIC_BREAKPOINT_PERCENTILE = 90  # split where similarity drops into the bottom 10%

# --------------------------------------------------------------------------
# Stage 4 — embedding
# --------------------------------------------------------------------------
# bge-small-en-v1.5: 384-dim, ~130MB, runs fine on CPU, scores better on retrieval
# than all-MiniLM-L6-v2 at the same size/speed. Swap freely — but if you do, you
# MUST re-embed everything (see the reference doc: mixing embeddings from two
# models in one index is silently broken, not an error).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 32
# bge models were trained with this instruction prefix on the *query* side only.
# Deviating from the model's trained format measurably hurts retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------
# Stage 5 — storage
# --------------------------------------------------------------------------
# "file" keeps everything in out/ as .npy + .jsonl — zero setup, fine at this scale.
# "pgvector" needs a running Postgres with the pgvector extension.
STORE_BACKEND = os.environ.get("STORE_BACKEND", "file")   # "file" | "pgvector"

# A Postgres connection string contains a password, so it is read from the
# environment and never committed. Works the same for a local server or a free
# cloud Postgres (Neon/Supabase) — it's just a different URL.
#   PowerShell:  $env:PG_DSN = "postgresql://user:pass@host/db?sslmode=require"
PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/textbook_rag")
PG_TABLE = "chunks"

# --------------------------------------------------------------------------
# Stage 6 — retrieval
# --------------------------------------------------------------------------
# Hybrid dense + BM25 fused with Reciprocal Rank Fusion. k=60 is the constant from
# the original 2009 RRF paper and has held up as a robust default for ~two decades.
RRF_K = 60
CANDIDATES_PER_RETRIEVER = 20       # how deep each retriever goes before fusion
TOP_K = 5                           # how many chunks actually reach the LLM

# --------------------------------------------------------------------------
# Stage 7 — generation
# --------------------------------------------------------------------------
# Reuses the same free Hugging Face router setup as the study-coach app.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GEN_TEMPERATURE = 0.0               # factual grounding, not creativity
