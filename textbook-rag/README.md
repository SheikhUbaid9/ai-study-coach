# textbook-rag — an inspectable RAG ingestion pipeline for FSc books

Drop your FSc textbook PDFs in `books/`, run one command, and ask questions that
get answered from those books with a chapter-and-page citation.

Built stage-by-stage on purpose: **every stage writes its output to `out/` as
JSONL, and every stage has a quality report.** Bad RAG ingestion almost never
throws an error — it silently retrieves the wrong thing forever. So the pipeline
is designed to be opened up and judged at each step, not treated as a black box.

## Setup

```bash
cd textbook-rag
pip install -r requirements.txt
```

## The easy way — the web UI

```bash
streamlit run app.py
```

Three tabs:

- **Library** — drag PDFs in, see which are ingested, delete ones you don't want.
- **Ingest** — run the pipeline with a progress log per stage, and read the
  quality report right underneath. A stage that reports `BAD` opens expanded by
  default, because that's the one you need to see.
- **Ask** — search your books. Leave the LLM checkbox off to see raw retrieved
  chunks with no API key at all; tick it for a written, cited answer.

Name files so the subject is detectable — `fsc-physics-part1.pdf`,
`fsc-chemistry-part2.pdf` — since Stage 3 reads the subject off the filename.

## The CLI way

```bash
python run.py ingest
```

Then inspect what it produced before trusting it:

```bash
python inspect_stage.py all
```

Then ask questions:

```bash
python run.py retrieve "explain torque"
```

`retrieve` needs no API key — it shows you the raw chunks retrieved and which
retriever found each one. To get a written answer, set a free Hugging Face token
(`$env:HF_TOKEN = "hf_..."` — see the study-coach README) and use:

```bash
python run.py ask "explain torque" --subject Physics
```

Re-running after a config change: parsing is by far the slowest stage, so reuse it.

```bash
python run.py ingest --from 2
```

## The stages, and why each is built the way it is

| # | Stage | Tool | The decision that matters |
|---|---|---|---|
| 1 | Parse | **Docling** | Keeps *structure* — headings, tables, page numbers — not just a wall of text. OCR fires only on pages with no text layer; forcing OCR on a digital book is slower and worse. |
| 2 | Chunk | structure-aware, semantic fallback | A textbook already has human-authored boundaries (chapters, numbered headings). Guessing statistically when real ones are printed on the page is strictly worse. Only oversized sections get split further, and only *those* pay for semantic splitting. |
| 3 | Tag | regex + frequency | Builds the citation (`Physics — Ch 12: Electrostatics, p.84-85`) and prepends the heading path to the text before embedding, so a chunk reading "It is measured in newton-metres" still lands near "torque". |
| 4 | Embed | **bge-small-en-v1.5** | 384-dim, ~130MB, CPU-fine, better at retrieval than MiniLM at the same size. Local and free — no API key. |
| 5 | Index | file *or* **pgvector** | See below. |
| 6 | Retrieve | dense + BM25, fused with **RRF** | Dense matches meaning ("how does a lens bend light" → "Refraction"); sparse matches exact rare tokens (formula names, "Le Chatelier"). Textbook questions need both. RRF merges on *rank*, not score, because a cosine of 0.82 and a BM25 of 14.3 aren't comparable numbers. |
| 7 | Generate | HF router (same model as study-coach) | Sources are pre-numbered `[1] [2]`; the model cites slots, and the real page is resolved in code. A model that can't write a page number can't get one wrong. |

## Storage: `file` vs `pgvector`

Default is `file` — vectors in a `.npy`, BM25 built in memory. **At this scale
that isn't a compromise:** a 300-page textbook is ~500 chunks, so five books is
~2,500 vectors. Brute-force scoring over 2,500 rows takes microseconds; an
approximate index would trade exact recall for a speedup you cannot perceive.

Use `pgvector` because you want to learn it (it's the same engine LangGraph uses
for production persistence), not because the corpus demands it.

### Running Postgres in the cloud, with no local install

Neither Postgres nor Docker needs to be installed locally. Free managed Postgres
with pgvector already enabled:

- **Neon** — https://neon.tech (free tier, pgvector built in)
- **Supabase** — https://supabase.com (free tier, pgvector available)

You create the account and copy the connection string yourself — then:

```powershell
$env:PG_DSN = "postgresql://user:pass@host:5432/postgres"
$env:STORE_BACKEND = "pgvector"
```

Check it before running a real ingest — embedding a whole book and *then*
discovering a bad connection string wastes the slowest part of the pipeline:

```bash
python run.py testdb
```

That verifies the database is reachable, enables the `vector` extension, and
reports whether the table already exists. Once it passes:

```bash
python run.py ingest --from 5
```

Stage 5 creates the table, the HNSW index, and the full-text index on first run,
and re-ingesting a book replaces its rows rather than duplicating them.

**On Supabase specifically:** use the **Session pooler** connection string
(port 5432), found under *Project Settings → Database → Connection string*. The
direct-connection host is IPv6-only on the free tier and fails on many home
networks. Avoid the *Transaction* pooler (port 6543) — it doesn't play well with
psycopg3's prepared statements. Replace `[YOUR-PASSWORD]` in the string with your
actual database password.

The connection string is read from the environment and never written to a file —
it contains a password.

## Reading the quality reports

`inspect_stage.py` prints `OK` / `CHECK` / `BAD` per check, with what to do about
it. The checks that catch the most damage:

- **No headings detected** (stage 1) → structure-aware chunking silently degrades
  into size-based blobs. Usually means a scanned book.
- **Most pages near-empty** (stage 1) → OCR didn't fire or Tesseract is missing.
  You think you ingested the book; you ingested nothing.
- **Low share of `structure` chunks** (stage 2) → heading detection is failing
  upstream; fix stage 1 rather than tuning chunk size.
- **Chunks over MAX_TOKENS** (stage 2) → the embedding model will truncate them.
  Content is lost with no error.
- **Missing page numbers** (stage 3) → citations degrade, and this *cannot* be
  recovered later; it has to come from stage 1 provenance.
- **Mixed embedding models** (stage 4) → the one genuinely silent killer.
  Similarity between vectors from two different models is meaningless, and
  nothing raises an error. Changing `EMBED_MODEL` means re-embedding everything.

## Tuning

Everything tunable is in `config.py`, with the honest caveat that there is no
universally correct chunk size or top-k — those are per-corpus empirical
questions. Change **one** parameter, re-run from stage 2, re-inspect, compare.
Changing several at once tells you nothing about which one helped.

## Not built yet

- **Cross-encoder reranking** — worth adding if you find the right chunk often
  lands at rank 6-10 rather than 1-3. Check `out/06_retrieved.jsonl` first;
  if the answer is usually already in the top 3, reranking buys you nothing.
- **RAGAS evaluation** — once there's a real corpus, build ~20 exam-style
  questions per subject as a golden set, then measure.
- **Diagram/figure handling** — a real gap for physics and chemistry. Diagrams
  currently contribute nothing; they'd need a vision model to caption them.
- **Wiring into study-coach** — the intended payoff: expose `s6_retrieve.retrieve`
  as a tool the coach node can call, so answers come from the student's own books.
