"""Upload + ingest + inspect UI for the textbook RAG pipeline.

    streamlit run app.py

Three tabs, matching how you actually work with this:
  Library — drop PDFs in, see what's ingested
  Ingest  — run the stages, watch the log, read the quality report
  Ask     — search the books, with or without an LLM

The quality report is on the same screen as the ingest button on purpose. The
failure mode this pipeline is built to avoid is ingesting a book badly and never
noticing, so the verdicts are put in front of you rather than left in a file.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from common import read_jsonl  # noqa: E402
from config import BOOKS_DIR, EMBED_MODEL, OUT_DIR, STORE_BACKEND  # noqa: E402

st.set_page_config(page_title="Textbook RAG", page_icon="📚", layout="wide")

STAGES = [
    ("1", "parse", "PDF → structured blocks (Docling)"),
    ("2", "chunk", "blocks → heading-scoped chunks"),
    ("3", "tag", "chunks → citations + embed text"),
    ("4", "embed", "chunks → vectors"),
    ("5", "index", "vectors → searchable index"),
]


def capture(fn, *args, **kwargs):
    """Run a stage and return (result, printed_output, error)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result = fn(*args, **kwargs)
        return result, buf.getvalue(), None
    except Exception as exc:  # surfaced in the UI rather than crashing the app
        return None, buf.getvalue(), f"{type(exc).__name__}: {exc}"


def stage_counts() -> dict[str, int]:
    counts = {}
    for name in ["01_blocks", "02_chunks", "03_tagged", "04_embedded"]:
        try:
            counts[name] = len(read_jsonl(name))
        except FileNotFoundError:
            counts[name] = 0
    return counts


# --------------------------------------------------------------------------
# Sidebar — status
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Status")
    st.caption(f"**Store:** `{STORE_BACKEND}`")
    st.caption(f"**Embed model:** `{EMBED_MODEL}`")

    if STORE_BACKEND == "pgvector":
        if st.button("Test database connection"):
            from s5_index import test_connection

            ok, out, err = capture(test_connection)
            st.code(out or err or "", language="text")
            if ok:
                st.success("Connected")
            else:
                st.error("Connection failed — see log above")

    st.divider()
    counts = stage_counts()
    st.caption("**Pipeline state**")
    for label, key in [
        ("Blocks", "01_blocks"), ("Chunks", "02_chunks"),
        ("Tagged", "03_tagged"), ("Embedded", "04_embedded"),
    ]:
        st.write(f"{label}: **{counts[key]}**")

    if any(counts.values()):
        if st.button("Clear all pipeline output", type="secondary"):
            shutil.rmtree(OUT_DIR, ignore_errors=True)
            st.rerun()


st.title("📚 Textbook RAG")
st.caption("Upload your FSc textbooks, ingest them, and check the quality before trusting them.")

tab_lib, tab_ingest, tab_ask = st.tabs(["Library", "Ingest", "Ask"])


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------

with tab_lib:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)

    st.subheader("Add a textbook")
    uploads = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Name files so the subject is detectable — e.g. fsc-physics-part1.pdf — "
             "since the subject is read off the filename.",
    )

    if uploads:
        saved, skipped = [], []
        for up in uploads:
            dest = BOOKS_DIR / up.name
            if dest.exists():
                skipped.append(up.name)
                continue
            dest.write_bytes(up.getbuffer())
            saved.append(f"{up.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        for s in saved:
            st.success(f"Saved {s}")
        for s in skipped:
            # Overwriting silently would be worse: you'd lose a book and not know.
            st.warning(f"{s} already exists — rename it, or delete the existing copy below.")
        if saved:
            st.info("Now open the **Ingest** tab to process these.")

    st.divider()
    st.subheader("Books on disk")
    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))

    if not pdfs:
        st.info("No PDFs yet. Upload one above.")
    else:
        try:
            ingested = {r["doc_id"] for r in read_jsonl("04_embedded")}
        except FileNotFoundError:
            ingested = set()

        for pdf in pdfs:
            c1, c2, c3, c4 = st.columns([5, 2, 2, 1])
            c1.write(f"**{pdf.name}**")
            c2.write(f"{pdf.stat().st_size / 1e6:.1f} MB")
            c3.write("✅ ingested" if pdf.stem in ingested else "⏳ not ingested")
            if c4.button("Delete", key=f"del_{pdf.name}"):
                pdf.unlink()
                st.rerun()

        if pdfs and not ingested:
            st.warning("No book has been ingested yet — open the **Ingest** tab.")


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

with tab_ingest:
    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))
    if not pdfs:
        st.info("Upload a PDF in the **Library** tab first.")
    else:
        st.write(f"**{len(pdfs)}** PDF(s) ready: " + ", ".join(p.name for p in pdfs))

        col1, col2 = st.columns([1, 3])
        start_from = col2.selectbox(
            "Start from stage",
            options=[1, 2, 3, 4, 5],
            format_func=lambda n: f"{n} — {STAGES[n - 1][1]}",
            help="Parsing is by far the slowest stage. When tuning chunk size, "
                 "start from 2 to reuse the blocks you already parsed.",
        )
        go = col1.button("Run ingestion", type="primary", use_container_width=True)

        if go:
            failed = False
            for num, name, desc in STAGES:
                if int(num) < start_from:
                    continue
                with st.status(f"Stage {num} — {name}: {desc}", expanded=False) as status:
                    module = __import__(f"s{num}_{name}")
                    result, out, err = capture(module.run)
                    if out:
                        st.code(out.strip(), language="text")
                    if err:
                        status.update(label=f"Stage {num} — {name}: FAILED", state="error")
                        st.error(err)
                        failed = True
                        break
                    if not result:
                        status.update(label=f"Stage {num} — {name}: produced nothing",
                                      state="error")
                        failed = True
                        break
                    status.update(label=f"Stage {num} — {name} ✅", state="complete")

            if not failed:
                st.success("Ingestion complete.")
                st.rerun()

        st.divider()
        st.subheader("Quality report")
        st.caption(
            "Read this before trusting the index. Bad ingestion rarely throws an "
            "error — it just quietly retrieves the wrong thing."
        )

        import inspect_stage

        for label, fn in [
            ("Stage 1 — Parse", inspect_stage.inspect_parse),
            ("Stage 2 — Chunk", inspect_stage.inspect_chunk),
            ("Stage 3 — Tag", inspect_stage.inspect_tag),
            ("Stage 4 — Embed", inspect_stage.inspect_embed),
        ]:
            _, out, err = capture(fn)
            if err and "not found" in err.lower():
                continue
            with st.expander(label, expanded=("BAD" in out)):
                st.code(out.strip() or err or "(no output)", language="text")


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------

with tab_ask:
    if stage_counts()["04_embedded"] == 0:
        st.info("Nothing indexed yet — ingest a book first.")
    else:
        try:
            subjects = sorted({r["subject"] for r in read_jsonl("04_embedded")})
        except FileNotFoundError:
            subjects = []

        c1, c2 = st.columns([4, 1])
        query = c1.text_input("Question", placeholder="e.g. explain torque")
        subject = c2.selectbox("Subject", ["All"] + subjects)

        use_llm = st.checkbox(
            "Write an answer with the LLM",
            value=False,
            help="Unchecked shows the retrieved chunks only — no HF_TOKEN needed. "
                 "Checked sends them to the model for a written, cited answer.",
        )

        if query:
            from s6_retrieve import retrieve

            with st.spinner("Retrieving..."):
                results, out, err = capture(
                    retrieve, query, subject=None if subject == "All" else subject
                )

            if err:
                st.error(err)
            elif not results:
                st.warning("Nothing retrieved.")
            else:
                if use_llm:
                    from s7_generate import generate

                    with st.spinner("Writing answer..."):
                        answer, gout, gerr = capture(generate, query, results)
                    if gerr:
                        st.error(gerr)
                        if "HF_TOKEN" in gerr:
                            st.info("Set HF_TOKEN in your shell, then restart Streamlit.")
                    else:
                        st.markdown(answer["answer"])
                        st.caption("Sources: " + " · ".join(
                            f"[{s['n']}] {s['citation']}" for s in answer["sources"]
                        ))
                        st.divider()

                st.subheader("Retrieved chunks")
                for r in results:
                    with st.expander(
                        f"#{r['rank']}  {r['citation']}  "
                        f"— found by {'+'.join(r['found_by'])}",
                        expanded=(r["rank"] == 1),
                    ):
                        st.write(r["text"])
                        # Showing per-retriever rank makes it visible whether dense
                        # or sparse is actually earning its keep for this question.
                        st.caption(f"RRF {r['rrf_score']} · ranks {r['ranks']}")
