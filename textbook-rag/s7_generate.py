"""Stage 7 — generate a grounded answer from retrieved textbook chunks.

Uses the same free Hugging Face router as the study-coach app (HF_TOKEN in the
environment). Three deliberate choices:

* **Grounding instruction first.** "Answer only from these excerpts; if it isn't
  there, say so" is the single highest-leverage anti-hallucination measure, and
  it belongs at the top of the prompt, not buried at the end.
* **Pre-numbered sources.** Each excerpt is labelled [1], [2] … and the model is
  asked to cite those numbers. It never writes a page number itself — it points
  at a slot, and we resolve the real citation in code. A model that can't invent
  a page number can't get one wrong.
* **Explicit permission to refuse.** Stating "not covered in your books" is a
  correct answer for a study tool. Silently answering from general knowledge is
  the failure mode that makes a textbook RAG untrustworthy.

Output: out/07_answer.json
"""

from __future__ import annotations

import json
import os
import sys

from common import log, warn
from config import GEN_MODEL, GEN_TEMPERATURE, HF_ROUTER_BASE_URL, OUT_DIR

SYSTEM_PROMPT = """You are a study assistant for FSc (intermediate) students.

Answer using ONLY the numbered textbook excerpts provided below. These excerpts are
the student's own textbooks — they are the authority here, not your general knowledge.

Rules:
1. If the excerpts do not contain the answer, say plainly: "This isn't covered in the
   excerpts I found from your books." Then, clearly separated, you may add what you
   know generally — but you must label it "(from general knowledge, not your book)".
2. Cite the excerpt number in square brackets after each claim, like [1] or [2][3].
   Cite only numbers that appear below. Never write a page or chapter number yourself.
3. Explain at the level of a student preparing for a board exam: define terms, keep
   the reasoning steps visible, and don't skip the "why".
4. If two excerpts disagree, say so rather than silently picking one.
"""


def build_context(results: list[dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(f"[{r['rank']}] (source: {r['citation']})\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def get_llm():
    from langchain_openai import ChatOpenAI

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Get a free read-scope token at "
            "https://huggingface.co/settings/tokens and set it in your shell."
        )
    return ChatOpenAI(
        base_url=HF_ROUTER_BASE_URL,
        api_key=token,
        model=GEN_MODEL,
        temperature=GEN_TEMPERATURE,
    )


def generate(query: str, results: list[dict]) -> dict:
    if not results:
        return {
            "query": query,
            "answer": "Nothing was retrieved from your books for this question.",
            "sources": [],
        }

    from langchain_core.messages import HumanMessage, SystemMessage

    context = build_context(results)
    user = f"Textbook excerpts:\n\n{context}\n\n---\n\nStudent's question: {query}"

    log("generate", f"asking {GEN_MODEL} with {len(results)} excerpts...")
    reply = get_llm().invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]
    )

    return {
        "query": query,
        "answer": reply.content,
        # The citation map is resolved here, in code — so [1] always points at a
        # real chunk even if the model's prose is wrong about everything else.
        "sources": [
            {"n": r["rank"], "citation": r["citation"], "chunk_id": r["chunk_id"]}
            for r in results
        ],
    }


def run(query: str, results: list[dict]) -> dict:
    try:
        out = generate(query, results)
    except Exception as exc:
        warn("generate", f"{type(exc).__name__}: {exc}")
        return {"query": query, "answer": None, "error": str(exc)}

    path = OUT_DIR / "07_answer.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print(out["answer"])
    print("-" * 72)
    for s in out["sources"]:
        print(f"  [{s['n']}] {s['citation']}")
    print("=" * 72 + "\n")
    return out


if __name__ == "__main__":
    from s6_retrieve import retrieve

    if len(sys.argv) < 2:
        print('usage: python s7_generate.py "your question"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    run(q, retrieve(q))
