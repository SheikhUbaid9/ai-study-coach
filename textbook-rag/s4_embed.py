"""Stage 4 — embed chunks with a local open-weights model (no API, no key).

bge-small-en-v1.5: 384 dimensions, ~130MB, CPU-friendly, and better at retrieval
than all-MiniLM-L6-v2 at the same size. First run downloads the weights once.

Two details that matter more than they look:

* **Normalised vectors.** We L2-normalise at encode time, which makes cosine
  similarity and dot product identical — so Stage 6 can use a plain dot product
  (cheaper) without changing the ranking.
* **Asymmetric query prefix.** bge was trained with an instruction prefix on the
  *query* side only, never on documents. Stage 6 applies it to queries; we
  deliberately don't apply it here. Deviating from a model's trained format
  measurably degrades retrieval, so this asymmetry is intentional, not a bug.

Vectors go to out/04_vectors.npy (row i ↔ line i of 04_embedded.jsonl) rather
than being inlined as 384 floats per line, which would make the JSONL unreadable.

Output: out/04_embedded.jsonl + out/04_vectors.npy
Inspect: python inspect.py embed
"""

from __future__ import annotations

import sys

from common import log, read_jsonl, warn, write_jsonl
from config import EMBED_BATCH_SIZE, EMBED_DIM, EMBED_MODEL, OUT_DIR, QUERY_PREFIX

_model = None


def get_model():
    """Loaded once per process and cached — Stage 2's semantic splitter reuses
    this same instance rather than loading a second copy into memory."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        log("embed", f"loading {EMBED_MODEL} (first run downloads ~130MB)...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: list[str], is_query: bool = False):
    """Encode to normalised vectors. `is_query` applies bge's instruction prefix."""
    import numpy as np

    model = get_model()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def run() -> list[dict]:
    import numpy as np

    chunks = read_jsonl("03_tagged")
    if not chunks:
        warn("embed", "no tagged chunks found")
        return []

    texts = [c["embed_text"] for c in chunks]
    log("embed", f"embedding {len(texts)} chunks...")
    vectors = embed_texts(texts)

    if vectors.shape[1] != EMBED_DIM:
        warn(
            "embed",
            f"model returned dim {vectors.shape[1]} but config says {EMBED_DIM} "
            "— update EMBED_DIM in config.py",
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vec_path = OUT_DIR / "04_vectors.npy"
    np.save(vec_path, vectors)
    log("embed", f"vectors {vectors.shape} -> {vec_path.name}")

    # Keep the JSONL free of raw floats; row order is the join key.
    embedded = [
        {**c, "vector_row": i, "embed_model": EMBED_MODEL} for i, c in enumerate(chunks)
    ]
    write_jsonl("04_embedded", embedded)
    return embedded


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
