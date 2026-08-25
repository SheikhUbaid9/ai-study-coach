"""Deploy the chat app to Hugging Face Spaces, from Python. No GitHub needed.

    python deploy/deploy_hf.py --name study-coach
    python deploy/deploy_hf.py --name study-coach --private
    python deploy/deploy_hf.py --name study-coach --dry-run     # list files, upload nothing

Uses the `huggingface_hub` library, which is a genuine library-based deploy path:
it creates the Space and pushes files over the API.

WHAT GETS UPLOADED: only the study-coach app plus the retrieval half of
textbook-rag. Ingestion (docling, the PDFs, the parsed output) stays on your
machine — it's the heavy part, and the deployed app doesn't need it because the
ingested books already live in Supabase.

SECRETS ARE NEVER UPLOADED. .env is excluded, and this script refuses to run if
it finds one staged. You set the keys yourself in the Space's own Settings ->
Variables and secrets page, where they're encrypted and invisible to the code
you push.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

# (source, destination-in-space). Everything else is left behind.
INCLUDE = [
    ("study-coach/app.py", "app.py"),
    ("study-coach/memory_graph.py", "memory_graph.py"),
    ("study-coach/schemas.py", "schemas.py"),
    ("study-coach/configuration.py", "configuration.py"),
    ("study-coach/llm.py", "llm.py"),
    ("study-coach/voice.py", "voice.py"),
    ("study-coach/persistence.py", "persistence.py"),
    ("study-coach/textbook_search.py", "textbook_search.py"),
    ("study-coach/.streamlit/config.toml", ".streamlit/config.toml"),
    # Retrieval half of the pipeline. s1/s2/s3 (parse, chunk, tag) are ingestion
    # only, so they and their docling dependency stay local.
    ("textbook-rag/config.py", "textbook-rag/config.py"),
    ("textbook-rag/common.py", "textbook-rag/common.py"),
    ("textbook-rag/s4_embed.py", "textbook-rag/s4_embed.py"),
    ("textbook-rag/s6_retrieve.py", "textbook-rag/s6_retrieve.py"),
    ("deploy/requirements.txt", "requirements.txt"),
    ("deploy/Dockerfile", "Dockerfile"),
]

# Space card. The YAML header is how HF knows to run it as a Streamlit app.
SPACE_README = """---
title: Study Coach
emoji: 🎓
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Study Coach

An AI study coach for FSc students, built on LangGraph.

- Answers from the student's own textbooks, citing chapter and page
- Remembers their profile, tasks, and teaching preferences across conversations
- Asks before deleting anything
- Blocks off-topic requests before they reach the coach

Textbook ingestion runs separately; this Space reads the already-ingested index.

## Configuration

Set these in **Settings -> Variables and secrets**:

| Secret | Why |
|---|---|
| `GROQ_API_KEY` | the model |
| `PG_DSN` | Supabase, for books and memory |
| `LLM_PROVIDER` | `groq` |
| `STORE_BACKEND` | `pgvector` |
| `MEMORY_BACKEND` | `postgres` |
"""


def stage(target: Path) -> list[str]:
    """Copy the deployable files into a temp dir. Returns what was staged."""
    staged = []
    for src_rel, dst_rel in INCLUDE:
        src = ROOT / src_rel
        if not src.exists():
            print(f"  !! missing, skipping: {src_rel}")
            continue
        dst = target / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        staged.append(dst_rel)

    (target / "README.md").write_text(SPACE_README, encoding="utf-8")
    staged.append("README.md")
    return staged


# Match the SHAPE of a real credential, not just its prefix. Docstrings are full
# of placeholders like "gsk_your_key_here" and matching those would cry wolf on
# every run, which trains you to ignore the check — worse than no check at all.
# Real keys are a prefix followed by a long unbroken run of key characters.
SECRET_PATTERNS = [
    (r"gsk_[A-Za-z0-9]{30,}", "Groq API key"),
    (r"hf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"lsv2_[A-Za-z0-9_]{30,}", "LangSmith key"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    # A DSN with a real password pointing at a real host. Excludes placeholders
    # ([YOUR-PASSWORD]) and localhost defaults, which are neither secret nor
    # reachable from anywhere else.
    (r"postgresql://[^\s:]+:(?!\[|YOUR|your|\$)[^\s@]{6,}@(?!localhost|127\.0\.0\.1)",
     "Postgres DSN with password"),
]


def assert_no_secrets(target: Path) -> None:
    """Refuse to upload if anything credential-shaped slipped in."""
    import re

    bad = []
    for path in target.rglob("*"):
        if not path.is_file():
            continue
        if path.name in (".env", ".env.local") or path.suffix in (".pem", ".key"):
            bad.append(f"{path.name} (secret file)")
            continue
        if path.suffix not in (".py", ".toml", ".txt", ".md", ".json"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, label in SECRET_PATTERNS:
            if re.search(pattern, text):
                bad.append(f"{path.relative_to(target)} - looks like a {label}")

    if bad:
        print("\n!! REFUSING TO UPLOAD - credentials found:")
        for b in bad:
            print(f"     {b}")
        print("   Remove them and re-run. Set secrets in the Space UI instead.")
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Space name, e.g. study-coach")
    ap.add_argument("--private", action="store_true", help="create it private")
    ap.add_argument("--dry-run", action="store_true", help="stage and check, don't upload")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        print("staging files:")
        staged = stage(target)
        for s in staged:
            print(f"  + {s}")

        assert_no_secrets(target)
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"\n{len(staged)} files, {size / 1024:.0f} KB, no secrets found")

        if args.dry_run:
            print("\ndry run - nothing uploaded")
            return 0

        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("\n!! pip install huggingface_hub")
            return 1

        api = HfApi()
        try:
            who = api.whoami()["name"]
        except Exception:
            # `huggingface-cli` was retired in favour of `hf`; the old command
            # still exists but only prints a deprecation notice and exits.
            print("\n!! Not logged in. Run:  hf auth login")
            print("   (needs a token with WRITE access, from")
            print("    https://huggingface.co/settings/tokens)")
            return 1

        repo_id = f"{who}/{args.name}"
        print(f"\ndeploying to https://huggingface.co/spaces/{repo_id}")

        api.create_repo(
            repo_id=repo_id, repo_type="space", space_sdk="docker",
            private=args.private, exist_ok=True,
        )
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(target))

        print("\nUploaded. Next, in the Space's Settings -> Variables and secrets, add:")
        for k in ("GROQ_API_KEY", "PG_DSN", "LLM_PROVIDER", "STORE_BACKEND", "MEMORY_BACKEND"):
            print(f"    {k}")
        print(f"\nThen open: https://huggingface.co/spaces/{repo_id}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
