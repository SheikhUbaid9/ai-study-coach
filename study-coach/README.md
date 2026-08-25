# Study Coach — a study assistant with real long-term memory

A LangGraph learning project, built to mirror the LangChain Academy course
(Modules 0-6) as a single working Streamlit app. Each concept below maps to a
specific file/function so you can trace exactly what runs when — see
`COURSE_MAP.md` for the full lesson-by-lesson mapping.

## Which model it talks to

Provider-agnostic — they all speak the same OpenAI-compatible API, so switching is
environment variables, not code. This matters because free tiers run out; when one
starts returning `402`, you move without touching the graph.

```powershell
# Groq — generous free tier, fast, 70B model (best at the structured extraction)
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "gsk_..."

# Google Gemini free tier
$env:LLM_PROVIDER  = "gemini"
$env:GEMINI_API_KEY = "..."

# Ollama — local, unlimited, no account, no key. Slower on CPU.
$env:LLM_PROVIDER = "ollama"

# Hugging Face (the default)
$env:HF_TOKEN = "hf_..."
```

Override the model with `LLM_MODEL`, or point somewhere else entirely with
`LLM_BASE_URL`. The sidebar always shows which provider and model are live.

A note on model size: Trustcall's patch schema uses `anyOf` (from `Optional`
fields), and several hosted providers can't compile that into a decoding grammar —
you get `422 failed to compile grammar`. The code falls back to plain JSON
extraction when that happens, which works on any model, but a larger model is
noticeably more reliable at it. See `_extract` in `memory_graph.py`.

## Setup

Runs on a free, open-source hosted model via Hugging Face's Inference Providers
router (no local install, no GPU needed) — see `llm.py` and `configuration.py`.

```bash
cd ustaad
pip install -r requirements.txt
```

1. Sign up (free) at https://huggingface.co/join
2. Create a **read-scope** access token at https://huggingface.co/settings/tokens
3. Set it as an environment variable, then run:

```powershell
$env:HF_TOKEN = "hf_..."   # PowerShell
streamlit run app.py
```

Default model is `meta-llama/Llama-3.1-8B-Instruct`, chosen because it reliably
supports tool calling (required for `UpdateMemory` and Trustcall). To try a
different one, set `$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"` (or similar) before
launching — see `configuration.py`. Not every hosted open-source model supports tool
calling well; if you see tool-call errors, switch models before assuming the graph
logic is broken.

## What's actually in here (Stages 1-4)

**Stage 1 — thread memory.** `MemorySaver` in `app.py` checkpoints the message list
per `thread_id`. The chat UI doesn't keep its own copy of history — it reads straight
from `graph.get_state(config)` every rerun, so you can see the checkpointer really is
the source of truth. Click "New conversation, same memory" in the sidebar to get a
fresh `thread_id` — the chat box goes blank, but Study Coach still remembers you (see Stage 2).

**Stage 2 — the Store.** `InMemoryStore` in `app.py`, read from in `memory_graph.coach()`.
Three namespaces, all keyed on `user_id` (not `thread_id`):
- `("profile", user_id)` → one document, key `"user_profile"` — overwritten in place.
- `("tasks", user_id)` → many documents, one UUID key per task — accumulate.
- `("instructions", user_id)` → one document holding the model's own standing orders.

The sidebar's "What Study Coach remembers" panel reads these same namespaces directly, so
you can watch them change in real time as you chat.

**Stage 3 — Trustcall.** `update_profile` and `update_tasks` in `memory_graph.py` use
`trustcall.create_extractor` instead of asking the model to rewrite a whole document.
Trustcall is given the *existing* value(s) plus the new conversation turn, and emits
a patch: which fields on which existing document changed, plus (for tasks, since
`enable_inserts=True`) any brand-new documents to insert. That's what lets one sentence
like "chapter 3 done but definite integrals confused me" turn into an update to one
task *and* a new task for the weak topic, without the model ever re-typing the tasks
it didn't touch.

**Stage 4 — procedural memory.** `update_instructions` in `memory_graph.py`. No schema
— it just hands the model its own current instructions plus your new message and asks
it to write an updated version of its own operating rules. That text gets injected into
every future system prompt (see `MODEL_SYSTEM_MESSAGE`), so "always use spaced
repetition, talk to me in Roman Urdu" sticks across every future conversation.

## Textbooks (the RAG integration)

Study Coach can answer concept questions from your own FSc textbooks, with a
chapter-and-page citation, instead of from the model's general knowledge.

Upload PDFs in the **My Books** tab and press Ingest. Once a book is ingested,
the coach node gains a second tool, `SearchTextbook`, alongside `UpdateMemory` —
so a question like *"explain torque"* routes to `search_textbook`, which pulls
real passages and hands them back as a `ToolMessage` for the coach to teach from.
Watch it happen in the "Behind the scenes" expander.

Two design points worth knowing:

- **It's a soft dependency.** If no book has been ingested, the tool isn't bound
  at all and Study Coach behaves exactly as before. Binding a tool the student
  can't use just invites the model to call it and then apologise.
- **The pipeline stays a separate project.** `../textbook-rag` runs standalone —
  its own UI (`streamlit run app.py`), its own CLI (`python run.py ingest`), its
  own config and quality reports. Study Coach borrows its retrieval through
  `textbook_search.py`. Both read the *same* index, so a book ingested in either
  place is immediately visible in the other.

See `../textbook-rag/README.md` for the ingestion pipeline itself.

## The routing pattern tying it together

Every turn goes through one node, `coach`, in `memory_graph.py`. It's bound to a
single tool, `UpdateMemory` (`schemas.py`), whose only job is to say *which* kind of
memory needs to change — not to change it. `route_message` reads that tool call and
sends the turn to `update_profile` / `update_tasks` / `update_instructions`, each of
which does the actual write and appends a `ToolMessage` describing what changed.
Control returns to `coach`, which now has that `ToolMessage` in context and can say
"I updated X, added Y" in its own words, instead of a generic "ok, saved."

Try the sidebar's "Behind the scenes" expander after any message — it lists exactly
which nodes ran, in order, and what each one decided or wrote.

## Try this to see it work

1. `user_id = ahmed`. Say: *"I'm Ahmed, doing FSc, exam is 2026-10-15, weak in Integration."*
   Check the sidebar — a Profile appears.
2. *"I need to revise Math chapter 3 (Integration) and Physics chapter 12."* — two Tasks appear.
3. Click "New conversation, same memory." Ask *"what should I do today?"* — it still
   knows your exam date and tasks, in a completely blank thread.
4. *"Math ch 3 done but definite integrals confused me, and physics is due tomorrow."*
   — watch one task flip to done, a new weak-topic task appear, and the other task's
   deadline update, all from one sentence.
5. *"Always plan my revisions in 45-minute blocks with spaced repetition, and talk to
   me in Roman Urdu."* — ask for a plan afterwards and see it follow the rule.

## Not built yet (Stages 5-8) — and why

- **Stage 5 (map-reduce research)** needs a web-search tool (e.g. Tavily) — an extra
  API key. Structurally it's `Send()` fanning out one `research_topic` node per weak
  topic, then a reduce node merging the returned sections.
- **Stage 6 (human-in-the-loop)** is `interrupt_before=[...]` on the compiled graph —
  straightforward to add here once there's a step worth pausing before (Stage 5's
  research, or any future delete/send action).
- **Stage 7 (assistants)** is really just "compile this same graph, invoke it with a
  different `user_id`/role in `configurable`" — `configuration.py` is already shaped
  for it.
- **Stage 8 (deployment)** swaps `MemorySaver`/`InMemoryStore` for Postgres-backed
  equivalents and adds a `langgraph.json` — no graph logic changes.

Ask to build any of these next once Stage 1-4 feels solid.
