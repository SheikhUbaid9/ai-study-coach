# Course → code map

You handed me the LangChain Academy LangGraph course transcript (Modules 0-6).
This app is built as a running demonstration of that course, not a toy — every
concept below names the exact lesson and the exact file/function that
implements it, so you can explain (or defend, in an interview/demo) what you
built without having to re-derive it on the spot.

Use this table first. It tells you, per module, whether the concept is **built**
(you can point at code and run it right now) or **planned** (explained, not
yet wired up, and why).

| Course module | Concept | Status | Where in this repo |
|---|---|---|---|
| M0 – Setup | Chat models, `invoke`, messages, tool binding | ✅ Built | `llm.py` (`get_llm`), `memory_graph.py` (`llm.bind_tools([UpdateMemory])`) |
| M1 – Foundations | `StateGraph`, nodes, normal + conditional edges | ✅ Built | `memory_graph.build_graph()` — `route_message` is the conditional edge |
| M1 – Foundations | `MessagesState` + `add_messages` reducer | ✅ Built | `memory_graph.py` — every node takes/returns `MessagesState` |
| M1 – Foundations | Router (LLM picks a tool vs. replying) | ✅ Built | `coach()` binds `UpdateMemory`; `route_message` reads `tool_calls` |
| M1 – Foundations | ReAct loop (tool → back to model → reason) | ✅ Built | `update_profile`/`update_tasks`/`update_instructions` all edge back to `coach` |
| M1 – Foundations | Memory via checkpointer (`MemorySaver`, `thread_id`) | ✅ Built | `app.py` `get_compiled_graph()`; sidebar's "New conversation" button demos a fresh `thread_id` |
| M1 – Foundations | Deployment (LangGraph API/Cloud, SDK) | ⏳ Planned | See `langgraph.json` note below — one config change away, no code change |
| M2 – State & Memory | State schema (TypedDict vs. Pydantic) | ✅ Built | `schemas.py` uses Pydantic (`Profile`, `Task`, `UpdateMemory`) specifically for the validation the course calls out |
| M2 – State & Memory | Reducers beyond `add_messages` | ➖ Not needed here | Only one channel (`messages`) is written per step in this graph — no branching writes to the same key, so no custom reducer was required. (`operator.add` pattern is what you'd reach for if you added parallel nodes — see M4 row) |
| M2 – State & Memory | Multiple schemas (private state, input/output filtering) | ➖ Not needed here | This graph's `MessagesState` is small enough that nothing needed hiding. Relevant if you add a research/report pipeline (M4) |
| M2 – State & Memory | Filtering/trimming long conversations | ⏳ Planned | Straightforward addition: trim `state["messages"]` before the `coach()` system prompt call once threads get long |
| M2 – State & Memory | Chatbot with external (long-term) memory | ✅ Built | This is the whole point of the app — `InMemoryStore`, namespaces in `memory_graph.py` |
| M3 – Human-in-the-loop | Streaming | ✅ Built | `app.py` uses `graph.stream(..., stream_mode="updates")` to drive the "Behind the scenes" trace live |
| M3 – Human-in-the-loop | Breakpoints (`interrupt_before`) | ⏳ Planned | Would sit in front of any future destructive/expensive step (e.g. a bulk task-delete tool) — no such step exists yet |
| M3 – Human-in-the-loop | Editing state mid-run | ⏳ Planned | Needs a breakpoint first (see above) |
| M3 – Human-in-the-loop | Time travel (`get_state_history`, forking) | ⏳ Planned | Works out of the box once you call it — `MemorySaver` already keeps every checkpoint, nothing in the graph blocks this |
| M4 – Parallelization | `Send()` map-reduce, sub-graphs | ⏳ Planned | This is "Stage 5" from the original roadmap — needs a web-search tool (e.g. Tavily), an extra API key |
| M4 – Parallelization | Research assistant | ⏳ Planned | Same as above |
| M5 – Long-term memory | Memory schema design (Profile) | ✅ Built | `schemas.py` `Profile` — directly modeled on the course's memory-profile lesson |
| M5 – Long-term memory | Memory collection (many docs, one schema) | ✅ Built | `schemas.py` `Task`, namespace `("tasks", user_id)` |
| M5 – Long-term memory | Trustcall for patch-based updates + inserts | ✅ Built | `memory_graph.update_tasks` / `update_profile` — `enable_inserts=True` on tasks |
| M5 – Long-term memory | Procedural memory (agent rewrites its own instructions) | ✅ Built | `memory_graph.update_instructions` |
| M5 – Long-term memory | Full memory agent (routing tool + spy/explanation) | ✅ Built | `coach()` + `route_message()` is exactly this pattern; the "Behind the scenes" expander in `app.py` is the spy |
| M6 – Deployment | `langgraph.json`, CLI build | ⏳ Planned | No code changes needed — this graph already compiles standalone; just needs the config file |
| M6 – Deployment | Double texting strategies | ⏳ Planned | Relevant once deployed via LangGraph API — not applicable to a single-user local Streamlit app |
| M6 – Deployment | Assistants (same graph, different config) | 🟡 Half built | `configuration.py`'s `Configuration` dataclass is exactly the shape M6 assistants use — only `user_id` is wired through so far, not a `role`/prompt override |

## How to actually use this for a demo or interview

Pick a row, then:
1. Open the file named in "Where in this repo."
2. Run the app (`streamlit run app.py`) and trigger that behavior live — the
   "Try this to see it work" section in `README.md` has a script.
3. Point at the "Behind the scenes" expander after any chat turn — it's your
   on-demand trace of which node ran and why, which doubles as proof you
   understand the control flow, not just that it runs.

If you want to close a "Planned" row, tell me which one — each is scoped
clearly enough above to build as its own focused change rather than a big
rewrite.
