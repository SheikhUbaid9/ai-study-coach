"""Study Coach - a personal study coach with real long-term memory, built on LangGraph.

Run with:
    streamlit run app.py

Needs HF_TOKEN set in the environment before launch (free token from
huggingface.co/settings/tokens) — see README.md.
"""

import contextlib
import io
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

import textbook_search
from memory_graph import build_graph

st.set_page_config(
    page_title="Study Coach",
    page_icon="🎓",
    # "centered" rather than "wide": the CSS caps the text column anyway, and
    # wide mode left the chat stranded in the middle of an empty page.
    layout="centered",
    # Expanded, not "auto" — auto collapses the sidebar and Streamlit then drops
    # it from the DOM entirely, which is why the voice and memory controls were
    # impossible to find.
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_compiled_graph():
    """Built once per running app process. checkpointer = per-thread chat history
    (Stage 1). store = cross-thread memory keyed by user_id (Stage 2+). Both are
    in-memory: restart the app and everything resets, matching this workspace's
    no-persistence-layer convention."""
    from persistence import get_persistence

    checkpointer, store, how = get_persistence()
    print(f"[app] memory backend: {how}")
    graph = build_graph().compile(
        checkpointer=checkpointer,
        store=store,
        # Human-in-the-loop: the graph stops BEFORE delete_tasks ever runs and
        # parks itself in the checkpointer. Nothing is deleted until the student
        # approves and we resume. This is the one place a wrong tool call would
        # destroy data rather than just produce a bad sentence, so it's the one
        # place worth interrupting.
        interrupt_before=["delete_tasks"],
    )
    return graph, store, how


def node_log_entry(node_name: str, node_output: dict | None) -> str:
    """Turn one node's raw output into a one-line, human-readable trace entry.

    node_output is None when a node wrote nothing to state — which is the normal,
    successful path for the guard (returning {} means "allow, change nothing").
    """
    messages = (node_output or {}).get("messages") or []

    if node_name == "guard":
        # No writes == allowed through. A block is the case that adds a message.
        if messages and isinstance(messages[0], AIMessage):
            return "🛡️ **guard**: off topic — blocked before reaching the coach"
        return "🛡️ **guard**: on topic, passed through"

    if not messages:
        return f"**{node_name}** ran (no message produced)"
    msg = messages[0]

    if node_name == "coach":
        if isinstance(msg, AIMessage) and msg.tool_calls:
            call = msg.tool_calls[0]
            if call["name"] == "SearchTextbook":
                q = call["args"].get("query", "?")
                return f"🧠 **coach**: decided to search the textbooks -> `{q}`"
            update_type = call["args"].get("update_type", "?")
            return f"🧠 **coach**: decided to update memory -> `{update_type}`"
        return "🧠 **coach**: composed a reply directly, no tool needed"

    if node_name == "search_textbook":
        first = msg.content.splitlines()[0] if msg.content else ""
        if "No matching passage" in msg.content:
            return "📖 **search_textbook**: nothing found in the books"
        n = msg.content.count("(source:")
        return f"📖 **search_textbook**: pulled {n} passage(s) from the books"

    if isinstance(msg, ToolMessage):
        return f"📝 **{node_name}**: {msg.content}"

    return f"**{node_name}** ran"


def render_memory_sidebar(store, user_id: str):
    st.sidebar.caption(
        f"Filed under `{user_id}` — survives a brand-new conversation."
    )

    profile_item = store.get(("profile", user_id), "user_profile")
    with st.sidebar.expander("Profile", expanded=False):
        st.json(profile_item.value if profile_item else {})

    task_items = store.search(("tasks", user_id))
    with st.sidebar.expander(f"Tasks ({len(task_items)})", expanded=False):
        if task_items:
            for item in task_items:
                st.json(item.value)
        else:
            st.caption("No tasks yet.")

    instr_item = store.get(("instructions", user_id), "user_instructions")
    with st.sidebar.expander("Standing instructions (procedural memory)", expanded=False):
        st.write(instr_item.value.get("memory", "") if instr_item else "(none yet)")


def _capture(fn, *args, **kwargs):
    """Run a pipeline stage, returning (result, printed_log, error)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result = fn(*args, **kwargs)
        return result, buf.getvalue(), None
    except Exception as exc:
        return None, buf.getvalue(), f"{type(exc).__name__}: {exc}"


def render_books_tab():
    """Upload and ingest textbooks without leaving Study Coach.

    This drives the same textbook-rag pipeline that runs standalone next door —
    it's one shared index, not a second copy. Ingest here or there; the coach
    reads whatever has been ingested either way.
    """
    st.subheader("Your textbooks")
    st.caption(
        "Upload your FSc PDFs here. Once ingested, Study Coach answers concept "
        "questions from these books and cites the chapter and page."
    )

    if not textbook_search.RAG_DIR.exists():
        st.error(
            f"Can't find the textbook pipeline at `{textbook_search.RAG_DIR}`. "
            "Study Coach still works — it just can't read books."
        )
        return

    import sys

    sys.path.append(str(textbook_search.RAG_DIR))
    from config import BOOKS_DIR

    uploads = st.file_uploader(
        "Add PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Name them so the subject is detectable — e.g. fsc-physics-part1.pdf",
    )
    if uploads:
        BOOKS_DIR.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            dest = BOOKS_DIR / up.name
            if dest.exists():
                st.warning(f"{up.name} already exists — rename it or delete the copy below.")
                continue
            dest.write_bytes(up.getbuffer())
            st.success(f"Saved {up.name} ({dest.stat().st_size / 1e6:.1f} MB)")

    books = textbook_search.library()
    st.divider()

    if not books:
        st.info("No books yet. Upload a PDF above to get started.")
        return

    for b in books:
        c1, c2, c3, c4 = st.columns([5, 2, 2, 1])
        c1.write(f"**{b['name']}**")
        c2.write(f"{b['size_mb']:.1f} MB" if b["local"] else "—")
        if not b["local"]:
            # Searchable, but the PDF itself lives on whichever machine ingested it.
            c3.write("✅ ingested")
            c4.caption("in database")
        else:
            c3.write("✅ ingested" if b["ingested"] else "⏳ not ingested")
            if c4.button("Delete", key=f"delbook_{b['name']}"):
                b["path"].unlink()
                st.rerun()

    st.divider()
    pending = [b for b in books if not b["ingested"]]
    if pending:
        st.warning(f"{len(pending)} book(s) not ingested yet — Study Coach can't read them.")

    # Ingestion needs docling, which is deliberately absent from the deployed
    # build (too large for a free host). Say so plainly instead of letting the
    # button throw an ImportError that looks like a broken app.
    try:
        import docling  # noqa: F401

        can_ingest = True
    except ImportError:
        can_ingest = False

    if not can_ingest:
        st.info(
            "**Ingestion runs on your own machine, not here.**\n\n"
            "The parsing library is too large for free hosting, so this deployed "
            "app does chat and search only. Ingest locally and the books appear "
            "here automatically — both read the same database.\n\n"
            "```\npip install -r requirements-ingest.txt\npython run.py ingest\n```"
        )
        return

    c1, c2 = st.columns([1, 3])
    start_from = c2.selectbox(
        "Start from stage", [1, 2, 3, 4, 5],
        format_func=lambda n: f"{n} — {['parse', 'chunk', 'tag', 'embed', 'index'][n - 1]}",
        help="Parsing is the slow stage. Start from 2 to reuse blocks you already parsed.",
    )
    if c1.button("Ingest books", type="primary", use_container_width=True):
        stages = [("1", "parse"), ("2", "chunk"), ("3", "tag"), ("4", "embed"), ("5", "index")]
        ok = True
        for num, name in stages:
            if int(num) < start_from:
                continue
            with st.status(f"Stage {num} — {name}", expanded=False) as status:
                module = __import__(f"s{num}_{name}")
                result, out, err = _capture(module.run)
                if out:
                    st.code(out.strip(), language="text")
                if err or not result:
                    status.update(label=f"Stage {num} — {name}: FAILED", state="error")
                    st.error(err or "stage produced nothing")
                    ok = False
                    break
                status.update(label=f"Stage {num} — {name} ✅", state="complete")
        if ok:
            st.success("Ingested. Ask a concept question in the Chat tab.")
            st.rerun()

    with st.expander("Quality report — read this before trusting the answers"):
        st.caption(
            "Bad ingestion rarely throws an error; it just quietly retrieves the "
            "wrong passage. These checks catch the silent failures."
        )
        import inspect_stage

        for label, fn in [
            ("Parse", inspect_stage.inspect_parse),
            ("Chunk", inspect_stage.inspect_chunk),
            ("Tag", inspect_stage.inspect_tag),
            ("Embed", inspect_stage.inspect_embed),
        ]:
            _, out, err = _capture(fn)
            if err and "not found" in err.lower():
                continue
            st.markdown(f"**{label}**")
            st.code(out.strip() or err or "(no output)", language="text")


CSS = """
<style>
  /* Narrow the content column — full-bleed text is hard to read. */
  .block-container { max-width: 860px; padding-top: 2.2rem; }

  /* Header */
  .sc-head { display:flex; align-items:baseline; gap:.6rem; margin-bottom:.15rem; }
  .sc-head h1 { font-size:1.75rem; font-weight:650; margin:0; letter-spacing:-.02em; }
  .sc-head span { font-size:.78rem; color:#1F6F5C; background:#E1EFEA;
                  padding:.14rem .5rem; border-radius:999px; font-weight:600; }
  .sc-sub { color:#6B6B6B; font-size:.9rem; margin:0 0 1.3rem; }

  /* Chat bubbles: flatten Streamlit's default card look */
  [data-testid="stChatMessage"] { background:transparent; padding:.35rem 0; }
  [data-testid="stChatMessageContent"] { line-height:1.62; }

  /* Tabs */
  [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid #E3DED4; }
  [data-baseweb="tab"] { font-weight:550; }

  /* Sidebar: tighten spacing, quieten labels */
  section[data-testid="stSidebar"] .block-container { padding-top:1.2rem; }
  section[data-testid="stSidebar"] h3 { font-size:.72rem; text-transform:uppercase;
      letter-spacing:.07em; color:#8A8A8A; margin:.9rem 0 .3rem; font-weight:700; }

  /* Expanders: lighter than the default heavy border */
  [data-testid="stExpander"] details { border:1px solid #E8E3DA; border-radius:8px; }

  /* Empty-state suggestion buttons */
  div[data-testid="stButton"] > button { border-radius:8px; font-weight:500; }
</style>
"""

STARTERS = [
    "Explain nociceptors from my textbook",
    "I'm weak in biology, exams in 40 days",
    "Make me a revision plan for this week",
    "Teach me using the Feynman technique",
]


def render_sidebar(store, graph):
    sb = st.sidebar

    sb.markdown("### Student")
    st.session_state.user_id = sb.text_input(
        "Profile name",
        value=st.session_state.user_id,
        label_visibility="collapsed",
        help="All long-term memory is filed under this. Change it to switch student.",
    )

    if sb.button("New conversation", use_container_width=True,
                 help="Fresh chat, same memory — proves memory outlives a conversation."):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pop("last_reply", None)
        st.rerun()
    sb.caption(f"Conversation `{st.session_state.thread_id[:8]}`")

    # --- at-a-glance counts instead of raw JSON --------------------------
    profile = store.get(("profile", st.session_state.user_id), "user_profile")
    tasks = store.search(("tasks", st.session_state.user_id))
    books = [b for b in textbook_search.library() if b["ingested"]] \
        if textbook_search.is_available() else []

    sb.markdown("### Memory")
    c1, c2, c3 = sb.columns(3)
    c1.metric("Facts", len((profile.value or {})) if profile else 0)
    c2.metric("Tasks", len(tasks))
    c3.metric("Books", len(books))

    render_memory_sidebar(store, st.session_state.user_id)

    # --- settings, folded away ------------------------------------------
    sb.markdown("### Settings")
    import voice as _voice

    with sb.expander("Voice"):
        if _voice.tts_available():
            st.session_state.read_aloud = st.toggle(
                "Read answers aloud", value=st.session_state.get("read_aloud", False)
            )
            st.session_state.voice_lang = st.selectbox(
                "Voice language", list(_voice.TTS_LANGS),
                help="Auto reads English answers in English and Urdu ones in Urdu, "
                     "including Roman Urdu. Override only if it guesses wrong.",
            )
        else:
            st.caption("Install `gtts` to enable spoken answers.")

    with sb.expander("System"):
        import llm

        from configuration import GUARDRAIL_ENABLED

        desc = llm.describe()
        if desc.startswith("not configured"):
            st.error(desc)
        else:
            st.caption(f"**Model** · `{desc}`")
        st.caption(f"**Guardrail** · {'on' if GUARDRAIL_ENABLED else 'OFF'}")
        st.caption(f"**Memory** · {st.session_state.get('mem_how', '?')}")
        st.caption(
            f"**Books** · {len(books)} indexed"
            if books else f"**Books** · none ({textbook_search.status()})"
        )


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="sc-head"><h1>Study Coach</h1><span>FSc</span></div>'
        '<p class="sc-sub">Your books, your tasks, your teacher — remembered between sessions.</p>',
        unsafe_allow_html=True,
    )

    graph, store, mem_how = get_compiled_graph()

    if "user_id" not in st.session_state:
        st.session_state.user_id = "ahmed"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    st.session_state.mem_how = mem_how
    render_sidebar(store, graph)

    config = {
        "configurable": {
            "user_id": st.session_state.user_id,
            "thread_id": st.session_state.thread_id,
        }
    }

    tab_chat, tab_books = st.tabs(["💬 Chat", "📚 My Books"])

    with tab_books:
        render_books_tab()

    with tab_chat:
        render_chat(graph, config, store)


def render_pending_approval(graph, config, store) -> bool:
    """If the graph is parked before delete_tasks, show the confirmation gate.

    Returns True when something is pending, so the caller can hide the chat input
    — leaving it usable would let a new message pile up behind a paused graph.

    The whole mechanism is three LangGraph calls:
      get_state(config).next   -> which node it's about to run (empty if running)
      stream(None, config)     -> resume from exactly where it stopped
      update_state(...)        -> write into state as if a node had produced it
    """
    snapshot = graph.get_state(config)
    if "delete_tasks" not in (snapshot.next or ()):
        return False

    from memory_graph import preview_deletion

    tool_call = snapshot.values["messages"][-1].tool_calls[0]
    args = tool_call["args"]
    doomed = preview_deletion(store, config["configurable"]["user_id"], args)

    st.warning("**Confirm before I delete anything**", icon="⚠️")
    st.caption(f"You asked: _{args.get('reason', '(no reason given)')}_")

    if not doomed:
        st.info("Nothing actually matches that — no tasks would be removed.")
    else:
        st.write(f"This will permanently delete **{len(doomed)} task(s)**:")
        for t in doomed:
            st.markdown(
                f"- **{t.get('topic')}** / {t.get('chapter')} "
                f"· _{t.get('status')}_ · {t.get('estimated_minutes')} min"
            )

    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Delete them", type="primary", use_container_width=True):
        # Resuming with None means "carry on from the interrupt" — the pending
        # delete_tasks node runs, then flows back to the coach as normal.
        for _ in graph.stream(None, config, stream_mode="updates"):
            pass
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        # Cancelling can't just skip the node: the model made a tool call, and a
        # tool call with no ToolMessage reply leaves the conversation malformed.
        # So we write the refusal in as though delete_tasks had run and said no.
        graph.update_state(
            config,
            {"messages": [ToolMessage(
                content="The student cancelled. Nothing was deleted. Confirm that "
                        "to them and carry on.",
                tool_call_id=tool_call["id"],
            )]},
            as_node="delete_tasks",
        )
        for _ in graph.stream(None, config, stream_mode="updates"):
            pass
        st.rerun()

    return True


def render_chat(graph, config, store):

    # Chat history for the CURRENT thread comes straight from the checkpointer —
    # this app keeps no separate copy of it, on purpose, so you can see that
    # LangGraph's checkpointer really is the source of truth for Stage 1.
    state = graph.get_state(config)
    history = state.values.get("messages", []) if state.values else []

    for msg in history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            with st.chat_message("assistant"):
                st.write(msg.content)

    # Empty state: a blank page gives no clue what this thing can do, and the
    # coach's best features (citing your book, remembering you) aren't guessable.
    # Clicking a starter sends it as if typed.
    # A paused graph takes priority over everything else on this tab.
    if render_pending_approval(graph, config, store):
        return

    clicked = None
    if not history:
        st.markdown("##### Try one of these")
        cols = st.columns(2)
        for i, prompt in enumerate(STARTERS):
            if cols[i % 2].button(prompt, key=f"starter_{i}", use_container_width=True):
                clicked = prompt
        st.write("")

    import voice

    # --- Read the last answer aloud, on demand ---------------------------
    # A button beside the chat beats a sidebar toggle: it's discoverable, and it
    # works retrospectively on an answer already on screen rather than only on
    # the next one. Audio is generated on click, not for every reply, so a chat
    # you never listen to costs nothing.
    if st.session_state.get("last_reply") and voice.tts_available():
        c1, c2 = st.columns([1, 5])
        if c1.button("🔊 Read answer", use_container_width=True):
            with st.spinner("Generating audio..."):
                mp3, err = voice.speak(
                    st.session_state.last_reply,
                    st.session_state.get("voice_lang", "Auto (match the reply)"),
                )
            if mp3:
                st.audio(mp3, format="audio/mp3", autoplay=True)
            else:
                st.caption(f"(couldn't read that out: {err})")
        c2.caption("Reads the most recent answer out loud.")

    # --- Voice input -----------------------------------------------------
    # Kept in an expander rather than always-visible: typing stays the primary
    # path, and a mic widget permanently occupying the screen implies otherwise.

    spoken_text = None
    with st.expander("🎤 Speak instead of typing"):
        stt_ok, stt_note = voice.stt_available()
        if not stt_ok:
            st.warning(f"Voice input unavailable — {stt_note}")
        else:
            clip = st.audio_input("Record your question")
            if clip is not None:
                audio_bytes = clip.getvalue()
                # Streamlit hands back the same clip on every rerun, so hash it
                # and skip anything already transcribed — otherwise one recording
                # gets sent to the coach repeatedly.
                digest = hash(audio_bytes)
                if st.session_state.get("last_clip") != digest:
                    with st.spinner("Transcribing..."):
                        text, err = voice.transcribe(audio_bytes)
                    if err:
                        st.error(f"Couldn't transcribe: {err}")
                    elif text:
                        st.session_state.last_clip = digest
                        spoken_text = text
                        st.info(f"Heard: “{text}”")

    typed = st.chat_input("Ask about your books, or tell me what you're working on…")
    if user_input := (typed or spoken_text or clicked):
        with st.chat_message("user"):
            st.write(user_input)

        log_entries = []
        final_reply = None
        node_error = None
        with st.chat_message("assistant"):
            status = st.empty()
            stream_box = st.empty()
            status.caption("thinking…")

            # stream_mode=["updates", "messages"] asks for BOTH: "updates" gives
            # one event per finished node (what the trace is built from), while
            # "messages" gives individual tokens as the model produces them. That
            # combination is what lets the answer type itself out while still
            # knowing which node produced it.
            #
            # Only tokens from the `coach` node are printed — the guard and the
            # Trustcall extractors are also LLM calls, and streaming their
            # internal output into the chat would be noise.
            buffer = ""
            try:
                for kind, payload in graph.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config,
                    stream_mode=["updates", "messages"],
                ):
                    if kind == "messages":
                        chunk, meta = payload
                        if meta.get("langgraph_node") != "coach":
                            continue
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            buffer += piece
                            status.empty()
                            # Cursor block makes it visibly live rather than
                            # looking like a half-loaded page.
                            stream_box.markdown(buffer + " ▌")
                    elif kind == "updates":
                        for node_name, node_output in payload.items():
                            log_entries.append(node_log_entry(node_name, node_output))
                            if node_name != "coach":
                                status.caption(f"running `{node_name}`…")
                            msgs = (node_output or {}).get("messages") or []
                            if msgs and isinstance(msgs[0], AIMessage) and msgs[0].content:
                                final_reply = msgs[0].content
            except Exception as exc:
                node_error = f"{type(exc).__name__}: {exc}"

            status.empty()
            # Prefer the streamed text: it's what the reader actually watched
            # appear, and it avoids a visible re-render at the end.
            final_reply = buffer.strip() or final_reply

            if node_error:
                stream_box.empty()
                st.error(f"A step failed: {node_error}")
                st.caption(
                    "The conversation is still usable — try rephrasing. Open the "
                    "trace below to see which node got there."
                )
            else:
                # Re-render without the cursor to settle the final text.
                stream_box.markdown(final_reply or "_(no reply generated)_")
                # Kept so the "Read answer" button still works after the rerun.
                if final_reply:
                    st.session_state.last_reply = final_reply

                # Auto-play, if the sidebar toggle is on. Generated after the text
                # is already on screen so speech synthesis never delays reading.
                if final_reply and st.session_state.get("read_aloud"):
                    mp3, tts_err = voice.speak(
                        final_reply, st.session_state.get("voice_lang", "Auto (match the reply)")
                    )
                    if mp3:
                        st.audio(mp3, format="audio/mp3", autoplay=True)
                    elif tts_err:
                        st.caption(f"(couldn't read that out: {tts_err})")

            with st.expander("🔍 Behind the scenes (LangGraph node trace)"):
                for entry in log_entries:
                    st.markdown(entry)
                if node_error:
                    st.markdown(f"❌ **failed:** `{node_error}`")

        if not node_error:
            st.rerun()


if __name__ == "__main__":
    main()
