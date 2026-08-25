"""Graph entry point for LangGraph Studio (`langgraph dev`).

Studio runs the graph through the LangGraph API, which supplies its OWN
checkpointer and store (Postgres in a real deployment, in-memory locally). So
this compiles the graph *bare* — no MemorySaver, no InMemoryStore — where
app.py supplies both itself.

That difference is the whole point of Module 8/deployment: the graph definition
doesn't change between running it yourself and running it on the platform. Only
who provides persistence does.

    pip install "langgraph-cli[inmem]"
    cd study-coach
    langgraph dev
"""

from memory_graph import build_graph

# interrupt_before is kept: it's part of the graph's contract, not an app-level
# choice, and Studio has a proper UI for approving or editing at a breakpoint —
# which is a far better way to see human-in-the-loop than my two buttons.
graph = build_graph().compile(interrupt_before=["delete_tasks"])
