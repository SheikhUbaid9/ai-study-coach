"""Runtime configuration passed to the graph via RunnableConfig["configurable"].

Splitting these out from the graph code is what Stage 7 (Assistants) is about later:
the same compiled graph, given a different user_id/role at invoke time, behaves like
a different assistant. For now we only use user_id, but the shape is already in place.

Model provider: Hugging Face's Inference Providers router exposes an OpenAI-compatible
endpoint, so we drive it with langchain-openai's ChatOpenAI rather than needing a
Hugging Face-specific client. Needs HF_TOKEN in the environment (a free, read-scope
access token from huggingface.co/settings/tokens). HF_MODEL lets you swap models
without touching code — the default must support tool calling for UpdateMemory/
Trustcall to work.
"""

import os
from dataclasses import dataclass

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# Topic guardrail: when on, a cheap check node runs BEFORE the coach and blocks
# messages that aren't about studying. Costs one small LLM call per turn — set
# GUARDRAIL=off to disable it while developing.
GUARDRAIL_ENABLED = os.environ.get("GUARDRAIL", "on").lower() not in ("off", "0", "false")


@dataclass
class Configuration:
    user_id: str = "default-user"
    model: str = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
