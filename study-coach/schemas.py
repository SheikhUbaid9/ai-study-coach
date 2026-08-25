"""Pydantic schemas for Study Coach's memory.

Three shapes of memory live in the Store (see memory_graph.py):

- Profile: ONE document per user, namespace ("profile", user_id). Overwritten in place.
- Task: MANY documents per user, namespace ("tasks", user_id), one per UUID key. Accumulate.
- instructions: not a schema at all, just a plain string the model writes for itself
  (namespace ("instructions", user_id)). See update_instructions in memory_graph.py.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Profile(BaseModel):
    """The student's evolving profile. A single document — updates overwrite fields in place."""

    name: Optional[str] = Field(None, description="Student's name")
    city: Optional[str] = Field(None, description="Student's city")
    field: Optional[str] = Field(
        None, description="What they're studying, e.g. 'FSc Pre-Engineering'"
    )
    exam_date: Optional[str] = Field(
        None, description="ISO date (YYYY-MM-DD) of their next major exam, if known"
    )
    weak_topics: list[str] = Field(
        default_factory=list, description="Topics the student has said they struggle with"
    )
    strong_topics: list[str] = Field(
        default_factory=list, description="Topics the student is confident in"
    )


class Task(BaseModel):
    """One study task. Many of these accumulate in the store, one per generated key."""

    topic: str = Field(description="Subject, e.g. 'Math' or 'Physics'")
    chapter: str = Field(description="Chapter or specific topic within the subject")
    estimated_minutes: int = Field(description="Estimated time to complete, in minutes")
    deadline: Optional[str] = Field(None, description="ISO date (YYYY-MM-DD), or null if none set")
    status: Literal["pending", "in_progress", "done"] = "pending"
    confidence_level: int = Field(
        default=3, ge=1, le=5, description="1 = very weak, 5 = very confident"
    )


class SearchTextbook(BaseModel):
    """Search the student's own uploaded FSc textbooks for a passage.

    Bound only when books have actually been ingested — see textbook_search.py.
    """

    query: str = Field(
        description=(
            "What to look up, phrased as the textbook would phrase it rather than as "
            "the student did. For 'I don't get why the ball speeds up going down', "
            "search 'acceleration due to gravity on an inclined plane'."
        )
    )
    subject: Optional[str] = Field(
        None,
        description=(
            "Almost always leave this EMPTY so every book is searched. Only set it "
            "if the student explicitly names a subject to restrict to in this very "
            "message. Do not infer it from what they study generally — a wrong value "
            "here filters out the passage they were asking for."
        ),
    )


class DeleteTasks(BaseModel):
    """Delete study tasks. THIS IS DESTRUCTIVE and cannot be undone.

    The graph pauses before this ever runs and asks the student to confirm — see
    `interrupt_before` in app.py. Call it when they clearly ask to remove tasks;
    they will get the final say regardless.
    """

    scope: Literal["completed", "all", "topic"] = Field(
        description=(
            "'completed' removes finished tasks only (the safe, common case); "
            "'all' removes every task; 'topic' removes tasks for one subject, "
            "which requires the topic field."
        )
    )
    topic: Optional[str] = Field(
        None, description="Subject to clear, e.g. 'Biology'. Only used when scope='topic'."
    )
    reason: str = Field(
        description="One short line on what the student asked for, shown to them "
                    "on the confirmation screen so they can check it matches."
    )


class UpdateMemory(BaseModel):
    """The router tool. The model calls this — instead of just replying — when the
    latest message contains information that should be saved to long-term memory."""

    update_type: Literal["profile", "tasks", "instructions"] = Field(
        description=(
            "'profile' for facts about the student (name, city, exam date, weak/strong topics); "
            "'tasks' for study tasks being added, completed, or changed; "
            "'instructions' when the student is telling Study Coach HOW it should behave going forward "
            "(e.g. 'always use spaced repetition', 'talk to me in Roman Urdu')."
        )
    )
