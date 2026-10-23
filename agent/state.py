"""The typed state that flows through the graph.

Each node receives the full state and returns a dict with only the keys it changed.
LangGraph merges that update into the state. Most keys are simply overwritten;
`messages` uses the `add_messages` reducer, so updates are *appended* instead.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Category = Literal["hardware", "network", "account", "software", "security"]
Severity = Literal["low", "medium", "high"]


class Classification(BaseModel):
    """Structured output of the classify node."""

    category: Category
    severity: Severity = Field(
        description="high = security incident, data at risk, or many people blocked"
    )


class TicketState(TypedDict, total=False):
    # Input
    issue: str  # what the employee reported
    ticket_id: str | None  # optional existing ticket to write the final note to

    # Set by classify
    category: Category
    severity: Severity

    # The conversation used by the retrieve <-> tools loop
    messages: Annotated[list[AnyMessage], add_messages]

    # Set by draft_reply / escalate, then reviewed by a human
    draft: str

    # Set after human_approval / finalize
    final_reply: str
    outcome: Literal["replied", "escalated", "rejected"]
