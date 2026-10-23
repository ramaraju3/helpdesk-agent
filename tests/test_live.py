"""One smoke test against the real Claude API. Costs money, so it's opt-in:

ANTHROPIC_API_KEY=... uv run pytest -m live
(or with the Workload Identity Federation variables set, as in CI)
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import build_graph
from agent.llm import has_claude_credentials, make_model
from agent.mcp_client import helpdesk_mcp

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not has_claude_credentials(), reason="needs Claude credentials"),
]


async def test_live_printer_issue():
    issue = "The printer on the 3rd floor shows offline on my laptop"
    config = {"configurable": {"thread_id": "live"}}
    async with helpdesk_mcp() as (session, tools):
        app = build_graph(make_model(issue), session, tools, InMemorySaver())
        await app.ainvoke({"issue": issue}, config)

        paused = await app.aget_state(config)
        assert paused.next == ("human_approval",)
        assert paused.values["category"] == "hardware"
        assert paused.values["draft"].strip()

        await app.ainvoke(Command(resume={"action": "approve"}), config)
    assert (await app.aget_state(config)).values["outcome"] == "replied"
