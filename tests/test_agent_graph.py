"""End-to-end graph tests: real MCP server (stdio subprocess) + scripted fake model.

Free and deterministic. Each test checks which nodes ran and the final state.
"""

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.fake_model import ScriptedChatModel, classification, tool_call
from agent.graph import build_graph
from agent.mcp_client import helpdesk_mcp
from mcp_server.server import get_ticket

CONFIG = {"configurable": {"thread_id": "test"}}

MEDIUM_SCRIPT = [
    classification("hardware", "medium"),
    tool_call("search_kb", {"query": "printer offline"}, "c1"),
    tool_call("search_docs", {"query": "laptop will not turn on"}, "c2"),
    AIMessage(content="Found KB-001 and a runbook section."),
    AIMessage(content="Please re-add the printer (KB-001)."),
]


async def run(app, graph_input):
    """Run until the graph pauses or ends. Returns (nodes that ran, interrupt payload)."""
    path, payload = [], None
    async for update in app.astream(graph_input, CONFIG, stream_mode="updates"):
        for node in update:
            if node == "__interrupt__":
                payload = update[node][0].value
            else:
                path.append(node)
    return path, payload


async def test_medium_severity_retrieves_drafts_and_pauses_then_finalizes():
    model = ScriptedChatModel(responses=MEDIUM_SCRIPT)
    async with helpdesk_mcp() as (session, tools):
        app = build_graph(model, session, tools, InMemorySaver())

        path, payload = await run(app, {"issue": "printer offline", "ticket_id": "T-1003"})
        assert path == [
            "classify",
            "retrieve",
            "tools",  # search_kb
            "retrieve",
            "tools",  # search_docs (RAG)
            "retrieve",
            "draft_reply",
        ]
        assert payload["draft"] == "Please re-add the printer (KB-001)."

        # Paused: the checkpointer holds the state and knows which node is next.
        snapshot = await app.aget_state(CONFIG)
        assert snapshot.next == ("human_approval",)
        tool_results = [m for m in snapshot.values["messages"] if isinstance(m, ToolMessage)]
        assert "KB-001" in tool_results[0].text  # real data from the MCP server
        assert "Laptop will not turn on" in tool_results[1].text  # retrieved runbook chunk

        path, payload = await run(app, Command(resume={"action": "approve"}))
        assert path == ["human_approval", "finalize"]
        assert payload is None

    final = (await app.aget_state(CONFIG)).values
    assert final["outcome"] == "replied"
    assert final["final_reply"] == "Please re-add the printer (KB-001)."
    assert get_ticket("T-1003")["notes"] == ["Reply sent: Please re-add the printer (KB-001)."]


async def test_high_severity_skips_retrieval_and_escalates():
    model = ScriptedChatModel(responses=[classification("security", "high")])
    async with helpdesk_mcp() as (session, tools):
        app = build_graph(model, session, tools, InMemorySaver())
        path, payload = await run(app, {"issue": "laptop stolen"})
        assert path == ["classify", "escalate"]
        assert "security on-call" in payload["draft"]

        path, _ = await run(app, Command(resume={"action": "approve"}))
        assert path == ["human_approval", "finalize"]
    assert (await app.aget_state(CONFIG)).values["outcome"] == "escalated"


async def test_reject_ends_without_writing_a_note():
    model = ScriptedChatModel(responses=MEDIUM_SCRIPT)
    async with helpdesk_mcp() as (session, tools):
        app = build_graph(model, session, tools, InMemorySaver())
        await run(app, {"issue": "printer offline", "ticket_id": "T-1003"})
        path, _ = await run(app, Command(resume={"action": "reject"}))
        assert path == ["human_approval"]  # finalize never ran
    assert (await app.aget_state(CONFIG)).values["outcome"] == "rejected"
    assert get_ticket("T-1003")["notes"] == []


async def test_edit_replaces_the_draft():
    model = ScriptedChatModel(responses=MEDIUM_SCRIPT)
    async with helpdesk_mcp() as (session, tools):
        app = build_graph(model, session, tools, InMemorySaver())
        await run(app, {"issue": "printer offline"})
        await run(app, Command(resume={"action": "edit", "reply": "Fixed on our side."}))
    assert (await app.aget_state(CONFIG)).values["final_reply"] == "Fixed on our side."


async def test_model_cannot_call_the_write_tool():
    """create_ticket_note isn't bound to the model, so asking for it just errors."""
    model = ScriptedChatModel(
        responses=[
            classification("hardware", "low"),
            tool_call("create_ticket_note", {"id": "T-1003", "note": "sneaky"}, "c1"),
            AIMessage(content="Done."),
            AIMessage(content="Reply."),
        ]
    )
    async with helpdesk_mcp() as (session, tools):
        app = build_graph(model, session, tools, InMemorySaver())
        await run(app, {"issue": "printer offline"})
        messages = (await app.aget_state(CONFIG)).values["messages"]
    error = next(m for m in messages if isinstance(m, ToolMessage))
    assert error.status == "error"
    assert get_ticket("T-1003")["notes"] == []
