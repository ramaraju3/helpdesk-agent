"""Unit tests for the MCP server.

Most tests call the tool functions directly (fast, no protocol). The last test talks
to the server over the real MCP protocol, using the SDK's in-memory transport.
"""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server.server import (
    create_ticket_note,
    get_ticket,
    kb_index,
    list_similar_tickets,
    mcp,
    reply_guidelines,
    search_kb,
)


def test_search_kb_ranks_best_match_first():
    results = search_kb("printer offline")
    assert results[0]["id"] == "KB-001"
    assert len(results) <= 3


def test_search_kb_no_match_returns_empty():
    assert search_kb("zzz qqq") == []


def test_get_ticket_includes_notes():
    ticket = get_ticket("T-1001")
    assert ticket["title"] == "3rd floor printer offline"
    assert ticket["notes"] == ["Print server restarted by Facilities IT; resolved."]


def test_get_ticket_unknown_id_raises():
    with pytest.raises(ValueError, match="not found"):
        get_ticket("T-9999")


def test_list_similar_tickets_filters_by_category():
    tickets = list_similar_tickets("Security")
    assert [t["id"] for t in tickets] == ["T-1013", "T-1014", "T-1015"]


def test_create_ticket_note_persists():
    create_ticket_note("T-1003", "Asked user to try another charger.")
    assert get_ticket("T-1003")["notes"] == ["Asked user to try another charger."]


def test_create_ticket_note_unknown_id_raises():
    with pytest.raises(ValueError):
        create_ticket_note("T-9999", "hello")


def test_resource_and_prompt_content():
    assert kb_index().splitlines()[0] == "KB-001 [hardware] Printer shows offline"
    assert "network issue" in reply_guidelines("network")


async def test_server_over_mcp_protocol():
    """What a real client sees: tool list, a tool call, the resource and the prompt."""
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == {
            "search_kb",
            "search_docs",
            "get_ticket",
            "list_similar_tickets",
            "create_ticket_note",
        }

        result = await client.call_tool("get_ticket", {"id": "T-1002"})
        assert not result.isError
        assert json.loads(result.content[0].text)["status"] == "resolved"

        bad = await client.call_tool("get_ticket", {"id": "nope"})
        assert bad.isError  # errors come back as a result, not an exception

        resource = await client.read_resource("kb://index")
        assert "KB-010" in resource.contents[0].text

        prompt = await client.get_prompt("reply_guidelines", {"category": "account"})
        assert "account issue" in prompt.messages[0].content.text
