"""Helpdesk MCP server.

Shows all three MCP primitives:
- Tools:     functions the *model* decides to call (search, RAG search, read, write).
- Resource:  read-only data the *client app* chooses to load (the KB index).
- Prompt:    a reusable prompt template the server owns (reply guidelines).

Run it directly over stdio:      uv run python -m mcp_server.server
Or open it in the MCP Inspector: uv run mcp dev mcp_server/server.py
"""

from mcp.server.fastmcp import FastMCP

from mcp_server import rag
from mcp_server.db import connect

mcp = FastMCP("helpdesk", log_level="WARNING")  # quiet per-request INFO logs


# ---------- Tools ----------
# FastMCP builds each tool's JSON schema from the type hints and uses the
# docstring as the description the model sees. The decorator returns the
# plain function, so tests can call these directly.


@mcp.tool()
def search_kb(query: str) -> list[dict]:
    """Search IT knowledge-base articles by keywords. Returns up to 3 best matches."""
    words = [w for w in query.lower().split() if len(w) > 2]
    with connect() as conn:
        rows = conn.execute("SELECT id, category, title, body FROM kb_articles").fetchall()
    scored = []
    for row in rows:
        text = f"{row['title']} {row['body']}".lower()
        score = sum(word in text for word in words)
        if score > 0:
            scored.append((score, dict(row)))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [article for _, article in scored[:3]]


@mcp.tool()
def search_docs(query: str) -> list[dict]:
    """Semantic search over IT runbooks and policies (onboarding, VPN, data handling,
    incidents, licensing...). Describe the problem in plain words. Returns the 3 most
    relevant sections with their source document and a similarity score."""
    return rag.search(query, k=3)


@mcp.tool()
def get_ticket(id: str) -> dict:
    """Get one ticket by id (e.g. 'T-1001'), including its notes."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise ValueError(f"Ticket {id} not found")
        notes = conn.execute(
            "SELECT note FROM ticket_notes WHERE ticket_id = ? ORDER BY id", (id,)
        ).fetchall()
    return {**dict(row), "notes": [n["note"] for n in notes]}


@mcp.tool()
def list_similar_tickets(category: str) -> list[dict]:
    """List past tickets in a category (hardware, network, account, software, security)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, severity, status, title FROM tickets WHERE category = ? ORDER BY id",
            (category.lower(),),
        ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def create_ticket_note(id: str, note: str) -> dict:
    """Add a note to an existing ticket. This is the only tool that writes data."""
    with connect() as conn:
        if conn.execute("SELECT 1 FROM tickets WHERE id = ?", (id,)).fetchone() is None:
            raise ValueError(f"Ticket {id} not found")
        cur = conn.execute("INSERT INTO ticket_notes (ticket_id, note) VALUES (?, ?)", (id, note))
    return {"ticket_id": id, "note_id": cur.lastrowid, "note": note}


# ---------- Resource ----------
# Addressed by URI. The client app reads it; the model does not "call" it.


@mcp.resource("kb://index")
def kb_index() -> str:
    """One line per KB article: id, category and title."""
    with connect() as conn:
        rows = conn.execute("SELECT id, category, title FROM kb_articles ORDER BY id").fetchall()
    return "\n".join(f"{r['id']} [{r['category']}] {r['title']}" for r in rows)


# ---------- Prompt ----------
# A template the server publishes so every client drafts replies the same way.


@mcp.prompt()
def reply_guidelines(category: str) -> str:
    """Instructions for drafting a reply to an end user about an IT issue."""
    return (
        f"You are an IT helpdesk agent replying to an employee about a {category} issue.\n"
        "- Be friendly and brief: at most 6 short lines.\n"
        "- Give numbered steps the user can try themselves, based only on the KB "
        "articles, runbook sections and past tickets in the findings.\n"
        "- Cite sources like (KB-001) or (vpn-and-remote-access.md).\n"
        "- If nothing relevant was found, say a technician will follow up."
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
