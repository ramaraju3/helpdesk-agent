"""Start the helpdesk MCP server as a subprocess and load its tools.

The agent never imports mcp_server's Python code. It only talks MCP over stdio,
so the same agent would work with any server that exposes these tools.
"""

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def helpdesk_mcp() -> AsyncIterator[tuple[ClientSession, list[BaseTool]]]:
    """Yield (session, tools). The server process lives until the `async with` exits."""
    client = MultiServerMCPClient(
        {
            "helpdesk": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "cwd": str(PROJECT_ROOT),
                "env": dict(os.environ),  # passes HELPDESK_DB through (tests use this)
            }
        }
    )
    # One session = one server process reused for every call in this run.
    # (client.get_tools() would instead start a fresh process per tool call.)
    async with client.session("helpdesk") as session:
        tools = await load_mcp_tools(session)  # MCP tools -> LangChain tools
        yield session, tools
