"""CLI: uv run python -m agent "printer on 3rd floor offline" [--ticket T-1003]

Runs the graph until it pauses for human approval, asks you on the terminal, then
resumes the *same* run from its checkpoint with your decision.
"""

import argparse
import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import build_graph
from agent.llm import make_model
from agent.mcp_client import helpdesk_mcp


async def run_until_pause(app, graph_input, config) -> dict | None:
    """Stream the graph, printing each node as it finishes.

    Returns the interrupt payload if the graph paused, or None if it finished.
    """
    async for update in app.astream(graph_input, config, stream_mode="updates"):
        for node, _ in update.items():
            if node == "__interrupt__":
                return update["__interrupt__"][0].value
            print(f"  -> {node}")
    return None


def ask_human(payload: dict) -> dict:
    print(f"\nCategory: {payload['category']}   Severity: {payload['severity']}")
    print(f"Draft reply:\n{payload['draft']}\n")
    while True:
        choice = input("[a]pprove, [e]dit, [r]eject? ").strip().lower()
        if choice in ("a", "approve"):
            return {"action": "approve"}
        if choice in ("r", "reject"):
            return {"action": "reject"}
        if choice in ("e", "edit"):
            return {"action": "edit", "reply": input("Your reply: ")}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Triage an IT helpdesk request.")
    parser.add_argument("issue", help='e.g. "printer on 3rd floor offline"')
    parser.add_argument("--ticket", help="existing ticket id to add the final reply to")
    args = parser.parse_args()

    model = make_model(args.issue)
    print(f"Model: {getattr(model, 'model', 'scripted fake (no Claude credentials found)')}")

    async with helpdesk_mcp() as (session, tools):
        app = build_graph(model, session, tools, checkpointer=InMemorySaver())
        # thread_id names this run's checkpoint; resuming with the same id continues it.
        config = {"configurable": {"thread_id": "cli-run"}}

        payload = await run_until_pause(
            app, {"issue": args.issue, "ticket_id": args.ticket}, config
        )
        while payload is not None:
            print("  ...paused at human_approval (state saved in the checkpointer)")
            decision = ask_human(payload)
            payload = await run_until_pause(app, Command(resume=decision), config)

        state = (await app.aget_state(config)).values
        print(f"\nOutcome: {state['outcome']}")
        if state.get("final_reply"):
            print(f"Final reply:\n{state['final_reply']}")


if __name__ == "__main__":
    asyncio.run(main())
