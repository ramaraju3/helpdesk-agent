"""The triage graph.

classify ─┬─ severity high ──> escalate ─────────────────────┐
          └─ otherwise ──> retrieve <──> tools               │
                              │ (no more tool calls)         │
                              └──> draft_reply ──────────────┤
                                                             v
                                 human_approval (interrupt: pause for a person)
                                     ├─ approve / edit ──> finalize ──> END
                                     └─ reject ──────────────────────> END
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from mcp import ClientSession

from agent.state import Classification, TicketState

CLASSIFY_PROMPT = (
    "Classify this IT helpdesk request. Severity is high for security incidents "
    "(phishing, stolen devices, compromised accounts) or outages blocking many people; "
    "medium if one person is blocked; low otherwise."
)

RETRIEVE_PROMPT = (
    "You are researching a {category} helpdesk request. Use the tools to find relevant "
    "KB articles, runbook sections (search_docs) and similar past tickets. Make at most "
    "3 tool calls, then reply with a one-line summary of what you found.\n\n"
    "KB index:\n{kb_index}"
)

# The model may read data, but never write. The only write happens in finalize,
# after a human has approved.
WRITE_TOOLS = {"create_ticket_note"}


def build_graph(
    model: BaseChatModel,
    session: ClientSession,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
):
    read_tools = [t for t in tools if t.name not in WRITE_TOOLS]
    tools_by_name = {t.name: t for t in tools}

    async def classify(state: TicketState) -> dict:
        # json_schema mode = the API constrains the reply to the Classification schema.
        classifier = model.with_structured_output(Classification, method="json_schema")
        result = await classifier.ainvoke(
            [SystemMessage(CLASSIFY_PROMPT), HumanMessage(state["issue"])]
        )
        return {
            "category": result.category,
            "severity": result.severity,
            "messages": [HumanMessage(state["issue"])],  # starts the research chat
        }

    async def retrieve(state: TicketState) -> dict:
        # A *resource*: the app (not the model) decides to read it and puts it in context.
        kb_index = (await session.read_resource("kb://index")).contents[0].text
        system = RETRIEVE_PROMPT.format(category=state["category"], kb_index=kb_index)
        response = await model.bind_tools(read_tools).ainvoke(
            [SystemMessage(system), *state["messages"]]
        )
        return {"messages": [response]}

    async def draft_reply(state: TicketState) -> dict:
        # A *prompt*: the server owns the reply guidelines; we fetch them by name.
        guidelines = await load_mcp_prompt(
            session, "reply_guidelines", arguments={"category": state["category"]}
        )
        # Give the drafter only the evidence (tool results), not the whole research chat.
        findings = "\n\n".join(m.text for m in state["messages"] if isinstance(m, ToolMessage))
        response = await model.ainvoke(
            [
                SystemMessage(guidelines[0].content),
                HumanMessage(f"Employee's request: {state['issue']}\n\nFindings:\n{findings}"),
            ]
        )
        return {"draft": response.text}

    async def escalate(state: TicketState) -> dict:
        # No LLM here: high severity follows a fixed, predictable script.
        return {
            "draft": f"This has been escalated as HIGH severity to the {state['category']} "
            "on-call team. Someone will contact you within 15 minutes."
        }

    def human_approval(state: TicketState) -> dict:
        # interrupt() saves the state via the checkpointer and stops the run. The value
        # passed in is shown to the caller. When the caller resumes with
        # Command(resume=...), this node runs again *from the top* and interrupt()
        # returns that resume value. So: no side effects before interrupt().
        decision = interrupt(
            {"draft": state["draft"], "category": state["category"], "severity": state["severity"]}
        )
        if decision["action"] == "reject":
            return {"outcome": "rejected"}
        if decision["action"] == "edit":
            return {"final_reply": decision["reply"]}
        return {"final_reply": state["draft"]}  # approve

    async def finalize(state: TicketState) -> dict:
        if state.get("ticket_id"):
            # We call the MCP tool ourselves: this write is not up to the model.
            await tools_by_name["create_ticket_note"].ainvoke(
                {"id": state["ticket_id"], "note": f"Reply sent: {state['final_reply']}"}
            )
        return {"outcome": "escalated" if state["severity"] == "high" else "replied"}

    # ----- Routing functions (conditional edges) -----

    def route_by_severity(state: TicketState) -> str:
        return "escalate" if state["severity"] == "high" else "retrieve"

    def route_after_retrieve(state: TicketState) -> str:
        return "tools" if state["messages"][-1].tool_calls else "draft_reply"

    def route_after_approval(state: TicketState) -> str:
        return END if state.get("outcome") == "rejected" else "finalize"

    # ----- Wiring -----

    graph = StateGraph(TicketState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("tools", ToolNode(read_tools))  # runs the tool calls the model asked for
    graph.add_node("draft_reply", draft_reply)
    graph.add_node("escalate", escalate)
    graph.add_node("human_approval", human_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_by_severity, ["escalate", "retrieve"])
    graph.add_conditional_edges("retrieve", route_after_retrieve, ["tools", "draft_reply"])
    graph.add_edge("tools", "retrieve")
    graph.add_edge("draft_reply", "human_approval")
    graph.add_edge("escalate", "human_approval")
    graph.add_conditional_edges("human_approval", route_after_approval, ["finalize", END])
    graph.add_edge("finalize", END)

    # A checkpointer is required for interrupt(): it's where the paused state lives.
    return graph.compile(checkpointer=checkpointer)
