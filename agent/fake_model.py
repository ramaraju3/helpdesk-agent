"""A scripted chat model so tests and CI never call a paid API.

It returns pre-written AIMessages in order, one per call, no matter what it's asked.
Because the graph calls the model in a fixed order (classify, retrieve..., draft),
a script makes a run fully deterministic.
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        if self._calls >= len(self.responses):
            raise RuntimeError(f"Script exhausted after {self._calls} calls")
        message = self.responses[self._calls]
        self._calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        # The script already contains any tool calls, so the tools are ignored.
        return self

    def with_structured_output(self, schema, **kwargs):
        # Mirrors ChatAnthropic's json_schema mode: the model replies with JSON text,
        # and a parser turns it into the pydantic object.
        return self | PydanticOutputParser(pydantic_object=schema)


# ---------- Helpers for writing scripts ----------


def classification(category: str, severity: str) -> AIMessage:
    return AIMessage(content=json.dumps({"category": category, "severity": severity}))


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def demo_script(issue: str) -> list[AIMessage]:
    """A canned run for the CLI when no API key is set.

    Uses keyword matching to pick a category and severity, so you can try both
    graph paths without a key, e.g. "printer offline" vs "my laptop was stolen".
    """
    text = issue.lower()
    keywords = {  # checked in order, so "laptop stolen" is security, not hardware
        "security": ["phishing", "stolen", "breach", "hacked"],
        "hardware": ["printer", "laptop", "charge", "monitor"],
        "network": ["wifi", "wi-fi", "vpn", "network"],
        "account": ["password", "locked", "login", "mfa"],
    }
    category = next((c for c, words in keywords.items() if any(w in text for w in words)), None)
    category = category or "software"
    severity = "high" if category == "security" or "everyone" in text else "medium"

    if severity == "high":
        return [classification(category, severity)]  # escalate path makes no more calls
    return [
        classification(category, severity),
        tool_call("search_kb", {"query": issue}, "call_1"),
        tool_call("search_docs", {"query": issue}, "call_2"),
        tool_call("list_similar_tickets", {"category": category}, "call_3"),
        AIMessage(content="I have enough context to draft a reply."),
        AIMessage(
            content="(scripted fake model) Thanks for reporting this! Please try the steps "
            "in the matching KB article and runbook. A technician will follow up if it persists."
        ),
    ]
