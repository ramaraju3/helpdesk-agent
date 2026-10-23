"""Pick the chat model: real Claude if credentials are configured, otherwise the fake.

Two ways to give the agent Claude credentials:
- ANTHROPIC_API_KEY: a static key (simplest, for local dev).
- Workload Identity Federation (WIF): ANTHROPIC_FEDERATION_RULE_ID plus the other
  ANTHROPIC_* federation variables. The workload (e.g. a GitHub Actions run) swaps
  a short-lived identity token from its own identity provider for a short-lived
  Claude token, so there's no long-lived secret to store or rotate.
"""

import os
from functools import cached_property

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from agent.fake_model import ScriptedChatModel, demo_script

DEFAULT_MODEL = "claude-opus-5-5"


def has_claude_credentials() -> bool:
    return bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_FEDERATION_RULE_ID")
    )


class SDKAuthChatAnthropic(ChatAnthropic):
    """ChatAnthropic that lets the Anthropic SDK find the credentials itself.

    Plain ChatAnthropic always passes `api_key=...` to the SDK client, an empty
    string when no key is set. The SDK treats any explicit argument as "the caller
    chose the credentials" and skips its own lookup, so federation never runs.
    Building the clients without credential arguments makes the SDK resolve them in
    its normal order: API key, auth token, profile, federation.
    """

    @cached_property
    def _client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(max_retries=self.max_retries)

    @cached_property
    def _async_client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(max_retries=self.max_retries)


def make_model(issue: str) -> BaseChatModel:
    if not has_claude_credentials():
        return ScriptedChatModel(responses=demo_script(issue))

    return SDKAuthChatAnthropic(
        model=os.environ.get("HELPDESK_MODEL", DEFAULT_MODEL),
        max_tokens=4096,
        # Note: no `temperature` - current Claude models reject sampling params.
        # If a safety classifier declines a request, the API retries it on a
        # fallback model inside the same call instead of just stopping.
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )
