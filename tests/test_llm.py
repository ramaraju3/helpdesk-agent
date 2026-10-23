"""Model selection and credential wiring. No network: clients are built, never called."""

import pytest

from agent.fake_model import ScriptedChatModel
from agent.llm import SDKAuthChatAnthropic, make_model

CREDENTIAL_VARS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
]


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch, tmp_path):
    """Start every test with no credentials, ignoring any real login on this machine."""
    for name in [*CREDENTIAL_VARS, "ANTHROPIC_CONFIG_DIR"]:
        monkeypatch.delenv(name, raising=False)
    # The SDK looks for login profiles in ~/.config/anthropic; point ~ at an empty folder.
    monkeypatch.setenv("HOME", str(tmp_path))


def test_no_credentials_uses_the_fake():
    assert isinstance(make_model("printer"), ScriptedChatModel)


def test_api_key_is_picked_up(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = make_model("printer")
    assert isinstance(model, SDKAuthChatAnthropic)
    assert model._async_client.api_key == "sk-ant-test"


def test_login_token_is_picked_up(monkeypatch):
    # Same path as an `ant auth login` profile: the SDK finds a token, no API key.
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-oauth-token")
    model = make_model("printer")
    assert isinstance(model, SDKAuthChatAnthropic)
    assert model._async_client.api_key is None


def test_federation_env_gives_the_sdk_federated_credentials(monkeypatch, tmp_path):
    token_file = tmp_path / "jwt"
    token_file.write_text("dummy.jwt.value")
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_test")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "00000000-0000-0000-0000-000000000000")
    monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_test")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(token_file))

    client = make_model("printer")._async_client
    assert client.api_key is None
    assert type(client.credentials).__name__ == "WorkloadIdentityCredentials"
