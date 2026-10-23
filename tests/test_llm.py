"""Model selection and credential wiring. No network: clients are built, never called."""

from agent.fake_model import ScriptedChatModel
from agent.llm import SDKAuthChatAnthropic, make_model

FEDERATION_ENV = {
    "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
    "ANTHROPIC_ORGANIZATION_ID": "00000000-0000-0000-0000-000000000000",
    "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_test",
}


def test_no_credentials_uses_the_fake(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_FEDERATION_RULE_ID", raising=False)
    assert isinstance(make_model("printer"), ScriptedChatModel)


def test_api_key_is_picked_up(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_FEDERATION_RULE_ID", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = make_model("printer")
    assert isinstance(model, SDKAuthChatAnthropic)
    assert model._async_client.api_key == "sk-ant-test"


def test_federation_env_gives_the_sdk_federated_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    token_file = tmp_path / "jwt"
    token_file.write_text("dummy.jwt.value")
    for name, value in {**FEDERATION_ENV, "ANTHROPIC_IDENTITY_TOKEN_FILE": str(token_file)}.items():
        monkeypatch.setenv(name, value)

    client = make_model("printer")._async_client
    assert client.api_key is None
    assert type(client.credentials).__name__ == "WorkloadIdentityCredentials"
