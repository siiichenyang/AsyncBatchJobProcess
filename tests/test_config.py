import pytest
from batch_processor.config import LLMConfig


def test_llm_config_default_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    config = LLMConfig.from_env()

    assert config.provider == "fake"


def test_llm_config_environment_value_replace_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-example")
    monkeypatch.setenv("LLM_API_KEY", "gpt-api-key")

    config = LLMConfig.from_env()

    assert config.provider == "openai"
    assert config.model == "gpt-example"
    assert config.api_key == "gpt-api-key"


def test_llm_config_set_provider_without_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        LLMConfig.from_env()
