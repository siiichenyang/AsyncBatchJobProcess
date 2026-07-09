import asyncio
from batch_processor.llm_client import FakeLLMClient


def test_fake_llm_client_returns_configured_response():
    client = FakeLLMClient("expected response")

    response = asyncio.run(client.generate("Give me a book"))

    assert response == "expected response"
