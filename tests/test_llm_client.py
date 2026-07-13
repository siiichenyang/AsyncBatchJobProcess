import asyncio
from typing import Self

from batch_processor.llm_client import (
    FakeLLMClient,
    OpenAILLMClient,
)


def test_fake_llm_client_returns_configured_response():
    client = FakeLLMClient("expected response")

    response = asyncio.run(client.generate("Give me a book"))

    assert response == "expected response"


class StubOpenAIResponse():
    output_text = "fake openai client response."

    async def create(self, *, model: str, input: str) -> Self:
        self.received_model = model
        self.received_input = input
        return self


class StubOpenAILLMClient():
    responses = StubOpenAIResponse()


def test_open_ai_llm_client_stub_generate():
    client = OpenAILLMClient(
        model="gpt-example",
        api_key="fake api key",
        client=StubOpenAILLMClient()
    )

    output_text = asyncio.run(client.generate("Hello"))

    assert output_text == "fake openai client response."
    assert client.client.responses.received_model == "gpt-example"
    assert client.client.responses.received_input == "Hello"
