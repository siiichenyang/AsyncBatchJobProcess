import asyncio
from typing import Self

from batch_processor.llm_client import (
    FakeLLMClient,
    OpenAILLMClient,
    close_llm_client,
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

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def test_fake_llm_client_close_is_noop():
    client = FakeLLMClient()

    asyncio.run(client.close())


def test_open_ai_llm_client_close_calls_underlying_client():
    stub = StubOpenAILLMClient()
    client = OpenAILLMClient(
        model="gpt-example",
        api_key="fake api key",
        client=stub,
    )

    asyncio.run(client.close())

    assert stub.closed is True


class FailingCloseClient:
    async def generate(self, prompt: str) -> str:
        return "fake response"

    async def close(self) -> None:
        raise RuntimeError("close failed")


def test_close_llm_client_does_not_raise_close_errors():
    client = FailingCloseClient()

    asyncio.run(close_llm_client(client))


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
