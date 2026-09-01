import logging
from typing import Protocol

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str:
        ...

    async def close(self) -> None:
        ...


async def close_llm_client(client: LLMClient) -> None:
    try:
        await client.close()
    except Exception:
        logger.error("Failed to close LLM client; ignoring close error", exc_info=True)


class FakeLLMClient:
    def __init__(self, response: str = "fake response"):
        self.response = response

    async def generate(self, prompt: str) -> str:
        return self.response

    async def close(self) -> None:
        """No-op for the fake client; kept for LLMClient lifecycle symmetry."""


class OpenAILLMClient:
    def __init__(self, model: str, api_key: str, client=None):
        self.model = model

        if self.model is None or not self.model.strip():
            raise ValueError("OpenAILLMClient requires a model")

        self.client = AsyncOpenAI(
            api_key=api_key,
        ) if client is None else client

    async def generate(self, prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text

    async def close(self) -> None:
        await self.client.close()
