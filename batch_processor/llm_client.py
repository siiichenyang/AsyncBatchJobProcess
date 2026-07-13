import asyncio
from typing import Protocol

from openai import AsyncOpenAI


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str:
        ...


class FakeLLMClient:
    def __init__(self, response: str = "fake response"):
        self.response = response

    async def generate(self, prompt: str) -> str:
        return self.response


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
