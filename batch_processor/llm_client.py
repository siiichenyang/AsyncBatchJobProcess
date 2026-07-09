from typing import Protocol


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str:
        ...


class FakeLLMClient:
    def __init__(self, response: str = "fake response"):
        self.response = response

    async def generate(self, prompt: str) -> str:
        return self.response
