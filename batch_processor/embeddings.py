import hashlib
from typing import Protocol


class EmbeddingClient(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...


class DeterministicEmbeddingClient:
    def __init__(self, dimensions: int = 16):
        if not dimensions > 0:
            raise ValueError(
                "dimensions must be greater than 0"
            )

        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        tokens = text.casefold().split()

        for token in tokens:
            digest = hashlib.sha256(
                token.encode("utf-8")
            ).digest()

            index = int.from_bytes(
                digest[:8],
                byteorder="big",
            ) % self.dimensions

            vector[index] += 1.0

        return vector
