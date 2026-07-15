import asyncio

from batch_processor.embeddings import (
    DeterministicEmbeddingClient,
)


def test_deterministic_embedding():
    client = DeterministicEmbeddingClient(dimensions=16)

    first = asyncio.run(
        client.embed("Python async Python")
    )

    second = asyncio.run(
        client.embed("python async python")
    )

    assert first == second
    assert len(first) == 16
    assert sum(first) == 3.0
