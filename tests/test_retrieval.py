import asyncio
import pytest

from batch_processor.documents import Document
from batch_processor.embeddings import DeterministicEmbeddingClient
from batch_processor.retrieval import Retriever
from batch_processor.vector_store import InMemoryVectorStore


def test_document_retrieval():
    document = Document(
        document_id="python-guide",
        source_path="memory",
        text="python async await garden soil water",
    )

    retriever = Retriever(
        embedding_client=DeterministicEmbeddingClient(dimensions=256),
        vector_store=InMemoryVectorStore(),
    )

    async def run_scenario():
        await retriever.index_document(
            document,
            chunk_size=3,
        )

        return await retriever.retrieve(
            "python async await",
            top_k=1,
        )

    results = asyncio.run(run_scenario())

    assert len(results) == 1
    assert results[0].chunk.document_id == "python-guide"
    assert results[0].chunk.text == "python async await"
    assert results[0].score == pytest.approx(1.0)
