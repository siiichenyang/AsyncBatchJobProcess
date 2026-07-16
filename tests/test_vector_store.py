import pytest
from batch_processor.chunking import TextChunk
from batch_processor.vector_store import (
    InMemoryVectorStore,
)


def test_inmemory_vector_store_vector():
    chunk1 = TextChunk(
        document_id="001",
        chunk_index=0,
        text="one two",
        start_word=0,
        end_word=2,
    )

    chunk2 = TextChunk(
        document_id="002",
        chunk_index=1,
        text="two three",
        start_word=1,
        end_word=3,
    )

    chunk3 = TextChunk(
        document_id="003",
        chunk_index=2,
        text="three four",
        start_word=2,
        end_word=4,
    )

    vector_store = InMemoryVectorStore()
    vector_store.add(chunk=chunk1, embedding=[1, 0])
    vector_store.add(chunk=chunk2, embedding=[1, 1])
    vector_store.add(chunk=chunk3, embedding=[0, 1])
    results = vector_store.search([1, 0], top_k=2)

    assert len(results) == 2
    assert [result.chunk for result in results] == [chunk1, chunk2]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(2 ** -0.5)


def test_inmemory_vector_store_rejects_invalid_top_k():
    vector_store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="zero"):
        vector_store.search([1, 0], top_k=0)
