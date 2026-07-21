import asyncio
import pytest
from pathlib import Path

from batch_processor.documents import (
    load_text_document,
)
from batch_processor.vector_store import InMemoryVectorStore
from batch_processor.retrieval_evals import (
    ChunkReference,
    hit_at_k,
    recall_at_k,
)
from batch_processor.embeddings import DeterministicEmbeddingClient
from batch_processor.retrieval import Retriever


def test_retrieval_metrics_partial_hit():
    relevant = [
        ChunkReference("doc-a", 0),
        ChunkReference("doc-a", 1),
    ]
    retrieved = [
        ChunkReference("doc-a", 1),
        ChunkReference("doc-b", 0),
    ]

    assert hit_at_k(retrieved, relevant, k=2) == 1
    assert recall_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k_does_not_count_duplicate_chunks():
    chunk_a = ChunkReference("doc-1", 0)
    chunk_b = ChunkReference("doc-1", 1)

    recall = recall_at_k(
        chunks=[chunk_a, chunk_a],
        targets=[chunk_a, chunk_b],
        k=2,
    )

    assert recall == 0.5


def test_retrieval_metrics_no_hit():
    relevant = [
        ChunkReference("doc-a", 0),
        ChunkReference("doc-b", 1),
    ]
    retrieved = [
        ChunkReference("doc-c", 1),
        ChunkReference("doc-d", 0),
    ]

    assert hit_at_k(retrieved, relevant, k=2) == 0
    assert recall_at_k(retrieved, relevant, k=2) == 0


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_retrieval_rejects_invalid_k(invalid_k):
    relevant = [
        ChunkReference("doc-a", 0),
        ChunkReference("doc-b", 1),
    ]
    retrieved = [
        ChunkReference("doc-c", 1),
        ChunkReference("doc-d", 0),
    ]
    with pytest.raises(ValueError, match="zero"):
        hit_at_k(retrieved, relevant, k=invalid_k)

    with pytest.raises(ValueError, match="zero"):
        recall_at_k(retrieved, relevant, k=invalid_k)


def test_retrieval_rejects_empty_targets():
    relevant = []
    retrieved = [
        ChunkReference("doc-c", 1),
        ChunkReference("doc-d", 0),
    ]
    with pytest.raises(ValueError, match="empty"):
        hit_at_k(retrieved, relevant, k=2)

    with pytest.raises(ValueError, match="empty"):
        recall_at_k(retrieved, relevant, k=2)


def test_retrieval_metrics_respect_k_cutoff():
    relevant = ChunkReference("doc-a", 0)
    irrelevant = ChunkReference("doc-b", 0)
    retrieved = [irrelevant, relevant]

    assert hit_at_k(retrieved, [relevant], k=1) == 0
    assert recall_at_k(retrieved, [relevant], k=1) == 0.0

    assert hit_at_k(retrieved, [relevant], k=2) == 1
    assert recall_at_k(retrieved, [relevant], k=2) == 1.0


def test_retrieval_eval_process(tmp_path):
    input_path = tmp_path / "input_doc.txt"

    input_path.write_text(
        "apple pear melon cherry cake cookie",
        encoding="utf-8",
    )

    document = load_text_document(
        Path(input_path),
        document_id="basket",
    )

    retriever = Retriever(
        DeterministicEmbeddingClient(dimensions=256),
        InMemoryVectorStore(),
    )

    async def run_scenario():

        await retriever.index_document(
            document,
            chunk_size=2,
            overlap=0,
        )

        return await retriever.retrieve(
            "Do we have apple or cherry",
            top_k=2,
        )

    search_results = asyncio.run(run_scenario())

    chunk_refs = [
        ChunkReference.from_chunk(chunk.chunk)
        for chunk in search_results
    ]

    targets = [
        ChunkReference("basket", 0),
        ChunkReference("basket", 1),
    ]

    assert hit_at_k(chunk_refs, targets, 1) == 1
    assert recall_at_k(chunk_refs, targets, 1) == 0.5
    assert hit_at_k(chunk_refs, targets, 2) == 1
    assert recall_at_k(chunk_refs, targets, 2) == 1.0
