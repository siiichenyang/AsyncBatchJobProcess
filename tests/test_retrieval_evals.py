import pytest

from batch_processor.retrieval_evals import (
    ChunkReference,
    hit_at_k,
    recall_at_k,
)


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
