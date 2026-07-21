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
    RetrievalEvalCase,
    evaluate_retrieval_case,
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


def test_retrieval_eval_process_single_case(tmp_path):
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

    test_case = RetrievalEvalCase(
        name="word test",
        query="Do we have apple or cherry",
        relevant_chunks=(
            ChunkReference(document.document_id, 0),
            ChunkReference(document.document_id, 1),
        ),
    )

    async def run_scenario():
        await retriever.index_document(
            document,
            chunk_size=2,
            overlap=0,
        )

        return await evaluate_retrieval_case(
            case=test_case,
            retriever=retriever,
            k=2,
        )

    eval_result = asyncio.run(run_scenario())

    assert eval_result.name == test_case.name
    assert eval_result.query == test_case.query
    assert eval_result.k == 2
    assert eval_result.retrieved_chunks == test_case.relevant_chunks
    assert eval_result.hit == 1
    assert eval_result.recall == 1.0


def test_eval_case_valid_dict():
    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_chunks": [
            {"document_id": "basket", "chunk_index": 0},
            {"document_id": "basket", "chunk_index": 1}
        ]
    }

    eval_case = RetrievalEvalCase.from_dict(data)

    assert eval_case == RetrievalEvalCase(
        name="fruit-query",
        query="Do we have apple or cherry?",
        relevant_chunks=(
            ChunkReference("basket", 0),
            ChunkReference("basket", 1),
        ),
    )


def test_eval_case_rejects_empty_relevant_chunks():
    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_chunks": []
    }

    with pytest.raises(ValueError, match="empty"):
        RetrievalEvalCase.from_dict(data)


@pytest.mark.parametrize("invalid_index", [-1, "str", True])
def test_eval_case_rejects_invalid_chunk_index(invalid_index):
    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_chunks": [
            {"document_id": "basket", "chunk_index": invalid_index},
        ]
    }

    with pytest.raises(ValueError, match="chunk_index"):
        RetrievalEvalCase.from_dict(data)


@pytest.mark.parametrize("invalid_id", ["", 123])
def test_eval_case_rejects_invalid_document_id(invalid_id):
    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_chunks": [
            {"document_id": invalid_id, "chunk_index": 0},
        ]
    }

    with pytest.raises(ValueError, match="document_id"):
        RetrievalEvalCase.from_dict(data)

    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_chunks": [
            {"chunk_index": 0},
        ]
    }

    with pytest.raises(ValueError, match="document_id"):
        RetrievalEvalCase.from_dict(data)
