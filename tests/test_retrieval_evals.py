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
    chunk_overlap_span,
    RelevantSpan,
    span_recall_at_k,
    span_hit_at_k,
)
from batch_processor.embeddings import DeterministicEmbeddingClient
from batch_processor.retrieval import Retriever
from batch_processor.chunking import TextChunk


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
        relevant_spans=(
            RelevantSpan(document.document_id, 0, 1),
            RelevantSpan(document.document_id, 3, 4),
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
    assert eval_result.retrieved_chunks == (
        ChunkReference("basket", 0),
        ChunkReference("basket", 1),
    )
    assert eval_result.hit == 1
    assert eval_result.recall == 1.0


def test_eval_case_valid_dict():
    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_spans": [
            {"document_id": "basket", "start_word": 0, "end_word": 1},
            {"document_id": "basket", "start_word": 3, "end_word": 6}
        ]
    }

    eval_case = RetrievalEvalCase.from_dict(data)

    assert eval_case == RetrievalEvalCase(
        name="fruit-query",
        query="Do we have apple or cherry?",
        relevant_spans=(
            RelevantSpan("basket", 0, 1),
            RelevantSpan("basket", 3, 6),
        ),
    )


def test_eval_case_rejects_empty_relevant_chunks():
    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_spans": []
    }

    with pytest.raises(ValueError, match="empty"):
        RetrievalEvalCase.from_dict(data)


@pytest.mark.parametrize("invalid_index", [-1, "str", True])
def test_eval_case_rejects_invalid_chunk_index(invalid_index):
    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_spans": [
            {"document_id": "basket", "start_word": invalid_index, "end_word": 10},
        ]
    }

    with pytest.raises(ValueError, match="invalid"):
        RetrievalEvalCase.from_dict(data)


@pytest.mark.parametrize("invalid_id", ["", 123])
def test_eval_case_rejects_invalid_document_id(invalid_id):
    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_spans": [
            {"document_id": invalid_id, "start_word": 0, "end_word": 1},
        ]
    }

    with pytest.raises(ValueError, match="document_id"):
        RetrievalEvalCase.from_dict(data)

    data = {
        "name": "fruit-query",
        "query": "Do we have apple or cherry?",
        "relevant_spans": [
            {"start_word": 0, "end_word": 1, },
        ]
    }

    with pytest.raises(ValueError, match="document_id"):
        RetrievalEvalCase.from_dict(data)


def test_chunk_overlap_span_true():
    chunk = TextChunk(
        document_id="doc-1",
        chunk_index=0,
        text="abc",
        start_word=0,
        end_word=10,
    )
    span = RelevantSpan(
        document_id="doc-1",
        start_word=5,
        end_word=20,
    )

    assert chunk_overlap_span(chunk, span)


def test_chunk_overlap_span_adjacent():
    chunk = TextChunk(
        document_id="doc-1",
        chunk_index=0,
        text="abc",
        start_word=0,
        end_word=4,
    )
    span = RelevantSpan(
        document_id="doc-1",
        start_word=4,
        end_word=6,
    )

    assert not chunk_overlap_span(chunk, span)


def test_chunk_overlap_span_different_doc():
    chunk = TextChunk(
        document_id="doc-1",
        chunk_index=0,
        text="abc",
        start_word=0,
        end_word=4,
    )
    span = RelevantSpan(
        document_id="doc-2",
        start_word=1,
        end_word=8,
    )

    assert not chunk_overlap_span(chunk, span)


def test_recall_hit_chunk_overlap_span():
    chunks = [
        TextChunk(
            document_id="doc-1",
            chunk_index=0,
            text="abc",
            start_word=0,
            end_word=4,
        ),
        TextChunk(
            document_id="doc-1",
            chunk_index=1,
            text="abc",
            start_word=15,
            end_word=17,
        ),
    ]
    spans = [
        RelevantSpan(
            document_id="doc-1",
            start_word=1,
            end_word=8,
        ),
        RelevantSpan(
            document_id="doc-1",
            start_word=10,
            end_word=17,
        ),
    ]

    assert span_hit_at_k(chunks, spans, k=1) == 1
    assert span_recall_at_k(chunks, spans, k=1) == 0.5
    assert span_recall_at_k(chunks, spans, k=2) == 1.0


def test_recall_ignore_same_chunk():
    chunks = [
        TextChunk(
            document_id="doc-1",
            chunk_index=0,
            text="abc",
            start_word=0,
            end_word=4,
        ),
        TextChunk(
            document_id="doc-1",
            chunk_index=0,
            text="abc",
            start_word=2,
            end_word=8,
        ),
    ]
    spans = [
        RelevantSpan(
            document_id="doc-1",
            start_word=2,
            end_word=3,
        ),
        RelevantSpan(
            document_id="doc-1",
            start_word=10,
            end_word=17,
        ),
    ]

    assert span_hit_at_k(chunks, spans, k=2) == 1
    assert span_recall_at_k(chunks, spans, k=2) == 0.5


def test_span_rejects_invalid_data():
    data = {
        "document_id": "doc-1",
        "start_word": 10,
        "end_word": 0,
    }

    with pytest.raises(ValueError, match="invalid"):
        RelevantSpan.from_dict(data)


def test_span_rejects_empty_doc_id():
    data = {
        "document_id": "  ",
        "start_word": 10,
        "end_word": 0,
    }

    with pytest.raises(ValueError, match="non-empty"):
        RelevantSpan.from_dict(data)
