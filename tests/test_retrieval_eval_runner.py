import asyncio
import pytest

from batch_processor.retrieval_eval_runner import (
    run_retrieval_eval_batch,
    RetrievalEvalRunRecord,
    build_retrieval_eval_summary,
)
from batch_processor.retrieval_eval_io import RetrievalEvalInputRecord
from batch_processor.retrieval_evals import (
    ChunkReference,
    RetrievalEvalCase,
    RetrievalEvalResult,
)
from batch_processor.vector_store import SearchResult
from batch_processor.chunking import TextChunk


class StubRetriever:
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SearchResult]:
        if query == "good-query":
            return [
                SearchResult(
                    chunk=TextChunk(
                        document_id="doc-1",
                        chunk_index=0,
                        text="aaa",
                        start_word=0,
                        end_word=1,
                    ),
                    score=1.0,
                )
            ]

        raise RuntimeError("retrieval failed")


def test_run_retrieval_eval_batch():
    records = [
        RetrievalEvalInputRecord(
            line_number=1,
            case=RetrievalEvalCase(
                name="good case",
                query="good-query",
                relevant_chunks=(
                    ChunkReference(
                        document_id="doc-1",
                        chunk_index=0,
                    ),
                )
            ),
            error=None,
        ),
        RetrievalEvalInputRecord(
            line_number=2,
            case=RetrievalEvalCase(
                name="bad case",
                query="bad-query",
                relevant_chunks=(
                    ChunkReference(
                        document_id="doc-1",
                        chunk_index=0,
                    ),
                )
            ),
            error=None,
        ),
        RetrievalEvalInputRecord(
            line_number=3,
            case=None,
            error="data must be json object",
        ),
    ]

    results = asyncio.run(
        run_retrieval_eval_batch(
            records,
            StubRetriever(),
            k=2,
        )
    )

    assert len(results) == 3

    assert results[0].line_number == 1
    assert results[0].result is not None
    assert results[0].error is None

    assert results[1].line_number == 2
    assert results[1].error == "RuntimeError: retrieval failed"
    assert results[1].result is None

    assert results[2].line_number == 3
    assert results[2].result is None
    assert results[2].error == "data must be json object"


@pytest.mark.parametrize("invalid_k", [-1, 0])
def test_run_eval_rejects_invalid_k(invalid_k):
    records = [
        RetrievalEvalInputRecord(
            line_number=1,
            case=RetrievalEvalCase(
                name="good case",
                query="good-query",
                relevant_chunks=(
                    ChunkReference(
                        document_id="doc-1",
                        chunk_index=0,
                    ),
                )
            ),
            error=None,
        ),
    ]

    with pytest.raises(ValueError, match="zero"):
        asyncio.run(
            run_retrieval_eval_batch(
                records,
                StubRetriever(),
                k=invalid_k,
            )
        )


def test_build_eval_summary_mix():
    records = [
        RetrievalEvalRunRecord(
            line_number=0,
            result=RetrievalEvalResult(
                name="succ-1",
                query="query-1",
                k=2,
                retrieved_chunks=(
                    ChunkReference(
                        document_id="doc-1",
                        chunk_index=0,
                    ),
                ),
                hit=1,
                recall=1.0,
            ),
            error=None,
        ),
        RetrievalEvalRunRecord(
            line_number=1,
            result=RetrievalEvalResult(
                name="succ-2",
                query="query-2",
                k=2,
                retrieved_chunks=(
                    ChunkReference(
                        document_id="doc-1",
                        chunk_index=0,
                    ),
                ),
                hit=0,
                recall=0.5,
            ),
            error=None,
        ),
        RetrievalEvalRunRecord(
            line_number=2,
            result=None,
            error="some error",
        ),
    ]

    k = 2
    summary = build_retrieval_eval_summary(
        records,
        k=k,
    )

    assert summary.total == 3
    assert summary.evaluated == 2
    assert summary.errors == 1
    assert summary.k == k
    assert summary.hit_rate == pytest.approx(0.5)
    assert summary.mean_recall == pytest.approx(0.75)


def test_build_eval_summary_all_error():
    records = [
        RetrievalEvalRunRecord(
            line_number=0,
            result=None,
            error="some error",
        ),
        RetrievalEvalRunRecord(
            line_number=1,
            result=None,
            error="some error",
        ),
        RetrievalEvalRunRecord(
            line_number=2,
            result=None,
            error="some error",
        ),
    ]

    k = 2
    summary = build_retrieval_eval_summary(
        records,
        k=k,
    )

    assert summary.total == 3
    assert summary.evaluated == 0
    assert summary.errors == 3
    assert summary.k == k
    assert summary.hit_rate == 0.0
    assert summary.mean_recall == 0.0


def test_build_eval_summary_rejects_k_not_match():
    records = [
        RetrievalEvalRunRecord(
            line_number=0,
            result=RetrievalEvalResult(
                name="succ-1",
                query="query-1",
                k=3,
                retrieved_chunks=(
                    ChunkReference(
                        document_id="doc-1",
                        chunk_index=0,
                    ),
                ),
                hit=1,
                recall=1.0,
            ),
            error=None,
        ),
    ]

    with pytest.raises(ValueError, match="expect"):
        build_retrieval_eval_summary(
            records,
            k=2,
        )


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_build_eval_summary_input_invalid_k(invalid_k):
    records = [
        RetrievalEvalRunRecord(
            line_number=0,
            result=RetrievalEvalResult(
                name="succ-1",
                query="query-1",
                k=3,
                retrieved_chunks=(
                    ChunkReference(
                        document_id="doc-1",
                        chunk_index=0,
                    ),
                ),
                hit=1,
                recall=1.0,
            ),
            error=None,
        ),
    ]

    with pytest.raises(ValueError, match="zero"):
        build_retrieval_eval_summary(
            records,
            k=invalid_k,
        )
