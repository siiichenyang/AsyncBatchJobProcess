import asyncio
import pytest
import json

from batch_processor.retrieval_eval_runner import (
    run_retrieval_eval_batch,
    RetrievalEvalRunRecord,
    build_retrieval_eval_summary,
    run_retrieval_eval_file,
    RetrievalEvalReport,
    write_retrieval_eval_report,
    RetrievalEvalSummary,
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
        if query == "miss-query":
            return [
                SearchResult(
                    chunk=TextChunk(
                        document_id="doc-2",
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


def test_run_retrieval_eval_file(tmp_path):
    input_path = tmp_path / "input.jsonl"

    input_path.write_text(
        '{"name": "good query", "query": "good-query", "relevant_chunks": [{"document_id": "doc-1", "chunk_index": 0}]}\n'
        '{"name": "miss query", "query": "miss-query", "relevant_chunks": [{"document_id": "doc-1", "chunk_index": 0}]}\n'
        '{"name": "schema error", "queryb": "good-query"}\n',
        encoding="utf-8",
    )

    retriever = StubRetriever()

    report = asyncio.run(
        run_retrieval_eval_file(
            str(input_path),
            retriever,
            k=1,
        )
    )

    assert len(report.records) == 3

    assert report.records[2].line_number == 3

    assert report.summary.total == 3
    assert report.summary.evaluated == 2
    assert report.summary.errors == 1
    assert report.summary.k == 1
    assert report.summary.hit_rate == 0.5
    assert report.summary.mean_recall == 0.5


def test_write_retrieval_eval_report(tmp_path):
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.json"

    report = RetrievalEvalReport(
        records=(
            RetrievalEvalRunRecord(
                line_number=1,
                result=RetrievalEvalResult(
                    name="good query",
                    query="good-query",
                    k=2,
                    retrieved_chunks=(
                        ChunkReference(
                            document_id="doc-1",
                            chunk_index=0,
                        ),
                        ChunkReference(
                            document_id="doc-1",
                            chunk_index=1,
                        )
                    ),
                    hit=1,
                    recall=1.0,
                ),
                error=None,
            ),
            RetrievalEvalRunRecord(
                line_number=2,
                result=None,
                error="error msg",
            ),
        ),
        summary=RetrievalEvalSummary(
            total=2,
            evaluated=1,
            errors=1,
            k=2,
            hit_rate=1.0,
            mean_recall=1.0,
        )
    )

    write_retrieval_eval_report(
        report,
        output_path=str(output_path),
        summary_path=str(summary_path),
    )

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    output_jsons = [json.loads(output_line) for output_line in output_lines]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(output_jsons) == 2
    assert [item["line_number"] for item in output_jsons] == [1, 2]
    assert output_jsons[0]["result"] == {
        "name": "good query",
        "query": "good-query",
        "k": 2,
        "retrieved_chunks": [
            {"document_id": "doc-1", "chunk_index": 0},
            {"document_id": "doc-1", "chunk_index": 1}
        ],
        "hit": 1,
        "recall": 1.0
    }
    assert output_jsons[0]["error"] is None

    assert output_jsons[1]["result"] is None
    assert output_jsons[1]["error"] == "error msg"

    assert summary == {
        "total": 2,
        "evaluated": 1,
        "errors": 1,
        "k": 2,
        "hit_rate": 1.0,
        "mean_recall": 1.0,
    }
