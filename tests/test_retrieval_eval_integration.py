import asyncio
from pathlib import Path

import pytest

from batch_processor.documents import load_text_document
from batch_processor.retrieval import Retriever
from batch_processor.embeddings import DeterministicEmbeddingClient
from batch_processor.vector_store import InMemoryVectorStore
from batch_processor.retrieval_eval_runner import (
    run_retrieval_eval_file,
    RetrievalEvalReport,
)


def test_retrieval_eval_integration():
    root_path = Path(__file__).resolve().parents[1]
    document_path = root_path / "data" / "rag" / "backend_topics.txt"
    eval_input_path = root_path / "data" / "rag" / "retrieval_eval_cases.jsonl"

    document = load_text_document(
        document_path,
        document_id="backend_topics",
    )

    retriever = Retriever(
        DeterministicEmbeddingClient(dimensions=4096),
        InMemoryVectorStore(),
    )

    async def scenario() -> RetrievalEvalReport:
        await retriever.index_document(
            document,
            chunk_size=6,
            overlap=0,
        )

        return await run_retrieval_eval_file(
            str(eval_input_path),
            retriever,
            k=1,
        )

    report = asyncio.run(scenario())

    assert report.summary.total == 4
    assert report.summary.evaluated == 4
    assert report.summary.errors == 0
    assert report.summary.k == 1
    assert report.summary.hit_rate == pytest.approx(0.75)
    assert report.summary.mean_recall == pytest.approx(0.625)

    assert [
        record.result.hit
        for record in report.records
        if record.result is not None
    ] == [1, 1, 1, 0]
