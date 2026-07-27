import asyncio
from pathlib import Path

import pytest

from batch_processor.retrieval_eval_comparison import (
    compare_chunking_strategies,
    ChunkingStrategy,
)
from batch_processor.documents import load_text_document
from batch_processor.embeddings import DeterministicEmbeddingClient


def test_comparison():
    root_path = Path(__file__).resolve().parents[1]
    document_path = root_path / "data" / "rag" / "backend_topics.txt"
    eval_input_path = root_path / "data" / "rag" / "retrieval_eval_cases.jsonl"

    document = load_text_document(
        document_path,
        document_id="backend_topics",
    )

    strategies = (
        ChunkingStrategy(
            name="no-overlap",
            chunk_size=6,
            overlap=0,
        ),
        ChunkingStrategy(
            name="overlap-2",
            chunk_size=6,
            overlap=2,
        ),
    )

    results = asyncio.run(
        compare_chunking_strategies(
            document,
            str(eval_input_path),
            DeterministicEmbeddingClient(dimensions=4096),
            strategies,
            k=1,
        )
    )

    assert isinstance(results, tuple)
    assert len(results) == 2

    assert [result.strategy.name for result in results] == [
        "no-overlap",
        "overlap-2",
    ]

    assert results[0].report.summary.hit_rate == pytest.approx(0.75)
    assert results[0].report.summary.mean_recall == pytest.approx(0.625)

    assert results[1].report.summary.hit_rate == pytest.approx(0.75)
    assert results[1].report.summary.mean_recall == pytest.approx(0.75)


def test_comparison_rejects_empty_strategies():
    root_path = Path(__file__).resolve().parents[1]
    document_path = root_path / "data" / "rag" / "backend_topics.txt"
    eval_input_path = root_path / "data" / "rag" / "retrieval_eval_cases.jsonl"

    document = load_text_document(
        document_path,
        document_id="backend_topics",
    )

    with pytest.raises(ValueError, match="empty"):
        asyncio.run(
            compare_chunking_strategies(
                document,
                str(eval_input_path),
                DeterministicEmbeddingClient(dimensions=4096),
                [],
                k=1,
            )
        )
