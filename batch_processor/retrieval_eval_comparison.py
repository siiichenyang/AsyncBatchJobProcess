from dataclasses import dataclass
from collections.abc import Sequence

from batch_processor.retrieval_eval_runner import (
    RetrievalEvalReport,
    run_retrieval_eval_file,
)
from batch_processor.embeddings import EmbeddingClient
from batch_processor.vector_store import InMemoryVectorStore
from batch_processor.documents import Document
from batch_processor.retrieval import Retriever


@dataclass(frozen=True)
class ChunkingStrategy:
    name: str
    chunk_size: int
    overlap: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("'name' must be non-empty str")

        if self.chunk_size <= 0:
            raise ValueError("'chunk_size' must be greater than zero")

        if not (0 <= self.overlap < self.chunk_size):
            raise ValueError("'overlap' invalid")


@dataclass(frozen=True)
class ChunkingComparisonResult:
    strategy: ChunkingStrategy
    report: RetrievalEvalReport


async def compare_chunking_strategies(
    document: Document,
    eval_input_path: str,
    embedding_client: EmbeddingClient,
    strategies: Sequence[ChunkingStrategy],
    *,
    k: int,
) -> tuple[ChunkingComparisonResult, ...]:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not strategies:
        raise ValueError("strategies must be non-empty")

    results = []
    for strategy in strategies:
        retriever = Retriever(
            embedding_client,
            InMemoryVectorStore(),
        )

        await retriever.index_document(
            document,
            chunk_size=strategy.chunk_size,
            overlap=strategy.overlap,
        )

        report = await run_retrieval_eval_file(
            eval_input_path,
            retriever,
            k=k,
        )

        results.append(
            ChunkingComparisonResult(
                strategy,
                report,
            )
        )

    return tuple(results)
