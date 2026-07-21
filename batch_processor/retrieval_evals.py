from typing import Self
from collections.abc import Sequence
from dataclasses import dataclass

from batch_processor.chunking import TextChunk
from batch_processor.retrieval import Retriever


@dataclass(frozen=True)
class ChunkReference:
    document_id: str
    chunk_index: int

    @classmethod
    def from_chunk(cls, chunk: TextChunk) -> Self:
        return cls(
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
        )


@dataclass(frozen=True)
class RetrievalEvalCase:
    name: str
    query: str
    relevant_chunks: tuple[ChunkReference, ...]


def hit_at_k(
    chunks: Sequence[ChunkReference],
    targets: Sequence[ChunkReference],
    k: int,
) -> int:

    return 1 if recall_at_k(
        chunks=chunks,
        targets=targets,
        k=k,
    ) > 0 else 0


def recall_at_k(
    chunks: Sequence[ChunkReference],
    targets: Sequence[ChunkReference],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    relevant_chunks = set(targets)
    if not relevant_chunks:
        raise ValueError("targets must not be empty")

    retrieved_chunks = set(chunks[:k])
    matched_chunks = retrieved_chunks & relevant_chunks

    return len(matched_chunks) / len(relevant_chunks)


@dataclass(frozen=True)
class RetrievalEvalResult:
    name: str
    query: str
    k: int
    retrieved_chunks: tuple[ChunkReference, ...]
    hit: int
    recall: float


async def evaluate_retrieval_case(
    case: RetrievalEvalCase,
    retriever: Retriever,
    *,
    k: int,
) -> RetrievalEvalResult:
    search_results = await retriever.retrieve(
        case.query,
        top_k=k,
    )

    chunk_refs = tuple(
        ChunkReference.from_chunk(chunk.chunk)
        for chunk in search_results
    )

    hit = hit_at_k(chunk_refs, case.relevant_chunks, k)
    recall = recall_at_k(chunk_refs, case.relevant_chunks, k)

    return RetrievalEvalResult(
        name=case.name,
        query=case.query,
        k=k,
        retrieved_chunks=chunk_refs,
        hit=hit,
        recall=recall,
    )
