from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkReference:
    document_id: str
    chunk_index: int


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
