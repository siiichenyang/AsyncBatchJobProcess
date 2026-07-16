from collections.abc import Sequence
from dataclasses import dataclass

from batch_processor.chunking import TextChunk
from batch_processor.similarity import cosine_similarity


@dataclass(frozen=True)
class VectorRecord:
    chunk: TextChunk
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float


class InMemoryVectorStore:
    def __init__(self):
        self._records: list[VectorRecord] = []

    def add(self, chunk: TextChunk, embedding: Sequence[float]) -> None:
        record = VectorRecord(chunk=chunk, embedding=tuple(embedding))
        self._records.append(record)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        search_vectors: list[SearchResult] = []
        for record in self._records:
            score = cosine_similarity(query_embedding, record.embedding)
            search_vectors.append(
                SearchResult(
                    chunk=record.chunk,
                    score=score,
                )
            )

        search_vectors.sort(
            key=lambda vector: vector.score,
            reverse=True,
        )

        return search_vectors[:top_k]
