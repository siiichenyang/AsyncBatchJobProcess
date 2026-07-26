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

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not isinstance(data, dict) or not data:
            raise ValueError("data must be json object")

        document_id = data.get("document_id", "")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("'document_id' must be non empty str")

        chunk_index = data.get("chunk_index", -1)
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
        ):
            raise ValueError("'chunk_index' non-negative integer")

        return cls(
            document_id=document_id,
            chunk_index=chunk_index,
        )


@dataclass(frozen=True)
class RetrievalEvalCase:
    name: str
    query: str
    relevant_chunks: tuple[ChunkReference, ...]

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not isinstance(data, dict) or not data:
            raise ValueError("data must be json object")

        for field_name in ("name", "query"):
            data_str = data.get(field_name, "")
            if not isinstance(data_str, str) or not data_str.strip():
                raise ValueError(f"{field_name!r} must be non empty str")

        relevant_chunks = data.get("relevant_chunks", [])
        if not isinstance(relevant_chunks, list) or not relevant_chunks:
            raise ValueError("'relevant_chunks' must be non empty list")

        return cls(
            name=data["name"],
            query=data["query"],
            relevant_chunks=tuple(
                ChunkReference.from_dict(chunk)
                for chunk in relevant_chunks
            ),
        )


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


@dataclass(frozen=True)
class RelevantSpan:
    document_id: str
    start_word: int
    end_word: int

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not isinstance(data, dict) or not data:
            raise ValueError("data must be json object")

        document_id = data.get("document_id", "")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be non-empty str")

        start_word = data.get("start_word", -1)
        if (
            isinstance(start_word, bool)
            or not isinstance(start_word, int)
            or start_word < 0
        ):
            raise ValueError("invalid start_word")

        end_word = data.get("end_word", -1)
        if (
            isinstance(end_word, bool)
            or not isinstance(end_word, int)
            or end_word <= start_word
        ):
            raise ValueError("invalid end_word")

        return cls(
            document_id=document_id,
            start_word=start_word,
            end_word=end_word,
        )


def chunk_overlap_span(
    chunk: TextChunk,
    span: RelevantSpan,
) -> bool:
    return (
        chunk.document_id == span.document_id
        and span.start_word < chunk.end_word
        and chunk.start_word < span.end_word
    )


def span_recall_at_k(
    chunks: Sequence[TextChunk],
    targets: Sequence[RelevantSpan],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    targets_set = set(targets)
    if not targets_set:
        raise ValueError("targets must not be empty")

    first_k_chunks = set(chunks[:k])
    matches = {
        span
        for span in targets_set
        if any(
            chunk_overlap_span(chunk, span)
            for chunk in first_k_chunks
        )
    }

    return len(matches) / len(targets_set)


def span_hit_at_k(
    chunks: Sequence[TextChunk],
    targets: Sequence[RelevantSpan],
    k: int,
) -> int:
    return 1 if span_recall_at_k(
        chunks,
        targets,
        k,
    ) > 0.0 else 0
