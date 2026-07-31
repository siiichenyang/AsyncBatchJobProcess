import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Self

from batch_processor.rag import (
    RAGAnswer,
    format_source_label,
)
from batch_processor.retrieval import Retriever
from batch_processor.llm_client import LLMClient
from batch_processor.rag import answer_rag_query


@dataclass(frozen=True)
class CitationEvalResult:
    citations: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    passed: bool


def evaluate_citations(
    answer: RAGAnswer,
) -> CitationEvalResult:
    pattern = r"\[[^\[\]#\n]+#[^\[\]#\n]+\]"

    answer_citations = re.findall(pattern, answer.answer)

    expected_citations = [
        format_source_label(source)
        for source in answer.sources
    ]

    invalid_citations = []
    for answer_citation in answer_citations:
        if answer_citation not in expected_citations:
            invalid_citations.append(answer_citation)

    passed = bool(answer_citations) and not invalid_citations

    return CitationEvalResult(
        citations=tuple(answer_citations),
        invalid_citations=tuple(invalid_citations),
        passed=passed,
    )


@dataclass(frozen=True)
class AnswerContentEvalResult:
    required_phrases: tuple[str, ...]
    missing_phrases: tuple[str, ...]
    passed: bool


def evaluate_answer_content(
    answer: RAGAnswer,
    required_phrases: Sequence[str],
) -> AnswerContentEvalResult:
    if not required_phrases:
        raise ValueError("required_phrases must be non-empty")

    if any(
        not isinstance(required_phrase, str) or not required_phrase.strip()
        for required_phrase in required_phrases
    ):
        raise ValueError("required_phrase must be non-empty str")

    normalized_answer = answer.answer.casefold()

    phrases = tuple(
        phrase.strip()
        for phrase in required_phrases
    )

    missing_phrases = tuple(
        phrase
        for phrase in phrases
        if phrase.casefold() not in normalized_answer
    )

    return AnswerContentEvalResult(
        required_phrases=phrases,
        missing_phrases=missing_phrases,
        passed=not missing_phrases,
    )


@dataclass(frozen=True)
class RAGEvalCase:
    name: str
    query: str
    required_phrases: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not isinstance(data, dict) or not data:
            raise ValueError("data must be non-empty dict")

        name = data.get("name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be non-empty str")

        query = data.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty str")

        required_phrases = data.get("required_phrases", [])
        if not isinstance(required_phrases, list) or not required_phrases:
            raise ValueError("required_phrases must be non-empty list")

        if any(
            not isinstance(phrase, str) or not phrase.strip()
            for phrase in required_phrases
        ):
            raise ValueError("required_phrases must contain non-empty strings")

        return cls(
            name=name.strip(),
            query=query.strip(),
            required_phrases=tuple(
                phrase.strip()
                for phrase in required_phrases
            )
        )


@dataclass(frozen=True)
class RAGEvalResult:
    name: str
    top_k: int
    rag_answer: RAGAnswer
    content_result: AnswerContentEvalResult
    citation_result: CitationEvalResult


async def evaluate_rag_case(
    case: RAGEvalCase,
    retriever: Retriever,
    llm_client: LLMClient,
    *,
    top_k: int,
) -> RAGEvalResult:
    rag_answer = await answer_rag_query(
        case.query,
        retriever,
        llm_client,
        top_k=top_k,
    )
    content_result = evaluate_answer_content(
        rag_answer,
        case.required_phrases,
    )
    citation_result = evaluate_citations(rag_answer)
    return RAGEvalResult(
        name=case.name,
        top_k=top_k,
        rag_answer=rag_answer,
        content_result=content_result,
        citation_result=citation_result,
    )
