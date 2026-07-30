import re
from dataclasses import dataclass
from collections.abc import Sequence

from batch_processor.rag import (
    RAGAnswer,
    format_source_label,
)


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
