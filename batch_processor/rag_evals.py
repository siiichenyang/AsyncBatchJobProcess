import re
from dataclasses import dataclass

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
