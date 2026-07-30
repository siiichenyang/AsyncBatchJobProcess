from batch_processor.rag_evals import (
    evaluate_citations,
    evaluate_answer_content,
)
from batch_processor.vector_store import SearchResult
from batch_processor.rag import (
    RAGAnswer,
    build_rag_prompt,
)
from batch_processor.chunking import TextChunk


def test_one_valid_citation():
    query = "good query"
    sources = (
        SearchResult(
            chunk=TextChunk(
                document_id="doc-1",
                chunk_index=0,
                text="Nothing is impossible.",
                start_word=2,
                end_word=3,
            ),
            score=1.0,
        ),
        SearchResult(
            chunk=TextChunk(
                document_id="doc-1",
                chunk_index=1,
                text="Python is great.",
                start_word=2,
                end_word=3,
            ),
            score=0.98,
        ),
    )

    answer = RAGAnswer(
        query="good query",
        answer="Apple is available [doc-1#0].",
        prompt=build_rag_prompt(query, sources).prompt,
        sources=sources,
    )

    result = evaluate_citations(answer)

    assert result.citations == ("[doc-1#0]",)
    assert result.invalid_citations == ()
    assert result.passed is True

    assert "[doc-1#1]\nPython is great." in answer.prompt


def test_one_valid_and_one_invalid_citation():
    query = "good query"
    sources = (
        SearchResult(
            chunk=TextChunk(
                document_id="doc-1",
                chunk_index=0,
                text="Nothing is impossible.",
                start_word=2,
                end_word=3,
            ),
            score=1.0,
        ),
        SearchResult(
            chunk=TextChunk(
                document_id="doc-1",
                chunk_index=1,
                text="Python is great.",
                start_word=2,
                end_word=3,
            ),
            score=0.98,
        ),
    )

    answer = RAGAnswer(
        query="good query",
        answer="Apple is available [doc-1#0] [other#9].",
        prompt=build_rag_prompt(query, sources).prompt,
        sources=sources,
    )

    result = evaluate_citations(answer)

    assert result.citations == ("[doc-1#0]", "[other#9]",)
    assert result.invalid_citations == ("[other#9]",)
    assert result.passed is False


def test_all_required_passed():
    answer = RAGAnswer(
        query="good query",
        answer="Apple is available [doc-1#0] [other#9].",
        prompt="a prompt",
        sources=(),
    )

    result = evaluate_answer_content(answer, ["apple", "available"])

    assert result.required_phrases == ("apple", "available",)
    assert result.missing_phrases == ()
    assert result.passed is True


def test_missing_required():
    answer = RAGAnswer(
        query="good query",
        answer="Apple is available [doc-1#0] [other#9].",
        prompt="a prompt",
        sources=(),
    )

    result = evaluate_answer_content(answer, ["apple", "missing"])

    assert result.required_phrases == ("apple", "missing",)
    assert result.missing_phrases == ("missing",)
    assert result.passed is False
