import pytest

from batch_processor.rag import build_rag_prompt
from batch_processor.vector_store import SearchResult
from batch_processor.chunking import TextChunk


def test_two_search_results():
    sources = [
        SearchResult(
            chunk=TextChunk(
                document_id="doc-1",
                chunk_index=1,
                text="apple",
                start_word=2,
                end_word=3,
            ),
            score=0.98,
        ),
        SearchResult(
            chunk=TextChunk(
                document_id="doc-1",
                chunk_index=0,
                text="applelll",
                start_word=0,
                end_word=1,
            ),
            score=0.2,
        ),
    ]

    query = "Do we have a fruit?"
    rag_prompt = build_rag_prompt(
        query=query,
        sources=sources,
    )

    assert query in rag_prompt.prompt
    assert "apple" in rag_prompt.prompt
    assert "applelll" in rag_prompt.prompt
    assert "[doc-1#1]" in rag_prompt.prompt
    assert "[doc-1#0]" in rag_prompt.prompt
    assert (
        rag_prompt.prompt.index("[doc-1#1]")
        < rag_prompt.prompt.index("[doc-1#0]")
    )

    assert isinstance(rag_prompt.sources, tuple)
    assert rag_prompt.sources == tuple(sources)


def test_empty_context():
    rag_prompt = build_rag_prompt(
        query="What's weather today?",
        sources=[],
    )

    assert "No context was retrieved." in rag_prompt.prompt
    assert rag_prompt.sources == ()


@pytest.mark.parametrize("empty_query", ["", "   ", "\n"])
def test_rejects_empty_query(empty_query):
    sources = [
        SearchResult(
            chunk=TextChunk(
                document_id="doc-1",
                chunk_index=1,
                text="apple",
                start_word=2,
                end_word=3,
            ),
            score=0.98,
        ),
    ]

    with pytest.raises(ValueError, match="non-empty"):
        build_rag_prompt(
            query=empty_query,
            sources=sources,
        )
