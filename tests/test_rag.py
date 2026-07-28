import asyncio

import pytest

from batch_processor.rag import (
    build_rag_prompt,
    answer_rag_query,
)
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


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.received_query = None
        self.received_top_k = None
        self.is_called = False

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SearchResult]:
        self.is_called = True
        self.received_query = query
        self.received_top_k = top_k

        return self.results


class RecordingLLMClient:
    def __init__(self, answer):
        self.answer = answer
        self.prompt = None
        self.is_called = False

    async def generate(self, prompt: str) -> str:
        self.is_called = True
        self.prompt = prompt

        return self.answer


def test_rag_process():
    query = "Do we have a fruit?"
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
    retriever = StubRetriever(sources)
    llm_client = RecordingLLMClient("Yes. We have an apple.")

    rag_answer = asyncio.run(
        answer_rag_query(
            query,
            retriever,
            llm_client,
            top_k=2,
        )
    )

    assert retriever.received_top_k == 2
    assert retriever.received_query == query
    assert all(
        citation in llm_client.prompt
        for citation in ("[doc-1#1]", "[doc-1#0]")
    )
    assert all(
        context in llm_client.prompt
        for context in ("apple", "applelll")
    )
    assert query in llm_client.prompt

    assert isinstance(rag_answer.sources, tuple)
    assert rag_answer.sources == tuple(sources)
    assert rag_answer.query == query
    assert rag_answer.answer == "Yes. We have an apple."
    assert rag_answer.prompt == llm_client.prompt


@pytest.mark.parametrize(
    "top_k, query, match",
    [
        (0, "Do we have a fruit?", "top_k"),
        (2, "", "query"),
    ]
)
def test_fast_fail(top_k, query, match):
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
    retriever = StubRetriever(sources)
    llm_client = RecordingLLMClient("Yes. We have an apple.")

    with pytest.raises(ValueError, match=match):
        asyncio.run(
            answer_rag_query(
                query,
                retriever,
                llm_client,
                top_k=top_k,
            )
        )

    assert not retriever.is_called
    assert not llm_client.is_called
