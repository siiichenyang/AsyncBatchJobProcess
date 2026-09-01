import asyncio

from batch_processor.rag_eval_runner import run_rag_eval_file
from batch_processor.vector_store import SearchResult
from batch_processor.chunking import TextChunk


class StubRetriver:
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SearchResult]:
        if query == "good-case":
            return [SearchResult(
                TextChunk(
                    document_id="doc-1",
                    chunk_index=0,
                    text="text-1",
                    start_word=0,
                    end_word=3,
                ),
                score=1.0,
            )]

        return [SearchResult(
            TextChunk(
                document_id="doc-1",
                chunk_index=1,
                text="text-2",
                start_word=3,
                end_word=6,
            ),
            score=0.5,
        )]


class StubLLMClient:
    async def generate(self, prompt: str) -> str:
        if "good-case" in prompt:
            return "coroutine event loop [doc-1#0]"
        if "bad-case" in prompt:
            return "bad result [doc-1#1]"
        return "something else"

    async def close(self) -> None:
        """No-op for test client."""


def test_rag_eval_runner(tmp_path):
    case_path = tmp_path / "input_cases.jsonl"

    case_path.write_text(
        (
            '{"name": "good case", "query": "good-case", "required_phrases": ["coroutine", "event loop"]}\n'
            '{"name": "bad case", "query": "bad-case", "required_phrases": ["rollback", "isolation"]}\n'
            '{"name": "bad input", "query": 123, "required_phrases": ["x"]}\n'

        ),
        encoding="utf-8",
    )

    retriever = StubRetriver()
    llm_client = StubLLMClient()
    top_k = 1

    report = asyncio.run(
        run_rag_eval_file(
            case_path,
            retriever,
            llm_client,
            k=top_k,
        )
    )

    assert report.summary.total == 3
    assert report.summary.evaluated == 2
    assert report.summary.errors == 1
    assert report.summary.answer_pass_rate == 0.5
    assert report.summary.citation_pass_rate == 1.0
