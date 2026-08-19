import asyncio

from batch_processor.eval_service import build_summary, run_eval_cases
from batch_processor.llm_client import FakeLLMClient


def test_run_eval_cases_processes_in_memory_batch():
    results = asyncio.run(
        run_eval_cases(
            [
                {
                    "name": "passing case",
                    "prompt": "first prompt",
                    "expected": "fake response",
                },
                {
                    "name": "failing case",
                    "prompt": "second prompt",
                    "expected": "different response",
                },
            ],
            FakeLLMClient(),
            max_concurrency=2,
            timeout_seconds=1,
            max_retries=0,
        )
    )

    assert [result.name for result in results] == [
        "passing case",
        "failing case",
    ]
    assert [result.passed for result in results] == [True, False]
    assert build_summary(results) == {
        "total": 2,
        "success": 2,
        "error": 0,
        "evaluated": 2,
        "passed": 1,
        "failed": 1,
        "pass_rate": 0.5,
    }
