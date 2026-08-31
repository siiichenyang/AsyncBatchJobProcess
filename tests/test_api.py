import asyncio

import httpx
import pytest
from pydantic import ValidationError

from batch_processor.api import EvalCaseRequest, app
from batch_processor.llm_client import FakeLLMClient


async def request_health() -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.get("/health")


async def request_eval_run(payload: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.post("/evals/run", json=payload)


def test_health():
    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_eval_case_rejects_blank_without_normalizing_valid_prompt():
    original_prompt = "  preserve this spacing  "

    case = EvalCaseRequest(name="spacing case", prompt=original_prompt)

    assert case.prompt == original_prompt
    with pytest.raises(ValidationError, match="prompt must not be blank"):
        EvalCaseRequest(name="blank prompt", prompt="   ")


def test_run_evals_returns_results_and_summary(monkeypatch):
    monkeypatch.setattr(
        "batch_processor.api.create_llm_client",
        lambda config: FakeLLMClient(),
    )
    response = asyncio.run(
        request_eval_run(
            {
                "cases": [
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
            }
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert [result["name"] for result in body["results"]] == [
        "passing case",
        "failing case",
    ]
    assert [result["passed"] for result in body["results"]] == [True, False]
    assert body["summary"] == {
        "total": 2,
        "success": 2,
        "error": 0,
        "evaluated": 2,
        "passed": 1,
        "failed": 1,
        "pass_rate": 0.5,
    }


def test_run_evals_rejects_empty_cases():
    response = asyncio.run(request_eval_run({"cases": []}))

    assert response.status_code == 422
