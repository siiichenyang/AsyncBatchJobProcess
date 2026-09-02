import asyncio

import httpx
import pytest
from pydantic import ValidationError

from batch_processor.api import EvalCaseRequest, app
from batch_processor.eval_run_store import EvalRunStore
from batch_processor.llm_client import FakeLLMClient, close_llm_client


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


async def request_eval_summary(run_id: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.get(f"/evals/{run_id}/summary")


@pytest.fixture(autouse=True)
def reset_llm_client(eval_run_db_path, monkeypatch):
    monkeypatch.setenv("EVAL_RUN_DB_PATH", eval_run_db_path)
    app.state.eval_run_store = EvalRunStore(eval_run_db_path)
    app.state.llm_client = FakeLLMClient()
    yield
    app.state.llm_client = None
    app.state.eval_run_store = None


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


def test_run_evals_returns_results_and_summary():
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
    assert "run_id" in body


def test_run_evals_persists_summary_for_lookup():
    response = asyncio.run(
        request_eval_run(
            {
                "cases": [
                    {
                        "name": "persistent case",
                        "prompt": "first prompt",
                        "expected": "fake response",
                    },
                ],
            }
        )
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]

    # Simulate a restart or another uvicorn worker: open a brand-new store
    # pointed at the same SQLite file and confirm the summary is still readable.
    db_path = app.state.eval_run_store.db_path
    old_store = app.state.eval_run_store
    old_store.close()
    app.state.eval_run_store = EvalRunStore(db_path)

    summary_response = asyncio.run(request_eval_summary(run_id))

    assert summary_response.status_code == 200
    assert summary_response.json() == response.json()["summary"]


def test_get_eval_summary_returns_404_for_missing_run():
    response = asyncio.run(request_eval_summary("missing-run-id"))

    assert response.status_code == 404


class CloseTrackingLLMClient:
    def __init__(self):
        self.closed = False

    async def generate(self, prompt: str) -> str:
        return "fake response"

    async def close(self) -> None:
        self.closed = True


def test_run_evals_reuses_shared_client():
    client = CloseTrackingLLMClient()
    app.state.llm_client = client

    response = asyncio.run(
        request_eval_run(
            {
                "cases": [
                    {
                        "name": "shared case",
                        "prompt": "first prompt",
                    },
                ],
            }
        )
    )

    assert response.status_code == 200
    assert client.closed is False

    asyncio.run(close_llm_client(client))
    assert client.closed is True


def test_lifespan_creates_and_closes_shared_client(monkeypatch):
    client = CloseTrackingLLMClient()
    monkeypatch.setattr(
        "batch_processor.api.create_llm_client",
        lambda config: client,
    )

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            assert app.state.llm_client is client

    asyncio.run(run_lifespan())

    assert client.closed is True


def test_lifespan_requests_share_same_client(monkeypatch):
    client = CloseTrackingLLMClient()
    monkeypatch.setattr(
        "batch_processor.api.create_llm_client",
        lambda config: client,
    )

    async def run_lifespan_requests():
        async with app.router.lifespan_context(app):
            first_response = await request_eval_run(
                {
                    "cases": [
                        {
                            "name": "first request",
                            "prompt": "first prompt",
                        },
                    ],
                }
            )
            second_response = await request_eval_run(
                {
                    "cases": [
                        {
                            "name": "second request",
                            "prompt": "second prompt",
                        },
                    ],
                }
            )

            assert first_response.status_code == 200
            assert second_response.status_code == 200
            assert app.state.llm_client is client
            assert client.closed is False

    asyncio.run(run_lifespan_requests())

    assert client.closed is True


class FailingCloseLLMClient:
    async def generate(self, prompt: str) -> str:
        return "fake response"

    async def close(self) -> None:
        raise RuntimeError("close failed")


def test_lifespan_ignores_close_errors(monkeypatch):
    monkeypatch.setattr(
        "batch_processor.api.create_llm_client",
        lambda config: FailingCloseLLMClient(),
    )

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(run_lifespan())


def test_run_evals_rejects_empty_cases():
    response = asyncio.run(request_eval_run({"cases": []}))

    assert response.status_code == 422
