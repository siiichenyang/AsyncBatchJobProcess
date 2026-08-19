import asyncio

import httpx
import pytest
from pydantic import ValidationError

from batch_processor.api import EvalCaseRequest, app


async def request_health() -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.get("/health")


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
