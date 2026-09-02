import asyncio

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import (
    BaseModel,
    field_validator,
    ValidationInfo,
    Field,
)

from batch_processor.config import LLMConfig
from batch_processor.eval_run_store import EvalRunStore
from batch_processor.eval_service import build_summary, run_eval_cases
from batch_processor.llm_client import LLMClient, close_llm_client
from batch_processor.main import create_llm_client


class HealthResponse(BaseModel):
    status: str


class EvalCaseRequest(BaseModel):
    name: str
    prompt: str
    expected: str | None = None

    @field_validator("name", "prompt")
    @classmethod
    def reject_blank_string(
        cls,
        value: str,
        info: ValidationInfo,
    ) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")

        return value


class EvalRunRequest(BaseModel):
    cases: list[EvalCaseRequest] = Field(min_length=1)
    max_concurrency: int = Field(default=2, ge=1)
    timeout_seconds: float = Field(default=30, gt=0)
    max_retries: int = Field(default=2, ge=0)


class EvalResultResponse(BaseModel):
    name: str
    status: Literal["success", "error"]
    result: dict[str, object]
    error: str | None
    latency_seconds: float
    retry_count: int
    passed: bool | None


class EvalSummaryResponse(BaseModel):
    total: int
    success: int
    error: int
    evaluated: int
    passed: int
    failed: int
    pass_rate: float


class EvalRunResponse(BaseModel):
    run_id: str
    results: list[EvalResultResponse]
    summary: EvalSummaryResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.eval_run_store = EvalRunStore.from_env()
    app.state.llm_client = create_llm_client(LLMConfig.from_env())
    try:
        yield
    finally:
        app.state.eval_run_store.close()
        await close_llm_client(app.state.llm_client)


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_eval_run_store(request: Request) -> EvalRunStore:
    return request.app.state.eval_run_store


app = FastAPI(title="LLM Agent Evaluation Backend", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/evals/run", response_model=EvalRunResponse)
async def run_evals(
    request: EvalRunRequest,
    llm_client: LLMClient = Depends(get_llm_client),
    eval_run_store: EvalRunStore = Depends(get_eval_run_store),
) -> EvalRunResponse:
    results = await run_eval_cases(
        [case.model_dump() for case in request.cases],
        llm_client,
        max_concurrency=request.max_concurrency,
        timeout_seconds=request.timeout_seconds,
        max_retries=request.max_retries,
    )
    result_dicts = [asdict(result) for result in results]
    summary = build_summary(results)
    run_id = str(uuid4())
    await asyncio.to_thread(eval_run_store.save_run, run_id, result_dicts, summary)

    return EvalRunResponse(
        run_id=run_id,
        results=[EvalResultResponse(**result) for result in result_dicts],
        summary=EvalSummaryResponse(**summary),
    )


@app.get("/evals/{run_id}/summary", response_model=EvalSummaryResponse)
async def get_eval_summary(
    run_id: str,
    eval_run_store: EvalRunStore = Depends(get_eval_run_store),
) -> EvalSummaryResponse:
    summary = await asyncio.to_thread(eval_run_store.get_summary, run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return EvalSummaryResponse(**summary)
