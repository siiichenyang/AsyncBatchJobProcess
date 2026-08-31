from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI
from pydantic import (
    BaseModel,
    field_validator,
    ValidationInfo,
    Field,
)

from batch_processor.config import LLMConfig
from batch_processor.eval_service import build_summary, run_eval_cases
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
    results: list[EvalResultResponse]
    summary: EvalSummaryResponse


app = FastAPI(title="LLM Agent Evaluation Backend")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/evals/run", response_model=EvalRunResponse)
async def run_evals(request: EvalRunRequest) -> EvalRunResponse:
    llm_client = create_llm_client(LLMConfig.from_env())
    results = await run_eval_cases(
        [case.model_dump() for case in request.cases],
        llm_client,
        max_concurrency=request.max_concurrency,
        timeout_seconds=request.timeout_seconds,
        max_retries=request.max_retries,
    )
    summary = build_summary(results)

    return EvalRunResponse(
        results=[
            EvalResultResponse(**asdict(result))
            for result in results
        ],
        summary=EvalSummaryResponse(**summary),
    )
