from fastapi import FastAPI
from pydantic import (
    BaseModel,
    field_validator,
    ValidationInfo,
    Field,
)
from typing import Literal


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
