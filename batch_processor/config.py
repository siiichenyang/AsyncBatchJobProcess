import os
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class BatchConfig:
    input_path: str = "input.jsonl"
    output_path: str = "output.jsonl"
    max_concurrency: int = 2
    timeout_seconds: float = 0.2
    max_retries: int = 2
    summary_path: str = "summary.json"

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be at least 0")


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None
    api_key: str | None

    @classmethod
    def from_env(cls) -> Self:
        provider = os.getenv("LLM_PROVIDER", "fake")

        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL") or None,
            api_key=os.getenv("LLM_API_KEY") or None,
        )

    def __post_init__(self) -> None:
        if self.provider != "fake" and self.api_key is None:
            raise ValueError(
                f"LLM provider {self.provider!r} requires LLM_API_KEY"
            )
