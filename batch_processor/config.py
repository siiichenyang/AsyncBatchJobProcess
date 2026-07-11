from dataclasses import dataclass


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
