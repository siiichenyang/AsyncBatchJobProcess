import asyncio
import logging
from dataclasses import asdict

from batch_processor.jsonl_io import (
    read_jsonl,
    write_jsonl,
    write_json,
)
from batch_processor.config import (
    BatchConfig,
    LLMConfig,
)
from batch_processor.llm_client import (
    LLMClient,
    FakeLLMClient,
    OpenAILLMClient,
    close_llm_client,
)
from batch_processor.eval_service import (
    TaskResult,
    build_summary,
    run_eval_cases,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_batch(config: BatchConfig, llm_client: LLMClient) -> list[TaskResult]:
    logger.info("Batch processor started.")
    records = read_jsonl(config.input_path)
    results = []
    cases = []
    for record in records:
        if record.data is not None:
            cases.append(record.data)
        elif record.error is not None:
            results.append(
                TaskResult(
                    name="<invalid-json>",
                    status="error",
                    result={},
                    error=record.error,
                    latency_seconds=0.0,
                    retry_count=0,
                )
            )
    processed_results = await run_eval_cases(
        cases,
        llm_client,
        max_concurrency=config.max_concurrency,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    results.extend(processed_results)
    output_records = [asdict(result) for result in results]
    write_jsonl(config.output_path, output_records)

    summary = build_summary(results)

    write_json(config.summary_path, summary)

    return results


def create_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == "fake":
        return FakeLLMClient()
    if config.provider == "openai":
        return OpenAILLMClient(model=config.model, api_key=config.api_key)

    raise ValueError(
        f"LLMConfig provider not supported: {config.provider!r}"
    )


async def main() -> None:
    llm_config = LLMConfig.from_env()
    llm_client = create_llm_client(llm_config)
    try:
        await run_batch(BatchConfig(), llm_client)
    finally:
        await close_llm_client(llm_client)


if __name__ == "__main__":
    asyncio.run(main())
