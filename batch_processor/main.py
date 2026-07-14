import asyncio
import logging
import time
from dataclasses import asdict, dataclass

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
)
from batch_processor.evals import Summary
from batch_processor.task_models import TaskCase


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    name: str
    status: str
    result: dict
    error: str | None
    latency_seconds: float
    retry_count: int
    passed: bool | None = None


async def process_task(
    task: object,
    llm_client: LLMClient,
) -> TaskResult:
    start_time = time.perf_counter()
    try:
        task_case = TaskCase.from_dict(task)
    except ValueError as exc:
        err_str = str(exc)
        logger.warning(err_str)
        return TaskResult(
            name=get_task_name(task),
            status="error",
            result={},
            error=err_str,
            latency_seconds=time.perf_counter() - start_time,
            retry_count=0,
        )

    try:
        response = await llm_client.generate(task_case.prompt)
    except Exception as exc:
        err_str = f'{type(exc).__name__}: {exc}'
        task_result = TaskResult(
            name=task_case.name,
            status="error",
            result={},
            error=err_str,
            latency_seconds=time.perf_counter() - start_time,
            retry_count=0,
        )
        logger.warning(err_str)
        return task_result
    else:
        task_result = TaskResult(
            name=task_case.name,
            status="success",
            result={"output": response},
            error=None,
            latency_seconds=time.perf_counter() - start_time,
            retry_count=0,
        )

    if task_case.expected is not None:
        task_result.passed = task_case.expected == response
    logger.info(f"Processed task successfully: {task_case.name}")
    return task_result


def get_task_name(task: object) -> str:
    if not isinstance(task, dict):
        return "<unknown>"

    name = task.get("name")
    if not isinstance(name, str) or not name.strip():
        return "<unknown>"


async def process_task_with_semaphore(
        task,
        semaphore,
        timeout_seconds,
        max_retries,
        llm_client: LLMClient,
) -> TaskResult:
    async with semaphore:
        start_process_time = time.perf_counter()
        for attempt_index in range(max_retries + 1):
            try:
                ret = await asyncio.wait_for(
                    process_task(task, llm_client),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                continue
            else:
                if attempt_index > 0:
                    ret.retry_count = attempt_index
                return ret

        task_name = task.get("name", "<unknown>")
        logger.warning(f"Task timed out: {task_name}")
        return TaskResult(
            name=task_name,
            status="error",
            result={},
            error="task timed out",
            latency_seconds=time.perf_counter() - start_process_time,
            retry_count=max_retries,
        )


async def run_batch(config: BatchConfig, llm_client: LLMClient) -> list[TaskResult]:
    logger.info("Batch processor started.")
    semaphore = asyncio.Semaphore(config.max_concurrency)

    records = read_jsonl(config.input_path)
    results = []
    input_tasks = []
    for record in records:
        if record.data is not None:
            task = process_task_with_semaphore(
                record.data,
                semaphore,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
                llm_client=llm_client,
            )
            input_tasks.append(task)
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
    processed_results = await asyncio.gather(*input_tasks)
    results.extend(processed_results)
    output_records = [asdict(result) for result in results]
    write_jsonl(config.output_path, output_records)

    summary = build_summary(results)

    write_json(config.summary_path, summary)

    return results


def build_summary(results: list[TaskResult]) -> dict[str, int | float]:
    summary = Summary()

    for result in results:
        summary.total += 1
        if result.status == "success":
            summary.success += 1
        else:
            summary.error += 1
        if result.passed is not None:
            summary.evaluated += 1
            if result.passed:
                summary.passed += 1
            else:
                summary.failed += 1

    if summary.evaluated > 0:
        summary.pass_rate = summary.passed / summary.evaluated

    return asdict(summary)


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
    await run_batch(BatchConfig(), llm_client)


if __name__ == "__main__":
    asyncio.run(main())
