import asyncio
import logging
import time
from dataclasses import asdict, dataclass

from batch_processor.jsonl_io import (
    read_jsonl,
    write_jsonl,
)
from batch_processor.config import BatchConfig
from batch_processor.llm_client import (
    LLMClient,
    FakeLLMClient,
)


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


async def process_task(task, llm_client: LLMClient) -> TaskResult:
    start_time = time.perf_counter()
    try:
        task_name = validate_task_input(task)
    except KeyError as exc:
        err_str = f"missing required field: {str(exc)}"
        task_result = TaskResult(
            name=task.get("name", "<unknown>"),
            status="error",
            result={},
            error=err_str,
            latency_seconds=time.perf_counter() - start_time,
            retry_count=0,
        )
        logger.warning(err_str)
        return task_result

    await asyncio.sleep(0.1)
    response = await llm_client.generate(task["name"])

    task_result = TaskResult(
        name=task_name,
        status="success",
        result={"output": response},
        error=None,
        latency_seconds=time.perf_counter() - start_time,
        retry_count=0,
    )
    if "expected" in task:
        task_result.passed = task["expected"] == response
    logger.info(f"Processed task successfully: {task_name}")
    return task_result


def validate_task_input(task: dict) -> str:
    if "name" not in task:
        raise KeyError("name")
    return task["name"]


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
    return results


async def main():
    await run_batch(BatchConfig(), FakeLLMClient())


if __name__ == "__main__":
    asyncio.run(main())
