import asyncio
import logging
import time
from dataclasses import asdict, dataclass

from batch_processor.jsonl_io import (
    read_jsonl,
    write_jsonl,
)


INPUT_PATH = "input.jsonl"
OUTPUT_PATH = "output.jsonl"
MAX_CONCURRENCY = 2
TASK_TIMEOUT_SECONDS = 0.2
MAX_RETRIES = 2


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


async def process_task(task) -> TaskResult:
    start_time = time.perf_counter()
    try:
        task_name = task["name"]
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

    task_result = TaskResult(
        name=task_name,
        status="success",
        result={
            "data": 500,
            "msg": "This is a task result."
        },
        error=None,
        latency_seconds=time.perf_counter() - start_time,
        retry_count=0,
    )
    logger.info(f"Processed task successfully: {task_name}")
    return task_result


async def process_task_with_semaphore(
        task,
        semaphore,
        timeout_seconds=TASK_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES) -> TaskResult:
    async with semaphore:
        start_process_time = time.perf_counter()
        for attempt_index in range(max_retries + 1):
            try:
                ret = await asyncio.wait_for(process_task(task), timeout=timeout_seconds)
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


async def run_batch(
    input_path: str,
    output_path: str,
    max_concurrency: int = MAX_CONCURRENCY,
    timeout_seconds: float = TASK_TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
) -> list[TaskResult]:
    logger.info("Batch processor started.")
    semaphore = asyncio.Semaphore(max_concurrency)

    records = read_jsonl(input_path)
    results = []
    input_tasks = []
    for record in records:
        if record.data is not None:
            task = process_task_with_semaphore(
                record.data,
                semaphore,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
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
    write_jsonl(output_path, output_records)
    return results


async def main():
    await run_batch(
        INPUT_PATH,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    asyncio.run(main())
