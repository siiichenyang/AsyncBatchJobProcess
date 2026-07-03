import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass

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
        timeout_seconds=TASK_TIMEOUT_SECONDS) -> TaskResult:
    async with semaphore:
        for attempt_index in range(MAX_RETRIES + 1):
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
            latency_seconds=timeout_seconds,
            retry_count=MAX_RETRIES,
        )


async def main():
    logger.info("Batch processor started.")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    with open(INPUT_PATH, "r", encoding="utf-8") as input_file, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        input_tasks = []
        results = []
        for line in input_file:
            try:
                input_data = json.loads(line)
                task = process_task_with_semaphore(input_data, semaphore)
                input_tasks.append(task)
            except json.JSONDecodeError as exc:
                task_result = TaskResult(
                    name="<invalid-json>",
                    status="error",
                    result={},
                    error=str(exc),
                    latency_seconds=0.0,
                    retry_count=0,
                )
                logger.warning("Invalid JSON.")
                results.append(task_result)
        processed_results = await asyncio.gather(*input_tasks)
        results.extend(processed_results)
        for data in results:
            output_task_result = asdict(data)
            print(output_task_result)
            output_data = json.dumps(output_task_result, ensure_ascii=False)
            output_file.write(output_data + "\n")


if __name__ == "__main__":
    asyncio.run(main())