import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from batch_processor.evals import Summary
from batch_processor.llm_client import LLMClient
from batch_processor.task_models import TaskCase


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

    return name


async def process_task_with_semaphore(
    task: object,
    semaphore: asyncio.Semaphore,
    timeout_seconds: float,
    max_retries: int,
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

        task_name = get_task_name(task)
        logger.warning(f"Task timed out: {task_name}")
        return TaskResult(
            name=task_name,
            status="error",
            result={},
            error="task timed out",
            latency_seconds=time.perf_counter() - start_process_time,
            retry_count=max_retries,
        )


async def run_eval_cases(
    cases: Sequence[object],
    llm_client: LLMClient,
    *,
    max_concurrency: int,
    timeout_seconds: float,
    max_retries: int,
) -> list[TaskResult]:
    logger.info("Run eval cases.")
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        process_task_with_semaphore(
            case,
            semaphore,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            llm_client=llm_client,
        )
        for case in cases
    ]

    return list(await asyncio.gather(*tasks))


def build_summary(results: Sequence[TaskResult]) -> dict[str, int | float]:
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
