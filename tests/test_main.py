import asyncio
import json
from dataclasses import asdict

from batch_processor.main import (
    process_task,
    process_task_with_semaphore,
    TaskResult,
)


def test_process_task_success():
    result = asyncio.run(process_task({"name": "test task"}))
    assert result.status == "success"
    assert result.name == "test task"
    assert result.error is None
    assert result.retry_count == 0


def test_process_task_missing_name():
    result = asyncio.run(process_task({"description": "missing name"}))
    assert result.status == "error"
    assert result.name == "<unknown>"
    assert result.result == {}
    assert result.error is not None
    assert "missing required field" in result.error
    assert result.retry_count == 0


def test_process_task_timeout_retry():
    semaphore = asyncio.Semaphore(1)
    timeout_seconds = 0.05
    max_retries = 1
    result = asyncio.run(
        process_task_with_semaphore(
            {"name": "slow task"},
            semaphore,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
    )
    assert result.status == "error"
    assert result.error == "task timed out"
    assert result.retry_count == max_retries
    assert result.latency_seconds >= (max_retries + 1) * timeout_seconds


def test_jsonl_output_format():
    result = TaskResult(
        name="jsonl output format",
        status="success",
        result={},
        error=None,
        latency_seconds=0.0,
        retry_count=0,
    )
    json_line = json.dumps(asdict(result), ensure_ascii=False)
    loaded = json.loads(json_line)
    assert loaded["name"] == "jsonl output format"
    assert loaded["status"] == "success"
    assert loaded["error"] is None
    assert loaded["retry_count"] == 0
