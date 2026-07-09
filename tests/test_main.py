import asyncio
import pytest
import json
from dataclasses import asdict

from batch_processor.main import (
    run_batch,
    process_task,
    process_task_with_semaphore,
    TaskResult,
    validate_task_input,
)
from batch_processor.config import BatchConfig
from batch_processor.llm_client import FakeLLMClient


def test_process_task_success():
    result = asyncio.run(process_task({"name": "test task"}, FakeLLMClient("expected response")))
    assert result.status == "success"
    assert result.name == "test task"
    assert result.error is None
    assert result.retry_count == 0
    assert result.result == {"output": "expected response"}


def test_process_task_missing_name():
    result = asyncio.run(process_task({"description": "missing name"}, FakeLLMClient()))
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
            max_retries=max_retries,
            llm_client=FakeLLMClient(),
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


def test_run_batch_writes_success_and_error_results(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    input_path.write_text(
        '{"name": "write a story.", "description": "a story about lost and find."}\n'
        '{"description": "a task without name"}\n'
        '{name: "this is an invalid task"}\n',
        encoding="utf-8",
    )

    results = asyncio.run(
        run_batch(
            BatchConfig(
                input_path=input_path,
                output_path=output_path,
            ),
            FakeLLMClient("expected response"),
        )
    )

    assert len(results) == 3

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    output_records = [json.loads(line) for line in output_lines]

    assert len(output_records) == 3

    assert any(
        record["name"] == "write a story."
        and record["status"] == "success"
        and record["error"] is None
        and record["result"]["output"] == "expected response"
        for record in output_records
    )

    assert any(
        record["name"] == "<unknown>"
        and record["status"] == "error"
        and record["error"] is not None
        and "missing required field" in record["error"]
        for record in output_records
    )

    assert any(
        record["name"] == "<invalid-json>"
        and record["status"] == "error"
        and record["error"] is not None
        and "Invalid JSON line" in record["error"]
        for record in output_records
    )


def test_basic_config_check():
    with pytest.raises(ValueError, match="max_concurrency"):
        BatchConfig(max_concurrency=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        BatchConfig(timeout_seconds=0)
    with pytest.raises(ValueError, match="max_retries"):
        BatchConfig(max_retries=-1)


def test_validate_task_input_returns_name():
    assert validate_task_input(
        {"name": "what's time now", "description": "query time"}
    ) == "what's time now"


def test_validate_task_input_requires_name():
    with pytest.raises(KeyError):
        validate_task_input({"description": "without name"})
