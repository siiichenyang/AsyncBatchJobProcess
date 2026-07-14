import asyncio
import math
import pytest
import json
from dataclasses import asdict

from batch_processor.main import (
    create_llm_client,
    main,
    run_batch,
    process_task,
    process_task_with_semaphore,
    TaskResult,
    TaskCase,
)
from batch_processor.config import (
    BatchConfig,
    LLMConfig,
)
from batch_processor.llm_client import (
    FakeLLMClient,
    OpenAILLMClient,
)


def test_process_task_success():
    result = asyncio.run(
        process_task(
            {"name": "test task", "prompt": "empty",
                "expected": "expected response"},
            FakeLLMClient("expected response"),
        )
    )
    assert result.status == "success"
    assert result.name == "test task"
    assert result.error is None
    assert result.retry_count == 0
    assert result.result == {"output": "expected response"}
    assert result.passed is True


def test_process_task_missing_name():
    result = asyncio.run(
        process_task({"prompt": "missing name"}, FakeLLMClient())
    )
    assert result.status == "error"
    assert result.name == "<unknown>"
    assert result.result == {}
    assert result.error is not None
    assert "missing required field" in result.error
    assert result.retry_count == 0


class SlowLLMClient:
    def __init__(self, sleep_seconds: float = 600):
        self.sleep_seconds = sleep_seconds

    async def generate(self, prompt: str) -> str:
        await asyncio.sleep(self.sleep_seconds)


def test_process_task_timeout_retry():
    semaphore = asyncio.Semaphore(1)
    timeout_seconds = 0.05
    max_retries = 1
    result = asyncio.run(
        process_task_with_semaphore(
            {"name": "slow task", "prompt": "empty"},
            semaphore,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            llm_client=SlowLLMClient(sleep_seconds=0.1),
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
        passed=True,
    )
    json_line = json.dumps(asdict(result), ensure_ascii=False)
    loaded = json.loads(json_line)
    assert loaded["name"] == "jsonl output format"
    assert loaded["status"] == "success"
    assert loaded["error"] is None
    assert loaded["retry_count"] == 0
    assert loaded["passed"] is True


def test_run_batch_writes_success_and_error_results(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    input_path.write_text(
        '{"name": "write a story.", "prompt": "a story about lost and find.", "expected": "expected response"}\n'
        '{"prompt": "a task without name"}\n'
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
        and record["passed"] is True
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


def test_process_task_failed_expected_match(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    input_path.write_text(
        '{"name": "query today\'s weather", "prompt": "query weather in Shanghai", "expected": "another response"}',
        encoding="utf-8",
    )

    results = asyncio.run(
        run_batch(
            BatchConfig(input_path, output_path),
            FakeLLMClient("fake response"),
        )
    )

    assert len(results) == 1

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    output_records = [json.loads(line) for line in output_lines]

    assert output_records[0]["passed"] is False


def test_process_task_without_expected_has_no_passed_value(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    input_path.write_text(
        '{"name": "query today\'s weather", "prompt": "query weather in Shanghai"}',
        encoding="utf-8",
    )

    results = asyncio.run(
        run_batch(
            BatchConfig(input_path, output_path),
            FakeLLMClient("fake response"),
        )
    )

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    output_records = [json.loads(line) for line in output_lines]

    assert len(results) == 1
    assert output_records[0]["passed"] is None


def test_simple_summary(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.json"

    input_path.write_text("""{"name": "query today's weather", "prompt": "query weather in Shanghai", "expected": "fake response"}
{"name": "draw a ascii art", "prompt": "draw a ascii art of an apple"}
{"prompt": "this task has no name"}
{"name": "translate sentence", "prompt": "translate this sentence to chinese", "expected": "wrong response"}
{badkey: "this is an invalid json"}
{"name": "write a poem", "prompt": "write a poem about weather", "expected": "fake response"}
""", encoding="utf-8")

    asyncio.run(
        run_batch(
            BatchConfig(
                input_path=input_path,
                output_path=output_path,
                summary_path=summary_path,
            ),
            FakeLLMClient(),
        )
    )

    summary_str = summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_str)

    assert summary["total"] == 6
    assert summary["success"] == 4
    assert summary["error"] == 2
    assert summary["evaluated"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert math.isclose(summary["pass_rate"], 0.666666666667)


def test_summary_zero_evaluated(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.json"

    input_path.write_text("""{"name": "query today's weather", "prompt": "query weather in Shanghai"}
{"name": "draw a ascii art", "prompt": "draw a ascii art of an apple"}
{"prompt": "this task has no name"}
""", encoding="utf-8")

    asyncio.run(
        run_batch(
            BatchConfig(
                input_path=input_path,
                output_path=output_path,
                summary_path=summary_path,
            ),
            FakeLLMClient(),
        )
    )

    summary_str = summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_str)

    assert summary["total"] == 3
    assert summary["success"] == 2
    assert summary["error"] == 1
    assert summary["evaluated"] == 0
    assert summary["passed"] == 0
    assert summary["failed"] == 0
    assert math.isclose(summary["pass_rate"], 0)


class StubCaseLLMClient:
    def __init__(
        self,
        error: Exception,
        response: str = "fake response",
    ):
        self.response = response
        self.error = error

    async def generate(self, prompt: str) -> str:
        match prompt:
            case "query weather in Shanghai":
                return self.response
            case "draw a ascii art of an apple":
                raise self.error
            case _:
                raise self.error


def test_processed_task_generation_error(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.json"

    input_path.write_text("""{"name": "prompt 1", "prompt": "query weather in Shanghai"}
{"name": "prompt 2", "prompt": "draw a ascii art of an apple", "expected": "error prompt"}
""", encoding="utf-8")

    asyncio.run(
        run_batch(
            BatchConfig(
                input_path=input_path,
                output_path=output_path,
                summary_path=summary_path,
            ),
            StubCaseLLMClient(error=RuntimeError("expected failure")),
        )
    )

    output_str = output_path.read_text(encoding="utf-8").splitlines()
    outputs = [json.loads(line) for line in output_str]

    assert outputs[0]["status"] == "success"
    assert outputs[1]["status"] == "error"
    assert "RuntimeError" in outputs[1]["error"]
    assert outputs[1]["passed"] is None

    summary_str = summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_str)

    assert summary["total"] == 2
    assert summary["success"] == 1
    assert summary["error"] == 1


def test_create_llm_client_fake_client_create():
    config = LLMConfig(
        provider="fake",
        model=None,
        api_key=None,
    )

    client = create_llm_client(config)

    assert isinstance(client, FakeLLMClient)


def test_create_llm_client_not_supported_client():
    config = LLMConfig(
        provider="not support provider",
        model="gpt-example",
        api_key="fake api key",
    )

    with pytest.raises(ValueError, match="LLMConfig provider not supported"):
        create_llm_client(config)


def test_create_llm_client_not_supported_client_main(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not support provider")
    monkeypatch.setenv("LLM_MODEL", "gpt-example")
    monkeypatch.setenv("LLM_API_KEY", "fake api key")

    with pytest.raises(ValueError, match="LLMConfig provider not supported"):
        asyncio.run(main())


def test_create_openai_llm_client():
    config = LLMConfig(
        provider="openai",
        model="gpt-example",
        api_key="fake api key",
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAILLMClient)


def test_create_openai_llm_client_without_input_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fake api key")

    with pytest.raises(ValueError, match="OpenAILLMClient requires a model"):
        asyncio.run(main())


class StubPromptChangeLLMClient:
    def __init__(self, response: str = "fake response"):
        self.prompt: str | None = None
        self.response = response

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_client_received_and_without_expected(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.jsonl"

    input_path.write_text("""{"name": "name str", "prompt": "prompt str"}
""", encoding="utf-8")

    client = StubPromptChangeLLMClient()

    asyncio.run(
        run_batch(
            BatchConfig(
                input_path=input_path,
                output_path=output_path,
                summary_path=summary_path,
            ),
            llm_client=client,
        )
    )

    assert client.prompt == "prompt str"


def test_task_without_name_or_prompt(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.jsonl"

    input_path.write_text("""{"prompt": "task without name"}
{"name": "", "prompt": "task with empty name"}
{"name": "task without prompt"}
{"name": "task with empty prompt", "prompt": ""}
""", encoding="utf-8")

    asyncio.run(
        run_batch(
            BatchConfig(
                input_path=input_path,
                output_path=output_path,
                summary_path=summary_path,
            ),
            llm_client=FakeLLMClient(),
        )
    )

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in output_lines]

    assert len(records) == 4

    assert any(
        record["status"] == "error"
        and "missing required field: 'name'" in record["error"]
        for record in records
    )

    assert any(
        record["status"] == "error"
        and "field content invalid 'name'" in record["error"]
        for record in records
    )

    assert any(
        record["status"] == "error"
        and "missing required field: 'prompt'" in record["error"]
        for record in records
    )

    assert any(
        record["status"] == "error"
        and "field content invalid 'prompt'" in record["error"]
        for record in records
    )


def test_process_task_handles_non_object():
    result = asyncio.run(
        process_task([], FakeLLMClient())
    )

    assert result.status == "error"
    assert result.name == "<unknown>"
    assert "JSON object" in result.error
