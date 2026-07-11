# Async JSONL Batch Job Processor

### Introduction
This project reads tasks from a JSONL file, processes them, and writes the results to an output file.

### How to run
From the project root:
1. `python -m venv .venv`
2. `.\.venv\Scripts\python.exe -m batch_processor.main`

### How to test
From the project root:
1. `.\.venv\Scripts\python.exe -m pip install pytest`
2. `.\.venv\Scripts\python.exe -m pytest`

### Input & Output
Input: `input.jsonl`
Output: `output.jsonl` (generated locally and ignored by git)
Summary: `summary.json` (generated locally and ignored by git)

### Input Fields
```
name: Required task name.
description: Optional task description.
expected: Optional expected response used for exact-match evaluation.
```

### Output Fields
In `output.jsonl`
```
name: The task name.
status: "success" or "error"
result: The generated client output.
error: The error message, or null if the task succeeds.
latency_seconds: The task processing time in seconds.
retry_count: The number of retry attempts.
passed: true if output matches expected, false if it does not, null if expected is not provided or the task is not evaluated.
```
In `summary.json`
```
total: Total number of input tasks.
success: The number of successfully processed task.
error: The number of occurred error task.
evaluated: The number of evaluated task which has expected field input.
passed: The number of passed evaluation tasks.
failed: The number of failed to pass evaluation task.
pass_rate: passed / evaluated.
```

### Configuration
Config path: `batch_processor/config.py`

- `input_path`: Input JSONL file path.
- `output_path`: Output JSONL file path. The default `output.jsonl` is generated locally and ignored by git.
- `summary_path`: Summary JSON output path.
- `max_concurrency`: Maximum number of tasks processed concurrently.
- `timeout_seconds`: Timeout for each task attempt.
- `max_retries`: Number of retry attempts after the initial attempt.
