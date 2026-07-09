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

### Input Fields
```
name: Required task name.
description: Optional task description.
expected: Optional expected response uesed for exact-match evaluation.
```

### Output Fields
```
name: The task name.
status: "success" or "error"
result: The simulated task output.
error: The error message, or null if the task succeeds.
latency_seconds: The task processing time in seconds.
retry_count: The number of retry attempts.
passed: true if output matches expected, false if it does not, null if expected is not provided or the task is not evaluated.
```

### Configuration
Config path: `batch_processor/config.py`

- `input_path`: Input JSONL file path.
- `output_path`: Output JSONL file path. The default `output.jsonl` is generated locally and ignored by git.
- `max_concurrency`: Maximum number of tasks processed concurrently.
- `timeout_seconds`: Timeout for each task attempt.
- `max_retries`: Number of retry attempts after the initial attempt.
