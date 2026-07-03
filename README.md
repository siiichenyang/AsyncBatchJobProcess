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
Output: `output.jsonl`

### Output Fields
```
name: The task name.
status: "success" or "error"
result: The simulated task output.
error: The error message, or null if the task succeeds.
latency_seconds: The task processing time in seconds.
retry_count: The number of retry attempts.
```