# Async JSONL Batch Job Processor

### Introduction
This project reads evaluation tasks from a JSONL file, processes them with a
fake or OpenAI LLM client, and writes per-task results plus aggregate metrics.

The fake client is the default so the project can be run and tested without an
API key or network access.

### Setup

From the project root:

1. `python -m venv .venv`
2. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

### Run with the fake client

No environment variables are required:

```powershell
.\.venv\Scripts\python.exe -m batch_processor.main
```

### Run with the OpenAI client

Set the required variables in the current PowerShell session, then run the
processor:

```powershell
$env:LLM_PROVIDER = "openai"
$env:LLM_MODEL = "<model-id>"
$env:LLM_API_KEY = "<your-api-key>"
.\.venv\Scripts\python.exe -m batch_processor.main
```

OpenAI mode uses the asynchronous
[OpenAI Python SDK](https://github.com/openai/openai-python) and the Responses
API. It sends real requests and may consume API quota.

`.env.example` documents the supported variables, but this project does not
automatically load a `.env` file. Renaming the example file is therefore not
enough; export the variables in the shell as shown above. Never commit a real
API key or a populated `.env` file.

### How to test

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests use fake and stub clients and do not send network requests.

### RAG foundation

Local UTF-8 text files can be loaded through `batch_processor/documents.py`
into immutable `Document` records containing a document ID, source path, and
original text. File-system and decoding errors are left visible to the caller.

The word-based text chunker in `batch_processor/chunking.py` splits a loaded
document into fixed-size windows and can repeat words between adjacent chunks
through the `overlap` setting.

Each `TextChunk` records its document ID, sequential chunk index, text, and
word offsets using the half-open range `[start_word, end_word)`. This first
implementation counts whitespace-separated words rather than model tokenizer
tokens.

`batch_processor/embeddings.py` defines an asynchronous embedding-client
abstraction and a deterministic local implementation. The local client uses
feature hashing to turn case-folded, whitespace-separated tokens into a
fixed-size count vector. It is useful for repeatable tests and pipeline
development, but it is not a trained semantic embedding model. Vector search
is not implemented yet.

### Input & Output
Input: `input.jsonl`
Output: `output.jsonl` (generated locally and ignored by git)
Summary: `summary.json` (generated locally and ignored by git)

### Input Fields
```
name: Required task name.
prompt: Required task prompt.
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

LLM environment variables:

- `LLM_PROVIDER`: Client provider. Defaults to `fake`; `openai` enables the
  OpenAI client.
- `LLM_MODEL`: Model ID. Required when `LLM_PROVIDER=openai`.
- `LLM_API_KEY`: API key. Required when `LLM_PROVIDER=openai`.
