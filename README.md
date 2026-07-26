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
development, but it is not a trained semantic embedding model.

`batch_processor/similarity.py` provides dependency-free cosine similarity for
equal-dimensional, non-zero vectors. Cosine similarity compares vector
direction rather than magnitude: aligned vectors score `1`, orthogonal vectors
score `0`, and opposite vectors score `-1`.

`batch_processor/vector_store.py` provides a small in-memory, exact vector
store. It keeps each precomputed embedding together with its `TextChunk`
metadata, scores every stored vector with cosine similarity, sorts matches by
descending score, and returns the requested top-k results. Embedding generation
remains separate from retrieval, so the store is synchronous and independent
of a particular embedding provider. This full-scan implementation is intended
for learning, local tests, and small datasets rather than large production
indexes.

`batch_processor/retrieval.py` is the asynchronous orchestration layer for the
retrieval pipeline. It chunks a `Document`, awaits an embedding for each chunk,
stores the resulting vectors, embeds a query, and delegates top-k ranking to the
vector store. File loading, chunking, embedding, and similarity calculation
remain separate components. Indexing is currently sequential, and this stage
returns retrieved context rather than generating an LLM answer. If embedding a
chunk fails, the error currently propagates after any earlier chunks have been
stored; retry-safe or idempotent re-indexing is not implemented yet.

#### Retrieval evaluation

`batch_processor/retrieval_eval_runner.py` evaluates a JSONL dataset against an
already-indexed retriever. Each input line contains a case name, its query, and
one or more relevant chunk references:

```json
{"name":"example","query":"search terms","relevant_chunks":[{"document_id":"doc-1","chunk_index":0}]}
```

`write_retrieval_eval_report` persists the resulting report as two files:

- The detail JSONL file contains one record per input line, in input order. Each
  record has `line_number`, `result`, and `error`. A successful `result` contains
  `name`, `query`, `k`, `retrieved_chunks`, `hit`, and `recall`; an error record
  has a null `result` and a diagnostic `error` string.
- The summary JSON file contains `total`, `evaluated`, `errors`, `k`,
  `hit_rate`, and `mean_recall`.

Invalid cases and retrieval failures remain in the detail output and contribute
to `total` and `errors`. They do not contribute to `evaluated`, so they are
excluded from the denominators of `hit_rate` and `mean_recall`.

A deterministic benchmark is stored in `data/rag/`. Its integration test loads
`backend_topics.txt`, indexes it with `chunk_size=6` and `overlap=0`, and runs
the cases in `retrieval_eval_cases.jsonl` with `k=1`. The expected aggregate
scores are a `0.75` hit rate and `0.625` mean recall. One case deliberately uses
synonyms that the local feature-hashing embedding cannot understand, making a
retrieval failure visible rather than hiding it behind an all-passing dataset.
The benchmark's chunk references are valid only for that fixed chunking
configuration; comparing another chunking strategy requires compatible ground
truth instead of reusing the chunk indexes blindly.

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
