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

#### RAG prompt construction

`batch_processor/rag.py` converts ranked retrieval results into an immutable
`RAGPrompt`. Retrieved chunks remain in ranking order and are labeled as
`[document_id#chunk_index]`, allowing a later model answer to cite the exact
retrieval unit. The structured `sources` tuple retains each full `SearchResult`,
including its similarity score, for tracing and debugging even though scores are
not included in the model prompt.

The prompt instructs the model to answer only from retrieved context, cite the
provided labels, treat retrieved text as reference material rather than
instructions, and state when the context is missing or insufficient. An empty
retrieval result is represented explicitly as `No context was retrieved.`

`answer_rag_query` completes the asynchronous RAG path using a pre-indexed
retriever: it retrieves top-k chunks, builds the prompt, calls the injected
`LLMClient`, and returns an immutable `RAGAnswer`. The result retains the query,
generated answer, exact model prompt, and ranked sources so the run can be
traced. Query and top-k validation happen before either dependency is awaited;
retrieval and model errors remain visible to the caller. The query path itself
does not enforce citation correctness.

`batch_processor/rag_evals.py` provides a deterministic first layer of citation
evaluation. It extracts bracketed labels from a generated answer, preserves
their occurrence order, and compares them with labels derived from the answer's
ranked sources. A citation check passes only when at least one label is present
and every cited label belongs to those sources; missing or fabricated labels
fail. This verifies citation presence and source membership only. It does not
prove that the cited chunk supports the surrounding claim, so citation
faithfulness remains a separate semantic evaluation problem.

The same module also provides `evaluate_answer_content` as a deterministic
answer-quality baseline. It strips each required phrase and performs a
case-insensitive substring check with `casefold()`; every phrase must appear for
the result to pass. The result preserves the required phrases and any missing
phrases in input order for debugging. This is not a semantic-equivalence check:
valid paraphrases can fail, while an incidental substring match can pass.

#### End-to-end RAG evaluation

`data/rag/rag_answer_eval_cases.jsonl` provides answer-level cases. Each JSONL
record contains a case name, the retrieval query, and the phrases that a valid
answer must contain:

```json
{"name":"async case","query":"async coroutine event loop","required_phrases":["coroutine","event loop"]}
```

`batch_processor/rag_eval_io.py` validates these records while retaining input
line numbers and per-line parsing errors. For each valid case,
`evaluate_rag_case` runs the normal retrieval and generation path exactly once,
then evaluates answer content and citations independently. Its structured result
keeps the generated answer, exact prompt, ranked source chunks, required or
missing phrases, and extracted or invalid citations for later diagnosis.

`batch_processor/rag_eval_runner.py` processes cases sequentially and isolates
retrieval, model, and evaluation failures to their individual records. The
detail report is JSONL with `line_number`, `result`, and `error`; the summary JSON
contains `total`, `evaluated`, `errors`, `k`, `answer_pass`, `citation_pass`,
`answer_pass_rate`, and `citation_pass_rate`. Error records remain visible in
`total` but are excluded from `evaluated` and therefore from both rate
denominators. `write_rag_eval_report` writes the detail and summary files.

#### Retrieval evaluation

`batch_processor/retrieval_eval_runner.py` evaluates a JSONL dataset against an
already-indexed retriever. Each input line contains a case name, its query, and
one or more relevant spans in the original document:

```json
{"name":"example","query":"search terms","relevant_spans":[{"document_id":"doc-1","start_word":4,"end_word":8}]}
```

Span offsets use the half-open interval `[start_word, end_word)`. A retrieved
chunk matches a span when both refer to the same document and their word ranges
have a non-empty overlap. Because the labels describe source positions rather
than generated chunk indexes, the same ground truth can be reused across
different chunk sizes and overlaps. The current any-overlap rule is intentionally
simple and may count a chunk that contains only a small part of the relevant
passage.

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
Those scores describe the baseline chunking configuration; future configurations
reuse the spans but may produce different retrieval scores.

`batch_processor/retrieval_eval_comparison.py` compares chunking strategies
sequentially while holding the document, evaluation cases, embedding client, and
`k` constant. Each strategy receives a fresh in-memory vector store so indexed
chunks from one run cannot contaminate another. The current deterministic
benchmark produces:

| Strategy | Chunk size | Overlap | Hit rate | Mean recall |
| --- | ---: | ---: | ---: | ---: |
| `no-overlap` | 6 | 0 | 0.75 | 0.625 |
| `overlap-2` | 6 | 2 | 0.75 | 0.75 |

Overlap improves recall for the query whose two relevant spans fall on opposite
sides of a baseline chunk boundary: one overlapping chunk covers both spans.
This small deterministic result demonstrates how to run a controlled comparison;
it does not establish that overlap is always better. Overlap also creates more
chunks and therefore increases indexing and storage work.

#### Common RAG failure modes

The separate retrieval, content, and citation signals help localize failures:

| Observed signal | Likely interpretation |
| --- | --- |
| Low hit rate or recall | The retriever did not provide enough relevant context; chunking, query wording, embeddings, or `k` may be responsible. |
| Good retrieval but low answer pass rate | Relevant context was available, but the model omitted required information or expressed it differently from the deterministic expectation. |
| Content passes but citation evaluation fails | The answer contains the expected information but omitted a source label or fabricated one that was not retrieved. |
| Citation evaluation passes | Every cited label was retrieved, but this alone does not prove that the cited text supports the associated claim. |

Prompt grounding remains a model instruction rather than a hard guarantee, so a
model can still invent unsupported details. The required-phrase evaluator is
also intentionally limited: paraphrases may produce false failures and
incidental substring matches may produce false passes. These cases require trace
inspection or a later semantic or model-based evaluator.

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
