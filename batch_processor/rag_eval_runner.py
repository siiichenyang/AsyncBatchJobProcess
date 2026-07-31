from dataclasses import asdict, dataclass
from collections.abc import Sequence

from batch_processor.rag_evals import (
    RAGEvalResult,
    evaluate_rag_case,
)
from batch_processor.rag_eval_io import (
    RAGEvalInputRecord,
    load_rag_eval_cases,
)
from batch_processor.llm_client import LLMClient
from batch_processor.retrieval import Retriever
from batch_processor.jsonl_io import write_json, write_jsonl


@dataclass(frozen=True)
class RAGEvalRunRecord:
    line_number: int
    result: RAGEvalResult | None
    error: str | None


async def run_rag_eval_batch(
    records: Sequence[RAGEvalInputRecord],
    retriever: Retriever,
    llm_client: LLMClient,
    *,
    k: int,
) -> list[RAGEvalRunRecord]:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    results = []
    for record in records:
        if record.error is not None:
            results.append(
                RAGEvalRunRecord(
                    line_number=record.line_number,
                    result=None,
                    error=record.error,
                )
            )
            continue

        try:
            eval_result = await evaluate_rag_case(
                record.case,
                retriever,
                llm_client,
                top_k=k,
            )
            results.append(
                RAGEvalRunRecord(
                    line_number=record.line_number,
                    result=eval_result,
                    error=None,
                )
            )
        except Exception as exc:
            results.append(
                RAGEvalRunRecord(
                    line_number=record.line_number,
                    result=None,
                    error=f"{type(exc).__name__}: {exc}"
                )
            )

    return results


@dataclass(frozen=True)
class RAGEvalSummary:
    total: int
    evaluated: int
    errors: int
    k: int
    answer_pass: int
    citation_pass: int
    answer_pass_rate: float
    citation_pass_rate: float


def build_rag_eval_summary(
    records: Sequence[RAGEvalRunRecord],
    *,
    k: int,
) -> RAGEvalSummary:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    total = len(records)
    evaluated = 0
    errors = 0
    answer_pass = 0
    citation_pass = 0
    for record in records:
        if record.error is not None:
            errors += 1
            continue

        if record.result.top_k != k:
            raise ValueError(
                f"expect k={k} find k={record.result.top_k} line={record.line_number}"
            )

        evaluated += 1
        answer_pass += record.result.content_result.passed
        citation_pass += record.result.citation_result.passed

    answer_pass_rate = 0.0
    citation_pass_rate = 0.0
    if evaluated > 0:
        answer_pass_rate = answer_pass / evaluated
        citation_pass_rate = citation_pass / evaluated

    return RAGEvalSummary(
        total=total,
        evaluated=evaluated,
        errors=errors,
        k=k,
        answer_pass=answer_pass,
        citation_pass=citation_pass,
        answer_pass_rate=answer_pass_rate,
        citation_pass_rate=citation_pass_rate,
    )


@dataclass(frozen=True)
class RAGEvalReport:
    records: tuple[RAGEvalRunRecord, ...]
    summary: RAGEvalSummary


async def run_rag_eval_file(
    input_path: str,
    retriever: Retriever,
    llm_client: LLMClient,
    *,
    k: int,
) -> RAGEvalReport:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    input_records = load_rag_eval_cases(input_path)

    eval_records = await run_rag_eval_batch(
        input_records,
        retriever,
        llm_client,
        k=k,
    )

    summary = build_rag_eval_summary(eval_records, k=k)

    return RAGEvalReport(
        records=tuple(eval_records),
        summary=summary,
    )


def write_rag_eval_report(
    report: RAGEvalReport,
    *,
    output_path: str,
    summary_path: str,
) -> None:
    output_records = [
        asdict(record)
        for record in report.records
    ]

    write_jsonl(output_path, output_records)
    write_json(summary_path, asdict(report.summary))
