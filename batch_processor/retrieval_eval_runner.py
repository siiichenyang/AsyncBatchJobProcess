from dataclasses import dataclass
from collections.abc import Sequence

from batch_processor.retrieval_evals import (
    RetrievalEvalResult,
    evaluate_retrieval_case,
)
from batch_processor.retrieval_eval_io import (
    RetrievalEvalInputRecord,
    load_retrieval_eval_cases,
)
from batch_processor.retrieval import Retriever


@dataclass(frozen=True)
class RetrievalEvalRunRecord:
    line_number: int
    result: RetrievalEvalResult | None
    error: str | None


async def run_retrieval_eval_batch(
    records: Sequence[RetrievalEvalInputRecord],
    retriever: Retriever,
    *,
    k: int,
) -> list[RetrievalEvalRunRecord]:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    results = []
    for record in records:
        if record.error is not None:
            results.append(
                RetrievalEvalRunRecord(
                    line_number=record.line_number,
                    result=None,
                    error=record.error,
                )
            )
            continue

        try:
            eval_result = await evaluate_retrieval_case(
                record.case,
                retriever,
                k=k,
            )
            results.append(
                RetrievalEvalRunRecord(
                    line_number=record.line_number,
                    result=eval_result,
                    error=None,
                )
            )
        except Exception as exc:
            results.append(
                RetrievalEvalRunRecord(
                    line_number=record.line_number,
                    result=None,
                    error=f"{type(exc).__name__}: {exc}"
                )
            )

    return results


@dataclass(frozen=True)
class RetrievalEvalSummary:
    total: int
    evaluated: int
    errors: int
    k: int
    hit_rate: float
    mean_recall: float


def build_retrieval_eval_summary(
    records: Sequence[RetrievalEvalRunRecord],
    *,
    k: int,
) -> RetrievalEvalSummary:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    total = len(records)
    evaluated = 0
    errors = 0
    hit_sum = 0
    recall_sum = 0
    for record in records:
        if record.error is not None:
            errors += 1
            continue

        if record.result.k != k:
            raise ValueError(
                f"expect k={k} find k={record.result.k} line={record.line_number}"
            )

        evaluated += 1
        hit_sum += record.result.hit
        recall_sum += record.result.recall

    hit_rate = 0.0
    mean_recall = 0.0
    if evaluated > 0:
        hit_rate = hit_sum / evaluated
        mean_recall = recall_sum / evaluated

    return RetrievalEvalSummary(
        total=total,
        evaluated=evaluated,
        errors=errors,
        k=k,
        hit_rate=hit_rate,
        mean_recall=mean_recall,
    )


@dataclass(frozen=True)
class RetrievalEvalReport:
    records: tuple[RetrievalEvalRunRecord, ...]
    summary: RetrievalEvalSummary


async def run_retrieval_eval_file(
    input_path: str,
    retriever: Retriever,
    *,
    k: int,
) -> RetrievalEvalReport:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    input_records = load_retrieval_eval_cases(input_path)

    eval_records = await run_retrieval_eval_batch(
        input_records,
        retriever,
        k=k,
    )

    summary = build_retrieval_eval_summary(eval_records, k=k)

    return RetrievalEvalReport(
        records=tuple(eval_records),
        summary=summary,
    )
