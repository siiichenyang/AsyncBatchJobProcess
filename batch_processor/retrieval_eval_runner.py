from dataclasses import dataclass
from collections.abc import Sequence

from batch_processor.retrieval_evals import (
    RetrievalEvalResult,
    evaluate_retrieval_case,
)
from batch_processor.retrieval_eval_io import RetrievalEvalInputRecord
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
