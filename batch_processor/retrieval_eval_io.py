from dataclasses import dataclass

from batch_processor.retrieval_evals import RetrievalEvalCase
from batch_processor.jsonl_io import read_jsonl


@dataclass(frozen=True)
class RetrievalEvalInputRecord:
    line_number: int
    case: RetrievalEvalCase | None
    error: str | None


def load_retrieval_eval_cases(
    path: str,
) -> list[RetrievalEvalInputRecord]:
    data = read_jsonl(path)

    records = []
    for line_number, line in enumerate(data, start=1):
        if line.error is None:
            try:
                case = RetrievalEvalCase.from_dict(line.data)
                records.append(
                    RetrievalEvalInputRecord(
                        line_number=line_number,
                        case=case,
                        error=None,
                    )
                )
            except ValueError as exc:
                records.append(
                    RetrievalEvalInputRecord(
                        line_number=line_number,
                        case=None,
                        error=f"Invalid retrieval eval case line {line_number}: {exc}",
                    )
                )
        else:
            records.append(
                RetrievalEvalInputRecord(
                    line_number=line_number,
                    case=None,
                    error=line.error,
                )
            )

    return records
