from dataclasses import dataclass

from batch_processor.rag_evals import RAGEvalCase
from batch_processor.jsonl_io import read_jsonl


@dataclass(frozen=True)
class RAGEvalInputRecord:
    line_number: int
    case: RAGEvalCase | None
    error: str | None


def load_rag_eval_cases(
    path: str,
) -> list[RAGEvalInputRecord]:
    data = read_jsonl(path)

    records = []
    for line_number, line in enumerate(data, start=1):
        if line.error is None:
            try:
                case = RAGEvalCase.from_dict(line.data)
                records.append(
                    RAGEvalInputRecord(
                        line_number=line_number,
                        case=case,
                        error=None,
                    )
                )
            except ValueError as exc:
                records.append(
                    RAGEvalInputRecord(
                        line_number=line_number,
                        case=None,
                        error=f"Invalid rag eval case line {line_number}: {exc}",
                    )
                )
        else:
            records.append(
                RAGEvalInputRecord(
                    line_number=line_number,
                    case=None,
                    error=line.error,
                )
            )

    return records
