from dataclasses import dataclass
import json
import logging


logger = logging.getLogger(__name__)


@dataclass
class JsonRecord:
    data: object | None
    error: str | None


def read_jsonl(path: str) -> list[JsonRecord]:
    data = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line_no, line in enumerate(input_file, start=1):
            try:
                input_data = json.loads(line)
                data.append(JsonRecord(data=input_data, error=None))
            except json.JSONDecodeError as exc:
                err_str = f"Invalid JSON line {line_no}: {exc}"
                logger.warning(err_str)
                data.append(JsonRecord(data=None, error=err_str))
    return data


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        for data in records:
            output_data = json.dumps(
                data, ensure_ascii=False)
            output_file.write(output_data + "\n")


def write_json(path: str, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False)
