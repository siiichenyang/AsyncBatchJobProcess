import json
import logging


logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line_no, line in enumerate(input_file, start=1):
            try:
                input_data = json.loads(line)
                data.append(input_data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON line: {line_no}.")
    return data


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        for data in records:
            output_data = json.dumps(
                data, ensure_ascii=False)
            output_file.write(output_data + "\n")
