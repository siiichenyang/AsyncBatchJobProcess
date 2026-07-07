import json
from batch_processor.jsonl_io import (
    JsonRecord,
    read_jsonl,
    write_jsonl,
)


def test_read_valid_records(tmp_path):
    input_path = tmp_path / "valid.jsonl"

    input_path.write_text(
        '{"name": "What\'s time now", "description": "Query current time"}\n'
        '{"name": "Draw a picture about flower", "description": "Sunflower."}\n',
        encoding="utf-8",
    )

    records = read_jsonl(str(input_path))

    assert records == [
        JsonRecord(data={"name": "What's time now",
                   "description": "Query current time"}, error=None),
        JsonRecord(data={"name": "Draw a picture about flower",
                   "description": "Sunflower."}, error=None),
    ]


def test_read_handle_invalid_records(tmp_path):
    input_path = tmp_path / "invalid.jsonl"

    input_path.write_text(
        '{"description": "A task without name."}\n'
        'This is not a valid json\n',
        encoding="utf-8",
    )

    records = read_jsonl(str(input_path))

    assert len(records) == 2
    assert records[0] == JsonRecord(
        data={"description": "A task without name."}, error=None)
    assert records[1].error is not None
    assert "Invalid JSON line" in records[1].error


def test_write(tmp_path):
    output_path = tmp_path / "output.jsonl"

    write_jsonl(
        str(output_path),
        [{"name": "Write a poem.", "status": "success"}]
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "name": "Write a poem.", "status": "success"
    }


def test_read_handle_blank_line(tmp_path):
    input_path = tmp_path / "input.jsonl"

    input_path.write_text(
        '{"name": "task a", "description": "delete a file."}\n'
        '\n',
        encoding="utf-8"
    )

    records = read_jsonl(str(input_path))

    assert len(records) == 2
    assert records[0] == JsonRecord(
        data={"name": "task a", "description": "delete a file."}, error=None)
    assert records[1].error is not None
    assert "Invalid JSON line" in records[1].error
