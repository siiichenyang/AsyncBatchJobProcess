import json
import pytest
from batch_processor.jsonl_io import (
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
        {"name": "What's time now", "description": "Query current time"},
        {"name": "Draw a picture about flower", "description": "Sunflower."}
    ]


def test_read_handle_invalid_records(tmp_path):
    input_path = tmp_path / "invalid.jsonl"

    input_path.write_text(
        '{"description": "A task without name."}\n'
        'This is not a valid json\n'
    )

    records = read_jsonl(str(input_path))

    assert records == [
        {"description": "A task without name."},
    ]


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

    assert records == [
        {"name": "task a", "description": "delete a file."},
    ]
